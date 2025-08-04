import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from tqdm import tqdm
from itertools import product
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# DINA 模型相关函数
def compute_eta(Q, A):
    kowns = np.sum(Q * Q, axis=0)
    cross = np.dot(A, Q)
    eta = np.ones(shape=(A.shape[0], Q.shape[1]))
    eta[cross < kowns] = 0
    return eta


def compute_propa(eta, s, g):
    propa = (g ** (1 - eta)) * ((1 - s) ** eta)
    propa[propa <= 0] = 1e-10
    propa[propa >= 1] = 1 - 1e-10
    return propa


def compute_eta_optimized(Q, A):
    """
    优化后的eta计算，使用矩阵运算加速
    """
    # Q: (skills×items), A: (patterns×skills)
    # 计算每个知识点在每个题目上的要求平方和
    kowns = np.einsum('ij,ij->j', Q, Q)  # 等价于np.sum(Q*Q, axis=0)

    # 计算每个模式在每个题目上的掌握情况
    cross = np.einsum('ik,jk->ij', A, Q)  # 等价于np.dot(A, Q)

    # 向量化比较
    eta = (cross >= kowns).astype(np.uint8)
    return eta


def compute_propa_optimized(eta, s, g):
    """
    优化后的propa计算，避免中间变量
    """
    # 向量化计算
    propa = np.where(eta, 1 - s, g)

    # 数值稳定性处理
    np.clip(propa, 1e-10, 1 - 1e-10, out=propa)
    return propa

def compute_gamma(X, pi, propa):
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    gamma = np.exp(np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi)
    gamma_sum = np.sum(gamma, axis=1)
    gamma = (gamma.T / gamma_sum).T
    return gamma


def compute_theta(X, gamma, eta):
    I0 = np.dot(gamma, 1 - eta)
    I1 = np.dot(gamma, eta)
    R0 = I0 * X
    R1 = I1 * X
    I0 = np.sum(I0, axis=0)
    I1 = np.sum(I1, axis=0)
    R0 = np.sum(R0, axis=0)
    R1 = np.sum(R1, axis=0)
    I0[I0 <= 0] = 1e-15
    I1[I1 <= 0] = 1e-15
    g = R0 / I0
    s = (I1 - R1) / I1
    pi = np.sum(gamma, axis=0) / gamma.shape[0]
    pi[pi <= 0] = 1e-15
    pi[pi >= 1] = 1 - 1e-15
    return pi, s, g


# def em(X, Q, maxIter=1000, tol=1e-3, prior=None):
#     n_stu = X.shape[0]
#     n_qus = X.shape[1]
#     n_kno = Q.shape[0]
#
#     # 使用更紧凑的数据类型
#     g = np.random.random(n_qus) * 0.25
#     s = np.random.random(n_qus) * 0.25
#
#     # 生成A_all时使用uint8类型节省内存
#     A_all = np.array(list(product([0, 1], repeat=n_kno)), dtype=np.uint8)
#
#     if prior is None:
#         pi = np.ones(A_all.shape[0]) / A_all.shape[0]
#     else:
#         pi = prior
#
#     # 预分配内存
#     eta = np.zeros((A_all.shape[0], Q.shape[1]), dtype=np.uint8)
#     propa = np.zeros_like(eta, dtype=np.float32)
#
#     for t in range(maxIter):
#         # 分块计算eta和propa
#         chunk_size = 256  # 每次处理256种模式
#         for i in range(0, A_all.shape[0], chunk_size):
#             chunk = A_all[i:i + chunk_size]
#             eta_chunk = compute_eta(Q, chunk)
#             propa_chunk = compute_propa(eta_chunk, s, g)
#             eta[i:i + chunk_size] = eta_chunk
#             propa[i:i + chunk_size] = propa_chunk
#
#         # 分块计算gamma
#         gamma = np.zeros((n_stu, A_all.shape[0]), dtype=np.float32)
#         for j in range(0, n_stu, 1000):  # 每次处理1000个学生
#             X_chunk = X[j:j + 1000]
#             log_pj = np.log(propa)
#             log_qj = np.log(1 - propa)
#             log_pi = np.log(pi)
#             gamma_chunk = np.exp(np.dot(X_chunk, log_pj.T) +
#                                  np.dot((1 - X_chunk), log_qj.T) +
#                                  log_pi)
#             gamma_sum = np.sum(gamma_chunk, axis=1)
#             gamma[j:j + 1000] = (gamma_chunk.T / gamma_sum).T
#
#         pi_t, s_t, g_t = compute_theta(X, gamma, eta)
#         update = max(np.max(np.abs(pi_t - pi)),
#                      np.max(np.abs(g_t - g)),
#                      np.max(np.abs(s_t - s)))
#
#         if update < tol:
#             return pi_t, g_t, s_t, gamma
#
#         if prior is None:
#             pi = pi_t
#         g = g_t
#         s = s_t
#
#     return pi, g, s, gamma
def em(X, Q, maxIter=200, tol=1e-5, prior=None):
    import cupy as cp  # ✅ GPU加速替代numpy

    n_stu = X.shape[0]
    n_qus = X.shape[1]
    n_kno = Q.shape[0]

    # 初始化参数
    g = np.random.random(n_qus) * 0.25
    s = np.random.random(n_qus) * 0.25

    # 所有知识状态组合
    A_all = np.array(list(product([0, 1], repeat=n_kno)), dtype=np.uint8)

    if prior is None:
        pi = np.ones(A_all.shape[0]) / A_all.shape[0]
    else:
        pi = prior

    # 预计算所有 eta 和 propa（在GPU上做）
    eta = compute_eta(Q, A_all)
    propa = compute_propa(eta, s, g)

    # 放到GPU上
    X_gpu = cp.asarray(X, dtype=cp.float32)
    eta_gpu = cp.asarray(eta)
    propa_gpu = cp.asarray(propa)
    log_pi_gpu = cp.log(cp.asarray(pi))
    log_pj_gpu = cp.log(propa_gpu)
    log_qj_gpu = cp.log(1 - propa_gpu)

    # 初始化 gamma
    gamma = cp.zeros((n_stu, A_all.shape[0]), dtype=cp.float32)

    stu_chunk_size = 512
    mode_chunk_size = 256
    for t in range(maxIter):
        # 更新 log_pj、log_qj（每轮参数更新）
        propa_gpu = cp.where(eta_gpu, 1 - cp.asarray(s), cp.asarray(g))
        cp.clip(propa_gpu, 1e-10, 1 - 1e-10, out=propa_gpu)
        log_pj_gpu = cp.log(propa_gpu)
        log_qj_gpu = cp.log(1 - propa_gpu)

        gamma.fill(0.0)

        for i in range(0, n_stu, stu_chunk_size):
            X_chunk = X_gpu[i:i + stu_chunk_size]

            for j in range(0, A_all.shape[0], mode_chunk_size):
                lpj = log_pj_gpu[j:j + mode_chunk_size]
                lqj = log_qj_gpu[j:j + mode_chunk_size]
                lpi = log_pi_gpu[j:j + mode_chunk_size]

                part1 = X_chunk @ lpj.T
                part2 = (1 - X_chunk) @ lqj.T
                gamma_chunk = cp.exp(part1 + part2 + lpi)

                gamma[i:i + stu_chunk_size, j:j + mode_chunk_size] = gamma_chunk

        gamma_sum = cp.sum(gamma, axis=1, keepdims=True)
        gamma = gamma / gamma_sum

        # 计算 theta
        gamma_cpu = cp.asnumpy(gamma)
        eta_cpu = eta
        pi_t, s_t, g_t = compute_theta(X, gamma_cpu, eta_cpu)

        # 收敛判断
        update = max(
            np.max(np.abs(pi_t - pi)),
            np.max(np.abs(g_t - g)),
            np.max(np.abs(s_t - s))
        )
        if update < tol:
            return pi_t, g_t, s_t, gamma_cpu

        # 更新参数
        if prior is None:
            pi = pi_t
        g = g_t
        s = s_t
        log_pi_gpu = cp.log(cp.asarray(pi))

    return pi, g, s, cp.asnumpy(gamma)

def solve(gamma, n_kownlege):
    A_all = np.array(list(product([0, 1], repeat=n_kownlege)))
    A_idx = np.argmax(gamma, axis=1)
    return A_all[A_idx], A_idx


def joint_loglike(X, Q, s, g, pi,gamma):
    A_all = np.array(list(product([0, 1], repeat=Q.shape[0])))
    eta = compute_eta(Q, A_all)
    propa = compute_propa(eta, s, g)
    log_pj = np.log(propa)
    log_qj = np.log(1 - propa)
    log_pi = np.log(pi)
    L = np.dot(X, log_pj.T) + np.dot((1 - X), log_qj.T) + log_pi
    L = L * gamma
    return np.sum(L)


def evaluate(X, Q, priors):
    n = len(priors)
    results = []
    for i in range(n):
        pi, g, s, gamma = em(X, Q, maxIter=100, tol=1e-6, prior=priors[i])
        A, A_idx = solve(gamma, Q.shape[0])
        results.append((pi, g, s, gamma, A, A_idx))
    return results


def score(results, A_test, A_test_idx, s, g):
    pmrs = []
    mmrs = []
    sr2s = []
    gr2s = []
    smaes = []
    gmaes = []
    for i in range(len(results)):
        pmrs.append(accuracy_score(results[i][5], A_test_idx))
        mmrs.append(accuracy_score(results[i][4].flatten(), A_test.flatten()))
        sr2s.append(r2_score(results[i][2], s))
        gr2s.append(r2_score(results[i][1], g))
        smaes.append(mean_absolute_error(results[i][2], s))
        gmaes.append(mean_absolute_error(results[i][1], g))
    df_result = pd.DataFrame({"pmr": pmrs, "mmr": mmrs, "s_r2": sr2s, "g_r2": gr2s, "s_mae": smaes, "g_mae": gmaes})
    return df_result


# 修改get_priors函数中的先验设置
def get_priors(A_all, p_know=0.5, p_know_list=None):
    # 先验1: 完全均匀分布
    prior1 = np.ones(A_all.shape[0]) / A_all.shape[0]

    # 先验2: 基于整体掌握概率
    prior2 = np.prod((p_know ** A_all) * ((1 - p_know) ** (1 - A_all)), axis=1)
    prior2 /= prior2.sum()

    # 先验3: 基于知识点差异的掌握概率（需要调整p_know_list）
    if p_know_list is None:
        p_know_list = np.linspace(0.3, 0.7, A_all.shape[1])  # 自动生成梯度概率
    prior3 = np.ones(A_all.shape[0])
    for k in range(A_all.shape[1]):
        prior3 *= (p_know_list[k] ** A_all[:, k]) * ((1 - p_know_list[k]) ** (1 - A_all[:, k]))
    prior3 /= prior3.sum()

    return [prior1, prior2, prior3]

# 改进后的Q矩阵构建（内存优化版）
def build_q_matrix(output_matrix, group_qs_ids, all_qs_ids, max_knowledge):
    """
    优化版Q矩阵构建函数
    参数:
        output_matrix: 原始知识点矩阵 (题目ID为行索引)
        group_qs_ids: 当前组的题目ID列表
        all_qs_ids: 所有有效题目ID集合
        max_knowledge: 最大知识点数限制
    返回:
        Q: 优化后的Q矩阵 (知识点×题目)
    """
    # --- 步骤1: 筛选有效知识点 ---
    # 统计每个知识点在output_matrix中的总覆盖率（排除当前组不存在的题目）
    knowledge_coverage = np.zeros(output_matrix.shape[1])
    for j, qs_id in enumerate(output_matrix.index):
        if qs_id in all_qs_ids:
            knowledge_coverage += output_matrix.loc[qs_id].values

    # 选择覆盖题目最多的前max_knowledge个知识点
    top_k_indices = np.argsort(-knowledge_coverage)[:max_knowledge]
    top_k_indices = sorted(top_k_indices)  # 保持原始顺序

    # --- 步骤2: 构建Q矩阵（保留原有题目ID匹配逻辑）---
    Q = np.zeros((len(top_k_indices), len(group_qs_ids)), dtype=int)

    missing_qs = 0
    for j, qs_id in enumerate(group_qs_ids):
        if qs_id in all_qs_ids:
            # 只选取top_k_indices对应的知识点列
            Q[:, j] = output_matrix.loc[qs_id].values[top_k_indices]
        else:
            missing_qs += 1
            Q[:, j] = 0  # 保持原有逻辑

    if missing_qs > 0:
        print(f"警告: 共 {missing_qs} 个题目未在output_matrix中找到，已填充为0")

    # --- 步骤3: 后处理验证 ---
    # 移除全零知识点（如果因题目缺失导致）
    non_zero_knowledge = np.where(Q.sum(axis=1) > 0)[0]
    Q = Q[non_zero_knowledge, :]

    # 确保每个知识点至少覆盖3题（否则移除）
    valid_knowledge = np.where(Q.sum(axis=1) >= 3)[0]
    if len(valid_knowledge) < Q.shape[0]:
        print(f"优化: 移除 {Q.shape[0] - len(valid_knowledge)} 个低覆盖知识点")
        Q = Q[valid_knowledge, :]

    print(f"最终Q矩阵形状: {Q.shape} (知识点×题目)")
    print(f"知识点覆盖统计: 平均每题 {Q.sum(axis=0).mean():.2f} 个知识点")

    return Q

def get_priors(A_all, p_know, p_know_list):
    prior1 = np.ones(A_all.shape[0]) / A_all.shape[0]
    prior2 = np.ones(A_all.shape[0])
    prior3 = np.ones(A_all.shape[0])
    n_kno = A_all.shape[1]
    # 扩展 p_know_list 以匹配 n_kno
    if len(p_know_list) < n_kno:
        p_know_list = p_know_list * (n_kno // len(p_know_list) + 1)
    p_know_list = p_know_list[:n_kno]  # 裁剪到 n_kno
    for l in range(A_all.shape[0]):
        for k in range(A_all.shape[1]):
            p = p_know_list[k]
            prior2[l] *= (p_know ** A_all[l, k] * (1 - p_know) ** (1 - A_all[l, k]))
            prior3[l] *= (p ** A_all[l, k] * (1 - p) ** (1 - A_all[l, k]))
    return [prior1, prior2, prior3, None]


def run_convergence_loop(X_original, Q, max_cycles=200, em_max_iter=200, tol=1e-6, prior=None):
    """
    执行自洽性检验的完整循环流程
    参数：
        X_original: 原始答题矩阵 (students × items)
        Q: 知识点关联矩阵 (items × skills)
        max_cycles: 最大循环轮次
        em_max_iter: 每次EM算法的最大迭代次数
        tol: 收敛阈值
        prior: 先验分布
    返回：
        history: 包含每轮结果的字典
        figs: 可视化图表对象
    """
    # 初始化记录
    history = {
        'cycle': [],
        'A': [],
        'X_sim': [],
        'params': {'g': [], 's': [], 'pi': []},
        'delta_A': [],
        'delta_X': [],
        'accuracy': [],
        'NLL': []
    }

    # 初始状态
    X_current = X_original.copy()
    A_prev = None

    # 主循环
    for cycle in range(max_cycles):
        print(f"\n=== Cycle {cycle + 1}/{max_cycles} ===")

        # 1. 拟合当前数据
        pi, g, s, gamma = em(X_current, Q, maxIter=em_max_iter, tol=tol, prior=prior)
        A_current, _ = solve(gamma, Q.shape[0])

        # 2. 计算变化量
        delta_A = np.mean(np.abs(A_current - A_prev)) if A_prev is not None else np.nan
        delta_X = np.mean(np.abs(X_current - X_original)) if cycle > 0 else np.nan

        # 3. 生成模拟数据
        eta = compute_eta(Q, A_current)
        propa = compute_propa(eta, s, g)
        X_sim = np.random.binomial(1, propa)

        # 4. 计算评估指标
        accuracy = np.mean(X_sim == X_original)
        nll = -joint_loglike(X_original, Q, s, g, pi, gamma)

        # 5. 记录结果
        history['cycle'].append(cycle)
        history['A'].append(A_current)
        history['X_sim'].append(X_sim)
        history['params']['g'].append(g)
        history['params']['s'].append(s)
        history['params']['pi'].append(pi)
        history['delta_A'].append(delta_A)
        history['delta_X'].append(delta_X)
        history['accuracy'].append(accuracy)
        history['NLL'].append(nll)

        # 6. 检查收敛
        if cycle > 1 and delta_A < tol and delta_X < tol:
            print(f"提前收敛于轮次 {cycle + 1}")
            break

        # 7. 更新参考
        A_prev = A_current.copy()
        X_current = X_sim.copy()

    # 可视化
    figs = plot_convergence(history, Q.shape[1])

    return history, figs


def plot_convergence(history, n_skills):
    """生成收敛过程可视化图表"""
    figs = []

    # 图1：参数变化趋势
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 子图1：Delta变化
    ax1.plot(history['cycle'], history['delta_A'], 'o-', label='ΔA (属性模式变化)')
    ax1.plot(history['cycle'], history['delta_X'], 's-', label='ΔX (答题矩阵变化)')
    ax1.set_ylabel('变化量')
    ax1.set_yscale('log')
    ax1.legend()

    # 子图2：参数变化
    ax2.plot(history['cycle'], np.mean(history['params']['g'], axis=1), '^-', label='平均猜测率')
    ax2.plot(history['cycle'], np.mean(history['params']['s'], axis=1), 'v-', label='平均失误率')
    ax2.set_xlabel('循环轮次')
    ax2.set_ylabel('参数值')
    ax2.legend()

    fig1.suptitle('自洽性检验收敛过程')
    figs.append(fig1)

    # 图2：知识点掌握模式变化
    if n_skills <= 10:  # 避免过多知识点导致图表混乱
        fig2, ax = plt.subplots(figsize=(12, 6))
        skill_coverage = np.array([A.mean(axis=0) for A in history['A']])

        for k in range(n_skills):
            ax.plot(history['cycle'], skill_coverage[:, k],
                    label=f'K{k + 1}', marker='o', linestyle='--')

        ax.set_xlabel('循环轮次')
        ax.set_ylabel('掌握比例')
        ax.set_title('各知识点掌握比例变化')
        ax.legend()
        figs.append(fig2)

    return figs


import numpy as np
import pandas as pd


def generate_synthetic_data(n_students=500, n_items=100, n_skills=5, seed=42, q_matrix=None):
    """生成模拟数据，支持传入自定义Q矩阵"""
    np.random.seed(seed)

    # 1. 生成或使用传入的Q矩阵
    if q_matrix is None:
        Q = generate_complex_q(n_skills, n_items)
    else:
        Q = q_matrix
        n_skills, n_items = Q.shape

    # 2. 生成学生知识掌握模式（带相关性）
    skill_means = np.linspace(0.4, 0.7, n_skills)
    cov = np.eye(n_skills) * 0.7 + 0.3
    A = np.random.binomial(1, np.clip(np.random.multivariate_normal(
        skill_means, cov, n_students), 0.1, 0.9))

    # 3. 生成题目参数
    g = np.random.beta(2, 5, n_items) * 0.3
    s = np.random.beta(2, 5, n_items) * 0.3
    hard_items = np.random.choice(n_items, n_items // 10, replace=False)
    s[hard_items] = np.random.uniform(0.25, 0.4, len(hard_items))

    # 4. 生成作答数据
    eta = (A @ Q) >= Q.sum(axis=0)
    p = g * (1 - eta) + (1 - s) * eta
    X = np.random.binomial(1, p)

    return X, Q, A, g, s
def generate_complex_q(n_skills=5, n_items=100, max_skills_per_item=4):
    """生成带交叉知识点的Q矩阵"""
    Q = np.zeros((n_skills, n_items))

    # 确保每个知识点至少关联20题
    for k in range(n_skills):
        base_items = np.random.choice(n_items, 20, replace=False)
        Q[k, base_items] = 1

    # 随机添加交叉知识点
    for j in range(n_items):
        current_skills = np.where(Q[:, j])[0]
        if len(current_skills) == 0:  # 确保每题至少1个知识点
            Q[np.random.randint(n_skills), j] = 1
            current_skills = np.where(Q[:, j])[0]

        # 添加额外知识点（40%概率）
        if np.random.rand() < 0.4 and len(current_skills) < max_skills_per_item:
            available_skills = [s for s in range(n_skills) if s not in current_skills]
            if available_skills:
                Q[np.random.choice(available_skills), j] = 1

    return Q
def add_noise(X, noise_level=0.05):
    """添加随机噪声到作答矩阵"""
    noise_mask = np.random.rand(*X.shape) < noise_level
    return np.where(noise_mask, 1 - X, X)
def train_and_predict(X_full, Q_full, true_A, true_g=None, true_s=None, test_size=30):
    """参数说明：
    true_A: 真实知识点掌握模式 (n_students, n_skills)
    true_g/true_s: 可选的真实题目参数
    """
    # 1. 分割题目索引
    item_indices = np.arange(X_full.shape[1])
    np.random.shuffle(item_indices)
    test_indices = item_indices[:test_size]
    train_indices = item_indices[test_size:]

    # 2. 分割数据
    X_train = X_full[:, train_indices]
    Q_train = Q_full[:, train_indices]
    X_test = X_full[:, test_indices]
    Q_test = Q_full[:, test_indices]

    # 3. 训练模型
    pi_train, g_train, s_train, gamma_train = em(X_train, Q_train, maxIter=200, tol=1e-5)
    A_pred, _ = solve(gamma_train, Q_train.shape[0])

    # 4. 预测测试集
    eta_test = compute_eta(Q_test, A_pred)

    # 处理g_test和s_test
    if true_g is not None and true_s is not None:
        g_test = true_g[test_indices]
        s_test = true_s[test_indices]
    else:
        g_test = np.full(Q_test.shape[1], np.mean(g_train))
        s_test = np.full(Q_test.shape[1], np.mean(s_train))

    propa_test = compute_propa(eta_test, s_test, g_test)
    X_pred = (propa_test > 0.5).astype(int)

    # 5. 评估
    accuracy = np.mean(X_pred == X_test)
    print(f"预测准确率: {accuracy:.2%}")

    # 返回结果
    return {
        'X_test': X_test,
        'X_pred': X_pred,
        'A_pred': A_pred,
        'true_A': true_A,  # 完整的真实知识点模式
        'test_indices': test_indices,
        'accuracy': accuracy
    }
def enhanced_evaluation(X_test, X_pred, A_true, A_pred, Q_test):
    """
    增强的评估函数（完整可运行版本）
    参数:
        X_test: 真实测试集作答矩阵 (n_students × n_test_items)
        X_pred: 预测的作答矩阵 (n_students × n_test_items)
        A_true: 真实知识点掌握模式 (n_students × n_skills)
        A_pred: 预测的知识点掌握模式 (n_students × n_skills)
        Q_test: 测试题的Q矩阵 (n_skills × n_test_items)
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt

    # 1. 题目层面评估
    real_difficulty = 1 - X_test.mean(axis=0)
    pred_difficulty = 1 - X_pred.mean(axis=0)
    diff_corr = np.corrcoef(real_difficulty, pred_difficulty)[0, 1]

    # 2. 学生层面评估
    auc_scores = []
    for i in range(A_true.shape[1]):  # 每个知识点的AUC
        auc_scores.append(roc_auc_score(A_true[:, i], A_pred[:, i]))

    # 3. 模型拟合度（对数似然）
    # 避免log(0)的情况
    epsilon = 1e-10
    X_pred_clip = np.clip(X_pred, epsilon, 1 - epsilon)
    log_likelihood = np.sum(X_test * np.log(X_pred_clip)) + \
                     np.sum((1 - X_test) * np.log(1 - X_pred_clip))

    print("\n===== 增强评估 ======")
    print(f"题目难度相关性: {diff_corr:.3f}")
    print(f"知识点AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    print(f"对数似然: {log_likelihood:.1f}")

    # 4. 可视化
    plt.figure(figsize=(12, 4))

    # 子图1：题目难度对比
    plt.subplot(131)
    plt.scatter(real_difficulty, pred_difficulty, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("真实难度")
    plt.ylabel("预测难度")
    plt.title(f"题目难度 (r={diff_corr:.2f})")

    # 子图2：知识点AUC分布
    plt.subplot(132)
    plt.boxplot(auc_scores)
    plt.xticks([1], ["知识点AUC"])
    plt.title(f"AUC均值={np.mean(auc_scores):.2f}")

    # 子图3：知识点掌握率偏差
    plt.subplot(133)
    bias = A_pred.mean(axis=0) - A_true.mean(axis=0)
    plt.bar(range(len(bias)), bias)
    plt.xlabel("知识点编号")
    plt.ylabel("预测偏差")
    plt.title("知识点掌握率偏差")

    plt.tight_layout()
    plt.show()

    return {
        'difficulty_correlation': diff_corr,
        'skill_auc_mean': np.mean(auc_scores),
        'skill_auc_std': np.std(auc_scores),
        'log_likelihood': log_likelihood
    }
def evaluate_parameter_recovery(true_g, true_s, est_g, est_s, test_indices):
    """评估题目参数恢复情况"""
    g_rmse = np.sqrt(np.mean((true_g[test_indices] - est_g) ** 2))
    s_rmse = np.sqrt(np.mean((true_s[test_indices] - est_s) ** 2))
    g_corr = np.corrcoef(true_g[test_indices], est_g)[0, 1]
    s_corr = np.corrcoef(true_s[test_indices], est_s)[0, 1]

    print("\n==== 题目参数恢复 ====")
    print(f"g参数 RMSE: {g_rmse:.4f} (相关性: {g_corr:.3f})")
    print(f"s参数 RMSE: {s_rmse:.4f} (相关性: {s_corr:.3f})")

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(true_g[test_indices], est_g, alpha=0.6)
    plt.title(f"猜测参数恢复 (r={g_corr:.3f})")

    plt.subplot(122)
    plt.scatter(true_s[test_indices], est_s, alpha=0.6)
    plt.title(f"失误参数恢复 (r={s_corr:.3f})")
    plt.tight_layout()

    return {'g_rmse': g_rmse, 's_rmse': s_rmse, 'g_corr': g_corr, 's_corr': s_corr}
def stability_test(n_runs=5):
    """多随机种子稳定性测试"""
    results = []
    for seed in range(42, 42 + n_runs):
        np.random.seed(seed)
        X, Q, true_A, true_g, true_s = generate_synthetic_data(seed=seed)
        X = add_noise(X, 0.05)  # 添加噪声

        res = train_and_predict(X, Q, true_A, true_g, true_s)
        metrics = enhanced_evaluation(res['X_test'], res['X_pred'],
                                      res['true_A'], res['A_pred'],
                                      Q[:, res['test_indices']])
        param_metrics = evaluate_parameter_recovery(
            true_g, true_s, res['g_train'], res['s_train'], res['test_indices'])

        results.append({
            'seed': seed,
            'accuracy': res['accuracy'],
            'difficulty_corr': metrics['difficulty_correlation'],
            'skill_auc': metrics['skill_auc_mean'],
            'g_rmse': param_metrics['g_rmse'],
            's_rmse': param_metrics['s_rmse']
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # 1. 生成数据
    Q = generate_complex_q(n_skills=5, n_items=100)
    X, Q, true_A, true_g, true_s = generate_synthetic_data(q_matrix=Q)
    X = add_noise(X, noise_level=0.05)

    # 2. 训练预测
    results = train_and_predict(X, Q, true_A, true_g, true_s)
    print(f"预测准确率: {results['accuracy']:.2%}")

    # 3. 评估
    metrics = enhanced_evaluation(
        results['X_test'], results['X_pred'],
        results['true_A'], results['A_pred'],
        Q[:, results['test_indices']]
    )

    # 4. 参数恢复评估
    g_rmse = np.sqrt(np.mean((true_g[results['test_indices']] - results['g_train']) ** 2))
    s_rmse = np.sqrt(np.mean((true_s[results['test_indices']] - results['s_train']) ** 2))
    print(f"g参数RMSE: {g_rmse:.4f}, s参数RMSE: {s_rmse:.4f}")

    plt.show()
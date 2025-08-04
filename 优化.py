import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from itertools import product


def reduce_matrices(X_raw, Q_raw, max_knowledge=10, target_questions=50, min_students=10):
    """
    优化后的矩阵缩减函数，解决学生数为0的问题
    参数:
        X_raw: 原始学生答题矩阵(students×items)
        Q_raw: 原始Q矩阵(items×knowledge)
        max_knowledge: 最大知识点数(建议8-12)
        target_questions: 目标题目数
        min_students: 保留的最少学生数
    返回:
        X_reduced: 缩减后的答题矩阵
        Q_reduced: 缩减后的Q矩阵
    """
    # 1. 验证输入数据
    if X_raw.shape[0] == 0:
        raise ValueError("输入数据中没有学生！")
    if X_raw.shape[1] != Q_raw.shape[0]:
        raise ValueError(f"题目数量不匹配：X有{X_raw.shape[1]}题，Q有{Q_raw.shape[0]}题")

    # 2. 处理缺失值（将NaN转为0）
    X = np.nan_to_num(X_raw, nan=0)

    # 3. 缩减Q矩阵知识点
    knowledge_importance = Q_raw.sum(axis=0)
    top_k_idx = np.argsort(-knowledge_importance)[:max_knowledge]
    Q = Q_raw[:, top_k_idx]

    # 4. 题目筛选（确保至少有一些作答记录）
    question_attempts = np.sum(X > 0, axis=0)  # 计算每题有多少学生作答
    valid_questions = question_attempts > 0  # 至少有一个学生作答的题目

    # 如果没有足够有效题目，调整目标
    actual_target = min(target_questions, sum(valid_questions))
    if actual_target < target_questions:
        print(f"警告：只有{sum(valid_questions)}道有效题目，低于目标{target_questions}")

    # 计算题目重要性
    question_importance = (
            0.7 * Q[valid_questions].sum(axis=1) / Q.shape[1] +  # 知识点覆盖
            0.3 * question_attempts[valid_questions] / X.shape[0]  # 作答率
    )

    # 选择最重要的题目
    keep_idx = np.argsort(-question_importance)[:actual_target]
    valid_questions = np.where(valid_questions)[0][keep_idx]
    Q = Q[valid_questions, :]
    X = X[:, valid_questions]

    # 5. 学生筛选（放宽条件）
    student_attempts = np.sum(X > 0, axis=1)
    valid_students = student_attempts >= max(1, min_students // 2)  # 降低阈值

    if sum(valid_students) == 0:
        print("警告：无有效学生，将保留所有学生")
        valid_students = np.ones(X.shape[0], dtype=bool)

    X = X[valid_students, :]

    # 6. 确保知识点覆盖
    knowledge_coverage = Q.sum(axis=0)
    for k in np.where(knowledge_coverage < 3)[0]:
        candidates = np.where(Q[:, k] == 0)[0]
        if len(candidates) > 0:
            Q[np.random.choice(candidates, 1), k] = 1

    print(f"缩减后维度: X{X.shape}, Q{Q.shape}")
    print(f"有效学生数: {X.shape[0]}, 题目数: {X.shape[1]}")
    print(f"Q矩阵密度: {np.mean(Q > 0):.2%}")
    print(f"平均每题知识点数: {np.mean(Q.sum(axis=1)):.2f}")

    return X, Q


def process_group(group_num, data_dir=".", target_questions=50):
    """处理单个组别的数据"""
    # 构建文件路径
    answer_file = os.path.join(data_dir, f"group_answer_matrices/Group_{group_num}_matrix.xlsx")
    q_file = os.path.join(data_dir, f"group_q_matrices/Q_matrix_Group_{group_num}.xlsx")

    # 加载数据并验证
    try:
        X = pd.read_excel(answer_file, index_col=0).values
        Q = pd.read_excel(q_file, index_col=0).values
        print(f"\nGroup_{group_num} 原始维度 - X: {X.shape}, Q: {Q.shape}")

        if X.shape[0] == 0:
            raise ValueError("加载的数据中没有学生记录！")

        X_reduced, Q_reduced = reduce_matrices(X, Q, target_questions=target_questions)

        # 保存结果
        output_dir = os.path.join(data_dir, "processed_results")
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(X_reduced).to_excel(os.path.join(output_dir, f"X_reduced_Group_{group_num}.xlsx"))
        pd.DataFrame(Q_reduced).to_excel(os.path.join(output_dir, f"Q_reduced_Group_{group_num}.xlsx"))

        return X_reduced, Q_reduced
    except Exception as e:
        print(f"处理Group_{group_num}时出错: {str(e)}")
        return None, None


def process_group_real(group_num, data_dir=".", target_questions=50):
    """处理real版本数据"""
    # 构建文件路径
    answer_file = os.path.join(data_dir, f"group_answer_matrices/Group_{group_num}_matrix.xlsx")
    q_file = os.path.join(data_dir, f"group_q_matrices(real)/Q_matrix_Group_{group_num}.xlsx")

    # 加载数据并验证
    try:
        X = pd.read_excel(answer_file, index_col=0).values
        Q = pd.read_excel(q_file, index_col=0).values
        print(f"\nGroup_{group_num}(real) 原始维度 - X: {X.shape}, Q: {Q.shape}")

        if X.shape[0] == 0:
            raise ValueError("加载的数据中没有学生记录！")

        X_reduced, Q_reduced = reduce_matrices(X, Q, target_questions=target_questions)

        # 保存结果
        output_dir = os.path.join(data_dir, "processed_results(real)")
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(X_reduced).to_excel(os.path.join(output_dir, f"X_reduced_Group_{group_num}.xlsx"))
        pd.DataFrame(Q_reduced).to_excel(os.path.join(output_dir, f"Q_reduced_Group_{group_num}.xlsx"))

        return X_reduced, Q_reduced
    except Exception as e:
        print(f"处理Group_{group_num}(real)时出错: {str(e)}")
        return None, None


def batch_process_all_groups(data_dir=".", target_questions=50):
    """批量处理所有组别"""
    # 自动检测组别数量
    answer_dir = os.path.join(data_dir, "group_answer_matrices")
    group_nums = sorted(list(set(
        int(f.split("_")[1]) for f in os.listdir(answer_dir)
        if f.startswith("Group_") and f.endswith(".xlsx")
    )))

    results = {}
    for group_num in tqdm(group_nums, desc="处理组别"):
        X, Q = process_group(group_num, data_dir, target_questions)
        if X is not None:
            results[f"Group_{group_num}"] = {
                "X_shape": X.shape,
                "Q_shape": Q.shape,
                "Q_density": np.mean(Q > 0),
                "avg_knowledge": np.mean(Q.sum(axis=1))
            }

    # 生成报告
    report = pd.DataFrame(results).T
    report_path = os.path.join(data_dir, "processed_results/processing_report.xlsx")
    report.to_excel(report_path)
    print("\n处理完成！结果汇总：")
    print(report)
    return results


def batch_process_all_groups_real(data_dir=".", target_questions=50):
    """批量处理real版本所有组别"""
    # 自动检测组别数量
    answer_dir = os.path.join(data_dir, "group_answer_matrices")
    group_nums = sorted(list(set(
        int(f.split("_")[1]) for f in os.listdir(answer_dir)
        if f.startswith("Group_") and f.endswith(".xlsx")
    )))

    results = {}
    for group_num in tqdm(group_nums, desc="处理组别(real)"):
        X, Q = process_group_real(group_num, data_dir, target_questions)
        if X is not None:
            results[f"Group_{group_num}"] = {
                "X_shape": X.shape,
                "Q_shape": Q.shape,
                "Q_density": np.mean(Q > 0),
                "avg_knowledge": np.mean(Q.sum(axis=1))
            }

    # 生成报告
    report = pd.DataFrame(results).T
    report_path = os.path.join(data_dir, "processed_results(real)/processing_report.xlsx")
    report.to_excel(report_path)
    print("\n处理完成(real)！结果汇总：")
    print(report)
    return results


def check_data_files(data_dir):
    """检查数据文件完整性"""
    print("\n检查数据文件...")
    answer_dir = os.path.join(data_dir, "group_answer_matrices")
    q_dir = os.path.join(data_dir, "group_q_matrices")
    q_real_dir = os.path.join(data_dir, "group_q_matrices(real)")

    # 检查答题矩阵
    print("\n答题矩阵文件:")
    for f in os.listdir(answer_dir):
        if f.endswith(".xlsx"):
            try:
                df = pd.read_excel(os.path.join(answer_dir, f))
                print(f"{f}: {df.shape} (学生×题目)")
            except Exception as e:
                print(f"{f}: 读取失败 - {str(e)}")

    # 检查Q矩阵
    print("\nQ矩阵文件:")
    for d in [q_dir, q_real_dir]:
        if os.path.exists(d):
            print(f"\n目录: {d}")
            for f in os.listdir(d):
                if f.endswith(".xlsx"):
                    try:
                        df = pd.read_excel(os.path.join(d, f))
                        print(f"{f}: {df.shape} (题目×知识点)")
                    except Exception as e:
                        print(f"{f}: 读取失败 - {str(e)}")


if __name__ == "__main__":
    # 数据目录设置
    data_directory = "."  # 修改为您的数据目录

    # 1. 检查数据文件
    check_data_files(data_directory)

    # 2. 处理real版本
    print("\n开始处理real版本数据...")
    real_results = batch_process_all_groups_real(data_directory, target_questions=50)

    # 3. 处理普通版本
    print("\n开始处理普通版本数据...")
    normal_results = batch_process_all_groups(data_directory, target_questions=50)

    # 4. 合并结果
    final_report = pd.concat([
        pd.DataFrame(real_results).T.add_prefix("real_"),
        pd.DataFrame(normal_results).T.add_prefix("normal_")
    ])
    final_report.to_excel("final_processing_report.xlsx")
    print("\n所有处理完成！最终报告已保存为 final_processing_report.xlsx")
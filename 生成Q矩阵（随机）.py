import pandas as pd
import numpy as np
from pathlib import Path


# ==================== 1. 模拟生成全局Q矩阵 ====================
def generate_global_qmatrix(num_items=869, num_skills=124, seed=42):
    """生成合理稀疏性的全局Q矩阵，目标空置率 ≈ 70%-80%"""
    np.random.seed(seed)

    # 均匀一点的知识点权重（避免太冷门）
    skill_weights = np.random.uniform(0.8, 1.2, size=num_skills)

    q_mat = np.zeros((num_items, num_skills))
    item_ids = [str(i) for i in range(100, 100 + num_items)]

    for i in range(num_items):
        # 每题关联知识点数量更多一点（目标非零占比高些）
        k = np.random.choice([2, 3, 4, 5], p=[0.2, 0.4, 0.3, 0.1])
        skills = np.random.choice(num_skills, k, replace=False, p=skill_weights / skill_weights.sum())
        q_mat[i, skills] = 1

    skill_names = [f"知识点_{i + 1}" for i in range(num_skills)]
    return pd.DataFrame(q_mat, index=item_ids, columns=skill_names)


# 生成并打印全局Q矩阵
try:
    q_matrix_global = generate_global_qmatrix()
    print(f"成功生成全局Q矩阵，维度: {q_matrix_global.shape}")
    print("示例数据（前5题）:")
    print(q_matrix_global.head())
    print("\n全局Q矩阵题目ID示例:", q_matrix_global.index[:10].tolist())
except Exception as e:
    print(f"生成Q矩阵失败: {str(e)}")
    raise


# ==================== 2. 模拟分组数据 ====================
def generate_group_data(q_matrix, group_sizes=[273, 276, 263]):
    """从全局Q矩阵中随机抽取题目生成分组数据"""
    groups = {}
    all_items = q_matrix.index.tolist()

    for i, size in enumerate(group_sizes):
        selected_items = np.random.choice(all_items, size, replace=False)
        groups[i] = pd.DataFrame(
            np.random.randint(0, 2, (500, size)),  # 模拟500名学生答题
            columns=selected_items
        )
    return groups


# 生成分组数据
group_files = {
    0: "group_answer_matrices/Group_0_matrix.xlsx",
    1: "group_answer_matrices/Group_1_matrix.xlsx",
    2: "group_answer_matrices/Group_2_matrix.xlsx"
}
group_data = generate_group_data(q_matrix_global)

# 保存模拟分组数据（实际使用时替换为真实数据）
for group_id, df in group_data.items():
    df.to_excel(group_files[group_id])

# ==================== 3. 为每个组提取Q子矩阵 ====================
for group_id, x_matrix_path in group_files.items():
    print(f"\n{'=' * 40}")
    print(f"处理组 {group_id}".center(40, '-'))

    try:
        # 3.1 加载该组的答题矩阵
        print(f"\n正在加载分组文件: {x_matrix_path}")
        x_matrix = pd.read_excel(x_matrix_path, index_col=0)
        print(f"分组数据加载成功，维度: {x_matrix.shape}")

        # 获取题目ID
        group_qs = x_matrix.columns.astype(str).tolist()
        print(f"\n分组题目ID示例(前10个): {group_qs[:10]}")
        print(f"分组题目总数: {len(group_qs)}")

        # 3.2 提取对应的Q子矩阵
        print("\n正在匹配题目ID...")
        matched_qs = q_matrix_global.index.intersection(group_qs)
        missing_qs = set(group_qs) - set(matched_qs)

        print(f"匹配到的题目数量: {len(matched_qs)}")
        print(f"缺失题目数量: {len(missing_qs)}")
        if missing_qs:
            print(f"缺失题目示例: {list(missing_qs)[:5]}")

        if not matched_qs.empty:
            q_subset = q_matrix_global.loc[matched_qs]
            # 移除全0知识点列（知识点至少被1题考察）
            q_subset = q_subset.loc[:, q_subset.sum() > 0]

            # 3.3 保存结果
            output_dir = Path("group_q_matrices")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"Q_matrix_Group_{group_id}.xlsx"

            q_subset.to_excel(output_path)
            print(f"\n✔ 成功保存组 {group_id} Q矩阵到: {output_path}")
            print(f"  提取题目: {len(matched_qs)}/{len(group_qs)}")

            # 3.4 打印统计信息
            print("\nQ矩阵统计:")
            print(f"- 维度: {q_subset.shape[0]}题 × {q_subset.shape[1]}知识点")
            print(f"- 空置率: {1 - np.mean(q_subset.values):.1%}")
            print(f"- 每题平均知识点: {q_subset.sum(axis=1).mean():.2f}")
            print(f"- 知识点覆盖示例: {', '.join(q_subset.columns[:5])}...")
        else:
            print("\n⚠️ 警告: 没有匹配到任何题目！")

    except Exception as e:
        print(f"\n❌ 处理组 {group_id} 时出错: {str(e)}")
        continue

print("\n处理完成！请在 group_q_matrices 文件夹查看结果")
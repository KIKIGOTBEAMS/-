import pandas as pd
import numpy as np
from pathlib import Path

# 1. 加载全局Q矩阵（替换为您的实际文件路径）
try:
    q_matrix_global = pd.read_excel('516matrix.xlsx', index_col=0)
    # 确保索引为字符串类型
    q_matrix_global.index = q_matrix_global.index.astype(str)
    print(f"成功加载全局Q矩阵，维度: {q_matrix_global.shape}")
    print("示例数据（前5题）:")
    print(q_matrix_global.head())
    print("\n全局Q矩阵题目ID示例:", q_matrix_global.index[:10].tolist())
except Exception as e:
    print(f"加载Q矩阵失败: {str(e)}")
    raise

# 2. 加载分组数据（假设已有分组X矩阵）
group_files = {
    0: "group_answer_matrices/Group_0_matrix.xlsx",
    1: "group_answer_matrices/Group_1_matrix.xlsx",
    2: "group_answer_matrices/Group_2_matrix.xlsx"
}

# 3. 为每个组提取Q子矩阵
for group_id, x_matrix_path in group_files.items():
    print(f"\n{'=' * 40}")
    print(f"处理组 {group_id}".center(40, '-'))

    try:
        # 3.1 加载该组的答题矩阵
        print(f"\n正在加载分组文件: {x_matrix_path}")
        x_matrix = pd.read_excel(x_matrix_path, index_col=0)
        print(f"分组数据加载成功，维度: {x_matrix.shape}")

        # 获取题目ID并确保为字符串
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
            # 移除全0知识点列
            q_subset = q_subset.loc[:, q_subset.sum() > 0]

            # 3.3 保存结果
            output_dir = Path("group_q_matrices(real)")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"Q_matrix_Group_{group_id}.xlsx"

            q_subset.to_excel(output_path)
            print(f"\n✔ 成功保存组 {group_id} Q矩阵到: {output_path}")
            print(f"  提取题目: {len(matched_qs)}/{len(group_qs)}")

            # 3.4 打印统计信息
            print("\nQ矩阵统计:")
            print(f"- 维度: {q_subset.shape[0]}题 × {q_subset.shape[1]}知识点")
            print(f"- 空置率: {1 - np.mean(q_subset.values):.1%}")
            print(f"- 知识点覆盖: {', '.join(q_subset.columns)}")
        else:
            print("\n⚠️ 警告: 没有匹配到任何题目！请检查:")
            print("1. 全局Q矩阵和分组矩阵的题目ID是否一致")
            print("2. 题目ID是否有前导/后缀空格或特殊字符")
            print("3. 数据文件是否完整")

    except Exception as e:
        print(f"\n❌ 处理组 {group_id} 时出错: {str(e)}")
        import traceback

        traceback.print_exc()
        continue

print("\n处理完成！请在 group_q_matrices 文件夹查看结果")
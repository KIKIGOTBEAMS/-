import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# 加载数据
data = pd.read_csv(r"C:\Users\BNT\转换\cleaned_data_20250326_0931.csv")
group_data = pd.read_csv('optimal_student_groups_leiden.csv')

# 设置最小学生数阈值（例如：只保留学生数 ≥ min_students 的组）
min_students = 2  # 可修改为更大的值（如10）以过滤小规模组

# 过滤组别
group_counts = group_data['group'].value_counts()
valid_groups = group_counts[group_counts >= min_students].index
print(f"符合条件的组数: {len(valid_groups)} (学生数 ≥ {min_students})")

# 创建输出文件夹
output_dir = "group_answer_matrices"
os.makedirs(output_dir, exist_ok=True)

# 逐组处理
for group_id in tqdm(valid_groups):
    group_students = group_data[group_data['group'] == group_id]['student_id'].values

    # 提取做题矩阵
    group_data_subset = data[data['student_id'].isin(group_students)].drop_duplicates(subset=['student_id', 'qs_id'])
    X_group = group_data_subset.pivot(index='student_id', columns='qs_id', values='qs_validity').fillna(0)

    # 保存为单独Excel文件
    output_path = os.path.join(output_dir, f"Group_{group_id}_matrix.xlsx")
    X_group.to_excel(output_path)
    print(f"已保存: {output_path} (学生数: {len(X_group)}, 题目数: {len(X_group.columns)})")

print(f"\n所有矩阵已保存至文件夹: {output_dir}")
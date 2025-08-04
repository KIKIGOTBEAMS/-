import pandas as pd
import numpy as np
import os


def check_x_matrix_zero_rate(folder_path):
    """
    检测指定文件夹下所有X矩阵文件中0值的比例

    参数:
        folder_path (str): 包含X矩阵文件的文件夹路径
    """
    # 获取文件夹下所有文件
    files = [f for f in os.listdir(folder_path) if f.startswith('X') and (f.endswith('.xlsx') or f.endswith('.csv'))]

    if not files:
        print(f"警告: 文件夹 {folder_path} 中没有找到X矩阵文件!")
        return

    print(f"\n{'=' * 50}")
    print(f"开始检测文件夹: {folder_path}")
    print(f"共找到 {len(files)} 个X矩阵文件")
    print(f"{'=' * 50}\n")

    results = []

    for file in files:
        try:
            # 读取文件
            file_path = os.path.join(folder_path, file)
            if file.endswith('.xlsx'):
                df = pd.read_excel(file_path, index_col=0)
            else:
                df = pd.read_csv(file_path, index_col=0)

            # 转换为numpy数组
            X = df.values

            # 计算0值比例
            total_cells = X.size
            zero_cells = (X == 0).sum()
            zero_rate = zero_cells / total_cells

            # 收集结果
            results.append({
                '文件名': file,
                '总单元格数': total_cells,
                '0值单元格数': zero_cells,
                '0值比例(%)': round(zero_rate * 100, 2),
                '形状': X.shape,
                '1值比例(%)': round(100 - zero_rate * 100, 2),
                '示例数据': f"{X[0, 0]} (首元素)" if X.size > 0 else '空文件'
            })

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue

    # 打印结果
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    # 汇总统计
    print(f"\n{'=' * 50}")
    print("0值比例统计摘要:")
    print(f"平均0值比例: {result_df['0值比例(%)'].mean():.2f}%")
    print(
        f"最高0值比例: {result_df['0值比例(%)'].max():.2f}% (文件: {result_df.loc[result_df['0值比例(%)'].idxmax(), '文件名']})")
    print(
        f"最低0值比例: {result_df['0值比例(%)'].min():.2f}% (文件: {result_df.loc[result_df['0值比例(%)'].idxmin(), '文件名']})")
    print(f"平均1值比例: {result_df['1值比例(%)'].mean():.2f}%")
    print(f"{'=' * 50}")


# 使用示例 - 检测多个文件夹
folders_to_check = [
    "processed_results",  # 原始数据文件夹
    "processed_results(real)"  # 真实数据文件夹
]

for folder in folders_to_check:
    if os.path.exists(folder):
        check_x_matrix_zero_rate(folder)
    else:
        print(f"文件夹不存在: {folder}")
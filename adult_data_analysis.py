# -*- coding: utf-8 -*-
"""
UCI Adult Income 数据集预处理分析
逐步数据预处理流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

# 设置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ============================================================
# 日志功能
# ============================================================
log_file = 'data_processing_log.txt'

def init_log():
    """初始化日志文件"""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("UCI Adult Income 数据集预处理分析日志\n")
        f.write(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

def log_print(message, to_console=True):
    """同时打印到终端和写入日志文件"""
    if to_console:
        print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

# ============================================================
# 数据加载
# ============================================================
def load_data():
    """从 UCI ML Repository 加载 Adult 数据集"""
    log_print("\n" + "=" * 70)
    log_print("【步骤 1】数据加载")
    log_print("=" * 70)
    
    try:
        from ucimlrepo import fetch_ucirepo
        log_print("正在从 UCI ML Repository 加载 Adult 数据集（ID=2）...")
        adult = fetch_ucirepo(id=2)
        
        # 合并特征和标签
        x, y = adult.data.features, adult.data.targets
        df = pd.concat([x, y], axis=1)
        
        log_print("✓ 数据集加载成功！")
        
        # 数据清理1：处理特殊缺失标记 '?'
        log_print("\n【数据清理】")
        log_print("1. 处理特殊缺失标记：将 '?' 替换为 NaN")
        for col in df.columns:
            if df[col].dtype == 'object':
                before_count = df[col].astype(str).str.strip().isin(['?', ' ?', '?  ']).sum()
                if before_count > 0:
                    log_print(f"   - {col}: 发现 {before_count} 个 '?' 标记，已转换为 NaN")
                    df[col] = df[col].replace(['?', ' ?', '?  '], np.nan)
        
        # 数据清理2：统一目标变量格式
        log_print("\n2. 统一目标变量格式：去除 income 列中的句点")
        if 'income' in df.columns:
            before_unique = df['income'].unique()
            log_print(f"   - 清理前的唯一值：{before_unique}")
            df['income'] = df['income'].str.strip().str.replace('.', '', regex=False)
            after_unique = df['income'].unique()
            log_print(f"   - 清理后的唯一值：{after_unique}")
            log_print(f"   ✓ 已将 '<=50K.' 合并为 '<=50K'，'>50K.' 合并为 '>50K'")
        
        log_print("\n✓ 数据清理完成！\n")
        return df
    
    except Exception as e:
        log_print(f"✗ 加载失败：{e}")
        return None

# ============================================================
# 数据基本信息
# ============================================================
def analyze_basic_info(df):
    """分析数据集的基本信息"""
    log_print("\n" + "=" * 70)
    log_print("【步骤 2】数据基本信息分析")
    log_print("=" * 70)
    
    # 1. 数据规模
    log_print("\n【1. 数据规模】")
    log_print(f"  样本数（行数）：{df.shape[0]:,}")
    log_print(f"  特征数（列数）：{df.shape[1]}")
    log_print(f"  数据总单元格数：{df.shape[0] * df.shape[1]:,}\n")
    
    # 2. 列名列表
    log_print("【2. 特征列表】")
    for idx, col in enumerate(df.columns, 1):
        log_print(f"  {idx:2d}. {col}")
    log_print("")
    
    # 3. 数据类型分布
    log_print("【3. 数据类型统计】")
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns.tolist()
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns.tolist()
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    log_print(f"  整数型特征：{len(int_cols)} 个")
    for col in int_cols:
        log_print(f"    - {col}")
    
    log_print(f"\n  浮点型特征：{len(float_cols)} 个")
    if float_cols:
        for col in float_cols:
            log_print(f"    - {col}")
    else:
        log_print("    （无浮点型特征）")
    
    log_print(f"\n  分类/对象型特征：{len(obj_cols)} 个")
    for col in obj_cols:
        log_print(f"    - {col}")
    log_print("")
    
    # 4. 数据前5行预览
    log_print("【4. 数据前5行预览】")
    log_print(df.head().to_string())
    log_print("")
    
    # 5. 数据描述性统计（数值型特征）
    log_print("【5. 数值型特征描述性统计】")
    numeric_stats = df.describe()
    log_print(numeric_stats.to_string())
    log_print("")
    
    # 6. 分类特征的唯一值统计
    log_print("【6. 分类特征的唯一值统计】")
    for col in obj_cols:
        unique_count = df[col].nunique()
        log_print(f"  {col}: {unique_count} 个唯一值")
        if col == 'income':
            log_print(f"    具体类别：{sorted(df[col].dropna().unique().tolist())}")
    log_print("")
    
    # 7. 目标变量分布（income）
    if 'income' in df.columns:
        log_print("【7. 目标变量（income）分布】")
        income_dist = df['income'].value_counts().sort_index()
        log_print(income_dist.to_string())
        
        income_pct = df['income'].value_counts(normalize=True).sort_index() * 100
        log_print("\n  百分比分布：")
        for idx, val in income_pct.items():
            log_print(f"    {idx}: {val:.2f}%")
        
        log_print("\n  说明：数据已清理，合并了带句点的标签")
        log_print("    - '<=50K' 包含原始的 '<=50K' 和 '<=50K.'")
        log_print("    - '>50K' 包含原始的 '>50K' 和 '>50K.'")
        log_print("")
    
    # 8. 缺失值统计
    log_print("【8. 缺失值初步检查】")
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        log_print(f"  总缺失值数量：{total_missing}")
        log_print("\n  各列缺失情况：")
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                missing_pct = (missing / len(df)) * 100
                log_print(f"    {col}: {missing} 个 ({missing_pct:.2f}%)")
    else:
        log_print("  ✓ 无缺失值\n")
    
    # 数据质量说明
    log_print("【9. 数据质量说明】")
    log_print("  ✓ 特殊缺失标记（'?'）已在数据加载时转换为标准 NaN")
    log_print("  ✓ 目标变量格式已统一（去除句点，仅保留 '<=50K' 和 '>50K'）")
    log_print("  ✓ 所有缺失值统计已在【8. 缺失值初步检查】中显示")
    log_print("")

# ============================================================
# 生成数据概览图表
# ============================================================
def generate_overview_charts(df):
    """生成数据概览图表"""
    log_print("\n" + "=" * 70)
    log_print("【步骤 3】生成数据概览图表")
    log_print("=" * 70 + "\n")
    
    # 图1：数值型特征分布
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'income' in numeric_cols:
        numeric_cols.remove('income')
    
    if len(numeric_cols) > 0:
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(df[col].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col, fontsize=11)
            axes[idx].set_ylabel('频数', fontsize=11)
            axes[idx].set_title(f'{col} 分布', fontsize=12)
            axes[idx].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('数值型特征分布图', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig('图1_数值特征分布.png', dpi=150, bbox_inches='tight')
        plt.close()
        log_print("✓ 已保存：图1_数值特征分布.png")
    
    # 图2：目标变量分布饼图
    if 'income' in df.columns:
        plt.figure(figsize=(8, 6))
        income_counts = df['income'].value_counts()
        colors = ['#66b3ff', '#ff9999']
        plt.pie(income_counts, labels=income_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors, textprops={'fontsize': 12})
        plt.title('目标变量（income）分布', fontsize=16, pad=20)
        plt.savefig('图2_目标变量分布.png', dpi=150, bbox_inches='tight')
        plt.close()
        log_print("✓ 已保存：图2_目标变量分布.png")
    
    # 图3：数据类型统计柱状图
    int_count = len(df.select_dtypes(include=['int64', 'int32']).columns)
    float_count = len(df.select_dtypes(include=['float64', 'float32']).columns)
    obj_count = len(df.select_dtypes(include=['object', 'category']).columns)
    
    plt.figure(figsize=(8, 6))
    categories = ['整数型', '浮点型', '分类/对象型']
    counts = [int_count, float_count, obj_count]
    colors_bar = ['#4CAF50', '#2196F3', '#FF9800']
    
    bars = plt.bar(categories, counts, color=colors_bar, edgecolor='black', alpha=0.8)
    plt.ylabel('特征数量', fontsize=12)
    plt.title('特征数据类型分布', fontsize=16, pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    # 在柱状图上显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('图3_数据类型分布.png', dpi=150, bbox_inches='tight')
    plt.close()
    log_print("✓ 已保存：图3_数据类型分布.png")
    
    log_print("")

# ============================================================
# 保存数据摘要表格
# ============================================================
def save_summary_table(df):
    """保存数据摘要信息到CSV"""
    log_print("\n" + "=" * 70)
    log_print("【步骤 4】保存数据摘要表格")
    log_print("=" * 70 + "\n")
    
    # 创建摘要表
    summary_data = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        unique = df[col].nunique()
        
        summary_data.append({
            '特征名': col,
            '数据类型': dtype,
            '缺失值数量': missing,
            '缺失值比例(%)': round(missing_pct, 2),
            '唯一值数量': unique,
            '样本数': len(df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 尝试保存，如果文件被占用则使用新文件名
    try:
        summary_df.to_csv('数据摘要表.csv', index=False, encoding='utf-8-sig')
        log_print("✓ 已保存：数据摘要表.csv\n")
    except PermissionError:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f'数据摘要表_{timestamp}.csv'
        summary_df.to_csv(new_filename, index=False, encoding='utf-8-sig')
        log_print(f"✓ 已保存：{new_filename}（原文件被占用）\n")
    
    return summary_df

# ============================================================
# 主函数
# ============================================================
def main():
    """主执行流程"""
    print("\n" + "=" * 70)
    print("UCI Adult Income 数据集预处理分析")
    print("=" * 70 + "\n")
    
    # 初始化日志
    init_log()
    
    # 1. 加载数据
    df = load_data()
    if df is None:
        log_print("\n✗ 数据加载失败，程序终止。")
        return
    
    # 2. 分析基本信息
    analyze_basic_info(df)
    
    # 3. 生成概览图表
    generate_overview_charts(df)
    
    # 4. 保存摘要表格
    summary_df = save_summary_table(df)
    
    # 完成总结
    log_print("\n" + "=" * 70)
    log_print("【数据加载与概览完成】")
    log_print("=" * 70)
    log_print(f"\n数据规模：{df.shape[0]:,} 行 × {df.shape[1]} 列")
    log_print(f"\n生成的文件：")
    log_print("  1. data_processing_log.txt - 完整日志文件")
    log_print("  2. 图1_数值特征分布.png - 数值特征分布图")
    log_print("  3. 图2_目标变量分布.png - 目标变量分布图")
    log_print("  4. 图3_数据类型分布.png - 数据类型统计图")
    log_print("  5. 数据摘要表.csv - 数据摘要表格")
    log_print("\n✅ 数据加载与基本分析完成！等待下一步预处理指令...\n")
    
    log_print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 70 + "\n")
    
    print("\n" + "=" * 70)
    print("✅ 数据加载与基本分析完成！")
    print("=" * 70)
    print("\n请查看生成的文件：")
    print("  • data_processing_log.txt - 详细日志")
    print("  • 图1_数值特征分布.png")
    print("  • 图2_目标变量分布.png")
    print("  • 图3_数据类型分布.png")
    print("  • 数据摘要表.csv")
    print("\n现在可以进行下一步的数据预处理操作。\n")
    
    return df

# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    df = main()


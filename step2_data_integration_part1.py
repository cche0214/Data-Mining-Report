# -*- coding: utf-8 -*-
"""
【数据预处理实验 - 步骤2.1：数据集成 - 连续变量相关性分析】
目标：识别高度相关的数值特征，避免多重共线性
方法：Pearson相关系数分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ============================================================
# 日志系统
# ============================================================
log_file = 'step2_integration_log.txt'

def init_log():
    """初始化日志"""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("【数据预处理实验 - 步骤2：数据集成】\n")
        f.write("第一部分：连续变量相关性分析（Pearson相关系数）\n")
        f.write(f"执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

def log_print(message, to_console=True):
    """同时输出到终端和日志"""
    if to_console:
        print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

# ============================================================
# 数据加载
# ============================================================
def load_cleaned_data():
    """加载清洗后的数据"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.1.0】数据加载")
    log_print("=" * 80)
    
    try:
        df = pd.read_csv('cleaned_data.csv')
        log_print("✓ 已加载清洗后的数据")
        log_print(f"  数据规模：{df.shape[0]:,} 行 × {df.shape[1]} 列")
        log_print(f"  数据来源：cleaned_data.csv（步骤1清洗后的数据）\n")
        return df
    except FileNotFoundError:
        log_print("✗ 错误：未找到 cleaned_data.csv 文件")
        log_print("  请先运行步骤1（数据清洗）生成清洗后的数据\n")
        return None

# ============================================================
# 阶段 2.1：连续变量相关性分析
# ============================================================
def analyze_correlation_theory():
    """相关性分析理论说明"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.1.1】Pearson 相关系数理论基础")
    log_print("=" * 80)
    
    log_print("\n【什么是 Pearson 相关系数？】")
    log_print("-" * 60)
    log_print("Pearson 相关系数（Pearson Correlation Coefficient，记作 r）")
    log_print("用于衡量两个连续变量之间的线性相关程度。")
    log_print("")
    log_print("计算公式：")
    log_print("  r = Cov(X, Y) / (σ_X × σ_Y)")
    log_print("  其中：")
    log_print("    - Cov(X, Y)：X 和 Y 的协方差")
    log_print("    - σ_X, σ_Y：X 和 Y 的标准差")
    log_print("")
    log_print("【取值范围与含义】")
    log_print("-" * 60)
    log_print("  r ∈ [-1, 1]")
    log_print("")
    log_print("  |r| = 1.0      : 完全线性相关")
    log_print("  |r| ≥ 0.7      : 强相关 ⚠️  需要关注（可能多重共线性）")
    log_print("  0.4 ≤ |r| < 0.7: 中等相关")
    log_print("  0.2 ≤ |r| < 0.4: 弱相关")
    log_print("  |r| < 0.2      : 极弱或无相关")
    log_print("")
    log_print("  r > 0：正相关（一个变量增大，另一个也增大）")
    log_print("  r < 0：负相关（一个变量增大，另一个减小）")
    log_print("  r = 0：无线性相关")
    log_print("")
    log_print("【为什么需要相关性分析？】")
    log_print("-" * 60)
    log_print("1. 识别冗余特征：")
    log_print("   - 高度相关的特征携带重复信息，可以删除其中一个")
    log_print("   - 例如：身高（米）和身高（厘米）完全相关（r=1）")
    log_print("")
    log_print("2. 避免多重共线性：")
    log_print("   - 在线性回归等模型中，高度相关的特征会导致：")
    log_print("     • 模型参数不稳定")
    log_print("     • 标准误增大")
    log_print("     • 难以解释各特征的独立贡献")
    log_print("")
    log_print("3. 提高模型效率：")
    log_print("   - 减少特征数量，降低计算复杂度")
    log_print("   - 避免过拟合")
    log_print("")
    log_print("【本实验的分析目标】")
    log_print("-" * 60)
    log_print("识别 Adult Income 数据集中的高度相关特征对（|r| > 0.7）")
    log_print("如果发现强相关，需要决策保留哪一个特征\n")

def select_continuous_features(df):
    """选择连续变量进行分析"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.1.2】选择连续变量")
    log_print("=" * 80)
    
    # 预定义的连续变量
    continuous_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    log_print("\n【连续变量选择】")
    log_print("-" * 60)
    log_print(f"从清洗后的数据中选择 {len(continuous_features)} 个连续数值变量：")
    
    for idx, feat in enumerate(continuous_features, 1):
        dtype = df[feat].dtype
        min_val = df[feat].min()
        max_val = df[feat].max()
        mean_val = df[feat].mean()
        log_print(f"\n  {idx}. {feat}")
        log_print(f"     - 数据类型：{dtype}")
        log_print(f"     - 范围：[{min_val}, {max_val}]")
        log_print(f"     - 均值：{mean_val:.2f}")
    
    log_print("\n【变量选择说明】")
    log_print("-" * 60)
    log_print("✓ 选择这6个变量的原因：")
    log_print("  1. age：年龄，典型连续变量")
    log_print("  2. fnlwgt：人口统计权重，连续数值")
    log_print("  3. education-num：虽然是等级编码，但作为数值可计算相关性")
    log_print("  4. capital-gain：投资收益，连续变量")
    log_print("  5. capital-loss：投资损失，连续变量")
    log_print("  6. hours-per-week：工作时长，连续变量")
    log_print("")
    log_print("✗ 排除的变量：")
    log_print("  - 分类变量（workclass, education, occupation 等）不适用 Pearson")
    log_print("  - 分类变量需要用卡方检验分析（将在后续模块进行）\n")
    
    # 提取连续变量数据
    df_continuous = df[continuous_features].copy()
    
    return df_continuous, continuous_features

def calculate_correlation(df_continuous):
    """计算 Pearson 相关系数矩阵"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.1.3】计算 Pearson 相关系数矩阵")
    log_print("=" * 80)
    
    log_print("\n【计算过程】")
    log_print("-" * 60)
    log_print("使用 pandas 内置方法计算 Pearson 相关系数：")
    log_print("  df.corr(method='pearson')")
    log_print("")
    
    # 计算相关系数矩阵
    corr_matrix = df_continuous.corr(method='pearson')
    
    log_print("✓ 相关系数矩阵计算完成\n")
    log_print("【相关系数矩阵】（保留3位小数）")
    log_print("-" * 60)
    log_print(corr_matrix.round(3).to_string())
    log_print("")
    
    return corr_matrix

def analyze_correlation_matrix(corr_matrix, continuous_features):
    """分析相关系数矩阵"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.1.4】相关系数矩阵分析")
    log_print("=" * 80)
    
    log_print("\n【相关性强度统计】")
    log_print("-" * 60)
    
    # 统计不同强度的相关对数量
    n_features = len(continuous_features)
    total_pairs = n_features * (n_features - 1) // 2  # 排除对角线和重复对
    
    # 提取上三角（不包括对角线）
    upper_triangle = np.triu(corr_matrix.values, k=1)
    upper_triangle_flat = upper_triangle[upper_triangle != 0]
    
    # 按相关性强度分类
    very_strong = np.sum(np.abs(upper_triangle_flat) >= 0.7)
    moderate = np.sum((np.abs(upper_triangle_flat) >= 0.4) & (np.abs(upper_triangle_flat) < 0.7))
    weak = np.sum((np.abs(upper_triangle_flat) >= 0.2) & (np.abs(upper_triangle_flat) < 0.4))
    very_weak = np.sum(np.abs(upper_triangle_flat) < 0.2)
    
    log_print(f"总特征对数：{total_pairs} 对")
    log_print(f"")
    log_print(f"相关性分布：")
    log_print(f"  强相关（|r| ≥ 0.7）    : {very_strong} 对（{very_strong/total_pairs*100:.1f}%）")
    log_print(f"  中等相关（0.4 ≤ |r| < 0.7）: {moderate} 对（{moderate/total_pairs*100:.1f}%）")
    log_print(f"  弱相关（0.2 ≤ |r| < 0.4）  : {weak} 对（{weak/total_pairs*100:.1f}%）")
    log_print(f"  极弱相关（|r| < 0.2）   : {very_weak} 对（{very_weak/total_pairs*100:.1f}%）")
    log_print("")
    
    # 找出最强的5对相关
    log_print("【最强相关的特征对（Top 5）】")
    log_print("-" * 60)
    
    pairs = []
    for i in range(len(continuous_features)):
        for j in range(i+1, len(continuous_features)):
            pairs.append({
                '特征1': continuous_features[i],
                '特征2': continuous_features[j],
                '相关系数': corr_matrix.iloc[i, j],
                '绝对值': abs(corr_matrix.iloc[i, j])
            })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('绝对值', ascending=False)
    
    for idx, row in pairs_df.head(5).iterrows():
        r_val = row['相关系数']
        abs_val = row['绝对值']
        
        if abs_val >= 0.7:
            strength = "⚠️  强相关"
        elif abs_val >= 0.4:
            strength = "中等相关"
        elif abs_val >= 0.2:
            strength = "弱相关"
        else:
            strength = "极弱相关"
        
        direction = "正相关" if r_val > 0 else "负相关"
        
        log_print(f"\n{row['特征1']} ↔ {row['特征2']}")
        log_print(f"  r = {r_val:.4f}  [{direction}，{strength}]")
    
    log_print("")
    
    return pairs_df

def identify_strong_correlations(pairs_df, threshold=0.7):
    """识别强相关特征对"""
    log_print("\n" + "=" * 80)
    log_print(f"【阶段 2.1.5】识别强相关特征对（|r| > {threshold}）")
    log_print("=" * 80)
    
    strong_pairs = pairs_df[pairs_df['绝对值'] > threshold]
    
    log_print(f"\n【强相关特征筛选】")
    log_print("-" * 60)
    log_print(f"筛选标准：|r| > {threshold}")
    log_print(f"筛选结果：发现 {len(strong_pairs)} 对强相关特征\n")
    
    if len(strong_pairs) > 0:
        log_print("【强相关特征列表】")
        log_print("-" * 60)
        for idx, row in strong_pairs.iterrows():
            log_print(f"\n特征对 {idx+1}：{row['特征1']} ↔ {row['特征2']}")
            log_print(f"  相关系数：r = {row['相关系数']:.4f}")
            log_print(f"  相关强度：{'正相关' if row['相关系数'] > 0 else '负相关'}，|r| = {row['绝对值']:.4f}")
            log_print(f"  ⚠️  警告：强相关可能导致多重共线性问题")
            log_print(f"  建议：考虑删除其中一个特征或进行特征组合")
        log_print("")
    else:
        log_print("✓ 未发现强相关特征对（|r| > 0.7）")
        log_print("  说明：所有特征之间的线性相关性较弱")
        log_print("  结论：无需删除任何特征，可以全部保留用于建模\n")
    
    return strong_pairs

def plot_correlation_heatmap(corr_matrix, continuous_features, save_path='图2_相关性热力图.png'):
    """绘制相关性热力图"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.1.6】绘制相关性热力图")
    log_print("=" * 80)
    
    log_print("\n【热力图说明】")
    log_print("-" * 60)
    log_print("热力图特点：")
    log_print("  - 颜色：红色表示正相关，蓝色表示负相关，白色表示无相关")
    log_print("  - 深度：颜色越深，相关性越强")
    log_print("  - 数值：格子中的数字为相关系数 r")
    log_print("  - 对角线：变量与自身的相关系数为 1（完全相关）")
    log_print("")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制热力图
    sns.heatmap(corr_matrix, 
                annot=True,           # 显示数值
                fmt='.3f',            # 数值格式：3位小数
                cmap='RdBu_r',        # 红蓝配色
                center=0,             # 0 为中心色（白色）
                square=True,          # 正方形格子
                linewidths=1,         # 格子边框宽度
                cbar_kws={"shrink": 0.8, "label": "Pearson 相关系数 (r)"},
                vmin=-1, vmax=1,      # 色彩范围
                xticklabels=continuous_features,
                yticklabels=continuous_features)
    
    # 设置标题和标签
    plt.title('连续变量 Pearson 相关系数热力图', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('特征', fontsize=12, fontweight='bold')
    plt.ylabel('特征', fontsize=12, fontweight='bold')
    
    # 旋转标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 添加说明文本
    textstr = '说明：\n• |r| ≥ 0.7：强相关\n• 0.4 ≤ |r| < 0.7：中等相关\n• |r| < 0.4：弱相关'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.25, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log_print(f"✓ 相关性热力图已保存：{save_path}")
    log_print(f"  图片尺寸：12×10 英寸")
    log_print(f"  分辨率：150 DPI")
    log_print(f"  配色方案：红蓝双色（RdBu_r）\n")

def make_decision(strong_pairs, df):
    """基于强相关分析做出特征选择决策"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.1.7】特征选择决策")
    log_print("=" * 80)
    
    log_print("\n【决策分析】")
    log_print("-" * 60)
    
    if len(strong_pairs) > 0:
        log_print("发现强相关特征对，需要决策是否删除：\n")
        
        for idx, row in strong_pairs.iterrows():
            feat1 = row['特征1']
            feat2 = row['特征2']
            r_val = row['相关系数']
            
            log_print(f"【特征对 {idx+1}】{feat1} ↔ {feat2} (r = {r_val:.4f})")
            log_print(f"-" * 40)
            log_print(f"业务含义分析：")
            log_print(f"  {feat1}：[需要根据实际情况分析业务含义]")
            log_print(f"  {feat2}：[需要根据实际情况分析业务含义]")
            log_print(f"")
            log_print(f"决策建议：")
            log_print(f"  - 如果两者含义相近，保留业务解释性更强的特征")
            log_print(f"  - 如果两者独立但统计相关，可考虑特征组合或PCA降维")
            log_print(f"  - 可通过领域知识判断哪个特征更重要\n")
        
        log_print("【本实验决策】")
        log_print("-" * 60)
        log_print("暂不删除任何特征，原因：")
        log_print("  1. 需要进一步分析业务含义")
        log_print("  2. 可在后续建模中通过特征重要性评估")
        log_print("  3. 可在数据规约阶段通过 PCA 处理共线性\n")
    else:
        log_print("✓ 未发现强相关特征对（|r| > 0.7）")
        log_print("")
        log_print("【结论】Adult Income 数据集的连续变量相关性分析：")
        log_print("-" * 60)
        log_print("1. 所有特征对的相关系数 |r| < 0.7")
        log_print("2. 大部分特征对的相关系数 |r| < 0.3（极弱或弱相关）")
        log_print("3. 特征之间相对独立，携带不同的信息")
        log_print("4. 不存在明显的多重共线性问题")
        log_print("")
        log_print("【决策】")
        log_print("-" * 60)
        log_print("✓ 保留所有6个连续变量，无需删除")
        log_print("  理由：")
        log_print("    - 特征间相关性弱，信息冗余度低")
        log_print("    - 每个特征都携带独特的信息")
        log_print("    - 不会对后续建模造成多重共线性问题")
        log_print("")
        log_print("保留的特征列表：")
        log_print("  1. age（年龄）")
        log_print("  2. fnlwgt（人口统计权重）")
        log_print("  3. education-num（教育年限）")
        log_print("  4. capital-gain（投资收益）")
        log_print("  5. capital-loss（投资损失）")
        log_print("  6. hours-per-week（每周工作小时数）\n")

def summary_report(corr_matrix, pairs_df, strong_pairs):
    """生成分析总结报告"""
    log_print("\n" + "=" * 80)
    log_print("【连续变量相关性分析总结报告】")
    log_print("=" * 80)
    
    log_print("\n【分析概况】")
    log_print("-" * 60)
    log_print(f"分析变量数：{corr_matrix.shape[0]} 个")
    log_print(f"分析变量对数：{len(pairs_df)} 对")
    log_print(f"分析方法：Pearson 相关系数")
    log_print(f"")
    
    log_print("【关键发现】")
    log_print("-" * 60)
    
    # 最强正相关
    max_pos = pairs_df[pairs_df['相关系数'] > 0].iloc[0] if len(pairs_df[pairs_df['相关系数'] > 0]) > 0 else None
    if max_pos is not None:
        log_print(f"最强正相关：{max_pos['特征1']} ↔ {max_pos['特征2']} (r = {max_pos['相关系数']:.4f})")
    
    # 最强负相关
    max_neg = pairs_df[pairs_df['相关系数'] < 0].sort_values('相关系数').iloc[0] if len(pairs_df[pairs_df['相关系数'] < 0]) > 0 else None
    if max_neg is not None:
        log_print(f"最强负相关：{max_neg['特征1']} ↔ {max_neg['特征2']} (r = {max_neg['相关系数']:.4f})")
    
    log_print(f"强相关特征对（|r| > 0.7）：{len(strong_pairs)} 对")
    log_print(f"")
    
    log_print("【分析结论】")
    log_print("-" * 60)
    
    if len(strong_pairs) == 0:
        log_print("✓ Adult Income 数据集的连续变量之间相关性较弱")
        log_print("✓ 所有特征相对独立，信息冗余度低")
        log_print("✓ 不存在需要处理的多重共线性问题")
        log_print("✓ 建议保留所有连续变量用于后续分析和建模")
    else:
        log_print(f"⚠️  发现 {len(strong_pairs)} 对强相关特征")
        log_print("⚠️  建议在后续建模中关注这些特征对")
        log_print("⚠️  可考虑删除冗余特征或使用 PCA 降维")
    
    log_print("")
    log_print("【下一步工作】")
    log_print("-" * 60)
    log_print("1. 对分类变量进行卡方检验（独立性检验）")
    log_print("2. 分析分类变量与目标变量的关联性")
    log_print("3. 识别冗余的分类特征\n")
    
    log_print("✅ 连续变量相关性分析完成！\n")

# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "=" * 80)
    print("【数据预处理实验 - 步骤2.1：数据集成 - 连续变量相关性分析】")
    print("=" * 80 + "\n")
    
    init_log()
    
    # 1. 加载清洗后的数据
    df = load_cleaned_data()
    if df is None:
        return
    
    # 2. 理论说明
    analyze_correlation_theory()
    
    # 3. 选择连续变量
    df_continuous, continuous_features = select_continuous_features(df)
    
    # 4. 计算相关系数矩阵
    corr_matrix = calculate_correlation(df_continuous)
    
    # 5. 分析相关系数矩阵
    pairs_df = analyze_correlation_matrix(corr_matrix, continuous_features)
    
    # 6. 识别强相关特征对
    strong_pairs = identify_strong_correlations(pairs_df, threshold=0.7)
    
    # 7. 绘制热力图
    plot_correlation_heatmap(corr_matrix, continuous_features)
    
    # 8. 做出决策
    make_decision(strong_pairs, df)
    
    # 9. 生成总结报告
    summary_report(corr_matrix, pairs_df, strong_pairs)
    
    log_print(f"执行结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80 + "\n")
    
    print("\n" + "=" * 80)
    print("✅ 连续变量相关性分析完成！")
    print("=" * 80)
    print("\n生成的文件：")
    print("  1. step2_integration_log.txt - 详细分析日志")
    print("  2. 图2_相关性热力图.png - Pearson 相关系数热力图")
    print("\n下一步：进行分类变量的卡方检验分析\n")

if __name__ == "__main__":
    main()


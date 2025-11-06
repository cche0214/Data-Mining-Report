# -*- coding: utf-8 -*-
"""
【数据预处理实验 - 步骤3.1：数据规约 - 数据规范化】
目标：对连续变量进行规范化处理，统一特征尺度
方法：Z-score标准化（对比Min-Max和小数定标）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ============================================================
# 日志系统
# ============================================================
log_file = 'step3_reduction_part1_log.txt'

def init_log():
    """初始化日志"""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("【数据预处理实验 - 步骤3.1：数据规约】\n")
        f.write("第一部分：数据规范化（Normalization）\n")
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
def load_integrated_data():
    """加载集成后的数据"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 3.1.0】数据加载")
    log_print("=" * 80)
    
    try:
        df = pd.read_csv('integrated_data.csv')
        log_print("✓ 已加载集成后的数据")
        log_print(f"  数据规模：{df.shape[0]:,} 行 × {df.shape[1]} 列")
        log_print(f"  数据来源：integrated_data.csv（步骤2集成后的数据）\n")
        return df
    except FileNotFoundError:
        log_print("✗ 错误：未找到 integrated_data.csv 文件")
        log_print("  请先运行步骤2（数据集成）生成集成后的数据\n")
        return None

# ============================================================
# 规范化理论
# ============================================================
def normalization_theory():
    """规范化理论说明"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 3.1.1】数据规范化理论基础")
    log_print("=" * 80)
    
    log_print("\n【什么是数据规范化？】")
    log_print("-" * 60)
    log_print("数据规范化（Normalization）是将不同量纲、不同数值范围的特征")
    log_print("转换到统一尺度的过程，消除特征间的数量级差异。")
    log_print("")
    log_print("【为什么需要规范化？】")
    log_print("-" * 60)
    log_print("1. 消除量纲影响：")
    log_print("   - 如：age（17-80）vs fnlwgt（12285-1490400），相差万倍")
    log_print("   - 某些算法（如KNN、SVM、神经网络）对数值尺度敏感")
    log_print("")
    log_print("2. 加速模型收敛：")
    log_print("   - 梯度下降算法在统一尺度下收敛更快")
    log_print("")
    log_print("3. 避免特征偏向：")
    log_print("   - 防止数值大的特征主导模型学习")
    log_print("")
    log_print("【三种常用规范化方法】")
    log_print("=" * 80)
    
    log_print("\n方法1：Min-Max 归一化（最大最小值规范化）")
    log_print("-" * 60)
    log_print("公式：x' = (x - min) / (max - min)")
    log_print("结果：将数据线性映射到 [0, 1] 区间")
    log_print("")
    log_print("优点：")
    log_print("  ✓ 保留原始分布的形状")
    log_print("  ✓ 结果范围明确 [0, 1]")
    log_print("  ✓ 适合有明确边界的数据")
    log_print("")
    log_print("缺点：")
    log_print("  ✗ 对异常值敏感（一个极值会压缩所有其他值）")
    log_print("  ✗ 新数据可能超出 [0, 1] 范围")
    log_print("")
    log_print("适用场景：")
    log_print("  - 数据分布均匀，无极端值")
    log_print("  - 需要固定范围的场景（如图像像素 [0, 255]）")
    log_print("")
    
    log_print("\n方法2：Z-score 标准化（零均值标准化）")
    log_print("-" * 60)
    log_print("公式：x' = (x - μ) / σ")
    log_print("      其中 μ=均值, σ=标准差")
    log_print("结果：均值=0，标准差=1")
    log_print("")
    log_print("优点：")
    log_print("  ✓ 对异常值稳健（相对Min-Max）")
    log_print("  ✓ 保留了数据的分布特征")
    log_print("  ✓ 适合偏态分布和有极端值的数据")
    log_print("")
    log_print("缺点：")
    log_print("  ✗ 结果范围不固定（通常在 [-3, 3] 之间）")
    log_print("  ✗ 需要假设数据接近正态分布（实际应用中较宽松）")
    log_print("")
    log_print("适用场景：")
    log_print("  - 数据存在极端值")
    log_print("  - 数据分布偏态（如本数据集的 capital-gain/loss）")
    log_print("  - 线性回归、逻辑回归、神经网络等")
    log_print("")
    
    log_print("\n方法3：小数定标规范化（Decimal Scaling）")
    log_print("-" * 60)
    log_print("公式：x' = x / 10^k")
    log_print("      其中 k 是使得 max(|x'|) < 1 的最小整数")
    log_print("结果：所有值移动到 [-1, 1] 之间")
    log_print("")
    log_print("优点：")
    log_print("  ✓ 简单易懂")
    log_print("  ✓ 保留原始数据的大小关系")
    log_print("")
    log_print("缺点：")
    log_print("  ✗ 不改变分布形状")
    log_print("  ✗ 对极端值敏感")
    log_print("  ✗ 实际应用较少")
    log_print("")
    log_print("适用场景：")
    log_print("  - 数据范围确定，需要简单缩放")
    log_print("  - 主要用于教学演示\n")

# ============================================================
# 识别需要规范化的变量
# ============================================================
def identify_variables_for_normalization(df):
    """识别需要规范化的变量"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 3.1.2】识别需要规范化的变量")
    log_print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    log_print(f"\n【数值型变量列表】共 {len(numeric_cols)} 个")
    log_print("-" * 60)
    
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        std_val = df[col].std()
        log_print(f"\n{col}:")
        log_print(f"  范围：[{min_val}, {max_val}]")
        log_print(f"  均值：{mean_val:.2f}")
        log_print(f"  标准差：{std_val:.2f}")
    
    log_print("\n\n【变量分类与决策】")
    log_print("=" * 80)
    
    # 需要规范化的变量
    continuous_vars = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    log_print("\n✅ 需要规范化的变量（5个连续变量）：")
    log_print("-" * 60)
    
    reasons = {
        'age': '年龄（17-80），与其他特征量纲不同',
        'fnlwgt': '人口权重（12285-1490400），数值极大，量纲差异显著',
        'capital-gain': '投资收益（0-99999），91.7%为0，极端偏态分布',
        'capital-loss': '投资损失（0-4356），95.3%为0，极端偏态分布',
        'hours-per-week': '每周工时（5-80），与其他特征量纲不同'
    }
    
    for idx, var in enumerate(continuous_vars, 1):
        log_print(f"  {idx}. {var}")
        log_print(f"     理由：{reasons[var]}")
    
    log_print("\n\n❌ 不规范化的变量（1个有序分类变量）：")
    log_print("-" * 60)
    log_print("  1. education-num")
    log_print("     类型：有序分类变量的数值编码")
    log_print("     范围：[1, 16]")
    log_print("     含义：教育等级（1=学前班, 9=高中, 13=本科, 16=博士）")
    log_print("     决策：保持原始编码（1-16），不进行规范化")
    log_print("")
    log_print("     理由：")
    log_print("       1. education-num 不是真正的连续变量，而是等级编码")
    log_print("       2. 1-16的等级关系有明确业务含义，规范化会破坏这种关系")
    log_print("       3. 在决策树、随机森林等算法中，保持等级编码是常见做法")
    log_print("       4. 数值范围[1,16]本身就很小，与[0,1]或[-3,3]差异不大")
    log_print("")
    
    return continuous_vars

# ============================================================
# 数据特点分析
# ============================================================
def analyze_data_characteristics(df, continuous_vars):
    """分析数据特点，为方法选择提供依据"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 3.1.3】数据特点分析（为方法选择提供依据）")
    log_print("=" * 80)
    
    log_print("\n【关键数据特征统计】")
    log_print("-" * 60)
    
    for var in continuous_vars:
        data = df[var]
        
        # 统计零值比例（针对capital-gain/loss）
        zero_pct = (data == 0).sum() / len(data) * 100 if var.startswith('capital') else 0
        
        # 偏度（衡量分布对称性）
        skewness = data.skew()
        
        # 峰度（衡量分布尾部）
        kurtosis = data.kurtosis()
        
        log_print(f"\n{var}:")
        log_print(f"  最小值：{data.min()}")
        log_print(f"  最大值：{data.max()}")
        log_print(f"  极差：{data.max() - data.min()}")
        log_print(f"  均值：{data.mean():.2f}")
        log_print(f"  中位数：{data.median():.2f}")
        log_print(f"  标准差：{data.std():.2f}")
        
        if var.startswith('capital'):
            log_print(f"  零值占比：{zero_pct:.2f}%")
        
        log_print(f"  偏度：{skewness:.2f} ({'右偏' if skewness > 0 else '左偏' if skewness < 0 else '对称'})")
        log_print(f"  峰度：{kurtosis:.2f}")
        
        # 判断分布类型
        if abs(skewness) > 1:
            dist_type = "高度偏态"
        elif abs(skewness) > 0.5:
            dist_type = "中等偏态"
        else:
            dist_type = "接近对称"
        
        log_print(f"  分布特征：{dist_type}")
    
    log_print("\n\n【数据特点总结】")
    log_print("-" * 60)
    log_print("✓ capital-gain 和 capital-loss 存在严重的右偏态分布")
    log_print("  - 大量零值（91.7%和95.3%）")
    log_print("  - 少数极端高值")
    log_print("  - 这是Adult数据集的天然特征，反映了真实社会现象")
    log_print("")
    log_print("✓ fnlwgt 的数值范围极大（12285-1490400）")
    log_print("  - 与其他特征的量纲差异达到万倍级别")
    log_print("")
    log_print("✓ age 和 hours-per-week 相对正常")
    log_print("  - 分布较为对称，但量纲不同")
    log_print("")
    log_print("【对方法选择的启示】")
    log_print("-" * 60)
    log_print("⚠️  Min-Max 归一化的问题：")
    log_print("  - capital-gain 的 91.7% 零值会被映射到 0")
    log_print("  - 少数极值（如99999）会被映射到 1")
    log_print("  - 中间大量正常值会被压缩到很小的区间")
    log_print("  - 结果：失去区分度，信息损失严重")
    log_print("")
    log_print("✓ Z-score 标准化的优势：")
    log_print("  - 对极端值相对稳健")
    log_print("  - 保留了数据的分布形状")
    log_print("  - 大量零值和极端值都能得到合理的表示")
    log_print("  - 结论：更适合本数据集\n")

# ============================================================
# 三种方法对比实验
# ============================================================
def compare_normalization_methods(df, var='capital-gain'):
    """对比三种规范化方法（以capital-gain为例）"""
    log_print("\n" + "=" * 80)
    log_print(f"【阶段 3.1.4】三种规范化方法对比实验（以 {var} 为例）")
    log_print("=" * 80)
    
    log_print(f"\n选择 {var} 作为对比示例的原因：")
    log_print("-" * 60)
    log_print("1. 该变量最具代表性（91.7%零值 + 极端高值）")
    log_print("2. 最能体现不同方法的优缺点")
    log_print("3. 对方法选择最有指导意义\n")
    
    data = df[var].values.reshape(-1, 1)
    
    # 方法1：Min-Max
    log_print("\n【方法1：Min-Max 归一化】")
    log_print("-" * 60)
    minmax_scaler = MinMaxScaler()
    data_minmax = minmax_scaler.fit_transform(data).flatten()
    
    log_print(f"公式：x' = (x - {df[var].min()}) / ({df[var].max()} - {df[var].min()})")
    log_print(f"")
    log_print(f"结果统计：")
    log_print(f"  范围：[{data_minmax.min():.6f}, {data_minmax.max():.6f}]")
    log_print(f"  均值：{data_minmax.mean():.6f}")
    log_print(f"  标准差：{data_minmax.std():.6f}")
    log_print(f"  零值映射到：{0:.6f}")
    log_print(f"  最大值映射到：{1:.6f}")
    log_print(f"")
    log_print(f"问题分析：")
    log_print(f"  ⚠️  91.7% 的零值都被映射到 0.0")
    log_print(f"  ⚠️  少数极值占据了大部分 [0, 1] 空间")
    log_print(f"  ⚠️  大量中等收益值（如5000-20000）被压缩到 0.05-0.20 的狭小区间")
    log_print(f"  ⚠️  区分度严重下降，信息损失明显")
    
    # 方法2：Z-score
    log_print("\n\n【方法2：Z-score 标准化】")
    log_print("-" * 60)
    zscore_scaler = StandardScaler()
    data_zscore = zscore_scaler.fit_transform(data).flatten()
    
    log_print(f"公式：x' = (x - {df[var].mean():.2f}) / {df[var].std():.2f}")
    log_print(f"")
    log_print(f"结果统计：")
    log_print(f"  范围：[{data_zscore.min():.6f}, {data_zscore.max():.6f}]")
    log_print(f"  均值：{data_zscore.mean():.6f}  （≈ 0）")
    log_print(f"  标准差：{data_zscore.std():.6f}  （≈ 1）")
    log_print(f"  零值映射到：{-df[var].mean()/df[var].std():.6f}")
    log_print(f"  最大值映射到：{(df[var].max()-df[var].mean())/df[var].std():.6f}")
    log_print(f"")
    log_print(f"优势分析：")
    log_print(f"  ✓ 零值被映射到 {-df[var].mean()/df[var].std():.2f}，仍保留区分度")
    log_print(f"  ✓ 极端值被映射到 {(df[var].max()-df[var].mean())/df[var].std():.2f}，标记为异常但不过度影响")
    log_print(f"  ✓ 中间值分布在合理范围内，保持了原有的相对关系")
    log_print(f"  ✓ 保留了数据的分布特征（右偏态）")
    
    # 方法3：小数定标
    log_print("\n\n【方法3：小数定标规范化】")
    log_print("-" * 60)
    max_abs = np.abs(df[var]).max()
    k = int(np.ceil(np.log10(max_abs)))
    data_decimal = df[var] / (10 ** k)
    
    log_print(f"公式：x' = x / 10^{k}")
    log_print(f"      其中 k={k}，使得 max(|x'|) = {max_abs/(10**k):.6f} < 1")
    log_print(f"")
    log_print(f"结果统计：")
    log_print(f"  范围：[{data_decimal.min():.6f}, {data_decimal.max():.6f}]")
    log_print(f"  均值：{data_decimal.mean():.6f}")
    log_print(f"  标准差：{data_decimal.std():.6f}")
    log_print(f"")
    log_print(f"局限性：")
    log_print(f"  ⚠️  只是简单地除以 10^{k}，没有真正改变分布")
    log_print(f"  ⚠️  极端值问题依然存在")
    log_print(f"  ⚠️  实际应用价值有限")
    
    # 绘制对比图
    plot_comparison(df[var], data_minmax, data_zscore, data_decimal, var)
    
    return data_minmax, data_zscore, data_decimal

def plot_comparison(original, minmax, zscore, decimal, var_name):
    """绘制三种方法的对比图"""
    log_print("\n\n【绘制对比可视化图表】")
    log_print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 原始数据
    axes[0, 0].hist(original, bins=50, color='gray', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'原始数据：{var_name}', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('原始值', fontsize=10)
    axes[0, 0].set_ylabel('频数', fontsize=10)
    axes[0, 0].axvline(original.mean(), color='red', linestyle='--', linewidth=2, label=f'均值={original.mean():.0f}')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Min-Max
    axes[0, 1].hist(minmax, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Min-Max 归一化', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('归一化值 [0, 1]', fontsize=10)
    axes[0, 1].set_ylabel('频数', fontsize=10)
    axes[0, 1].axvline(minmax.mean(), color='red', linestyle='--', linewidth=2, label=f'均值={minmax.mean():.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Z-score
    axes[1, 0].hist(zscore, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Z-score 标准化 ✅ 推荐', fontsize=12, fontweight='bold', color='green')
    axes[1, 0].set_xlabel('标准化值', fontsize=10)
    axes[1, 0].set_ylabel('频数', fontsize=10)
    axes[1, 0].axvline(zscore.mean(), color='red', linestyle='--', linewidth=2, label=f'均值≈{zscore.mean():.3f}')
    axes[1, 0].axvline(0, color='blue', linestyle='-', linewidth=1, alpha=0.5, label='零线')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 小数定标
    axes[1, 1].hist(decimal, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('小数定标规范化', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('规范化值', fontsize=10)
    axes[1, 1].set_ylabel('频数', fontsize=10)
    axes[1, 1].axvline(decimal.mean(), color='red', linestyle='--', linewidth=2, label=f'均值={decimal.mean():.3f}')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'三种规范化方法对比（{var_name}）\n数据特点：91.7%零值 + 极端高值', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('图4_规范化方法对比.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    log_print("✓ 对比图已保存：图4_规范化方法对比.png")
    log_print("  说明：可视化展示了三种方法对同一变量的处理效果\n")

# ============================================================
# 方法选择决策
# ============================================================
def make_normalization_decision():
    """基于数据特点做出方法选择"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 3.1.5】规范化方法选择决策")
    log_print("=" * 80)
    
    log_print("\n【决策依据】")
    log_print("-" * 60)
    log_print("基于 Adult Income 数据集的实际特点：")
    log_print("")
    log_print("1. 数据分布特征：")
    log_print("   - capital-gain/loss 严重右偏态（大量零值+极端高值）")
    log_print("   - fnlwgt 数值范围极大（量纲差异万倍）")
    log_print("   - 这些是数据的天然特征，不是质量问题")
    log_print("")
    log_print("2. 三种方法的表现：")
    log_print("   - Min-Max：极端值导致大量正常值被压缩，区分度严重下降 ❌")
    log_print("   - Z-score：对极端值稳健，保留分布特征，区分度良好 ✅")
    log_print("   - 小数定标：效果有限，实际应用价值不大 ❌")
    log_print("")
    log_print("3. 建模需求：")
    log_print("   - 需要统一不同特征的量纲")
    log_print("   - 需要保留原始数据的分布特征")
    log_print("   - 需要对极端值有一定的稳健性")
    log_print("")
    log_print("【最终决策】")
    log_print("=" * 80)
    log_print("")
    log_print("✅ 采用：Z-score 标准化（StandardScaler）")
    log_print("")
    log_print("【应用范围】")
    log_print("-" * 60)
    log_print("统一对以下 5 个连续变量进行 Z-score 标准化：")
    log_print("  1. age")
    log_print("  2. fnlwgt")
    log_print("  3. capital-gain")
    log_print("  4. capital-loss")
    log_print("  5. hours-per-week")
    log_print("")
    log_print("保持原样的变量：")
    log_print("  - education-num（有序分类变量，不规范化）")
    log_print("")
    log_print("【预期效果】")
    log_print("-" * 60)
    log_print("✓ 所有规范化变量的均值 ≈ 0")
    log_print("✓ 所有规范化变量的标准差 ≈ 1")
    log_print("✓ 消除了量纲差异")
    log_print("✓ 保留了分布特征\n")

# ============================================================
# 执行统一规范化
# ============================================================
def apply_zscore_normalization(df, continuous_vars):
    """对选定变量应用Z-score标准化"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 3.1.6】执行 Z-score 标准化")
    log_print("=" * 80)
    
    log_print(f"\n【开始规范化】对 {len(continuous_vars)} 个变量进行 Z-score 标准化")
    log_print("-" * 60)
    
    df_normalized = df.copy()
    
    # 统计信息（规范化前）
    log_print("\n【规范化前的统计信息】")
    log_print("-" * 60)
    stats_before = []
    
    for var in continuous_vars:
        mean_val = df[var].mean()
        std_val = df[var].std()
        min_val = df[var].min()
        max_val = df[var].max()
        
        stats_before.append({
            '变量': var,
            '均值': mean_val,
            '标准差': std_val,
            '最小值': min_val,
            '最大值': max_val
        })
        
        log_print(f"\n{var}:")
        log_print(f"  均值 = {mean_val:.2f}")
        log_print(f"  标准差 = {std_val:.2f}")
        log_print(f"  范围 = [{min_val}, {max_val}]")
    
    # 执行标准化
    log_print("\n\n【执行标准化操作】")
    log_print("-" * 60)
    
    scaler = StandardScaler()
    df_normalized[continuous_vars] = scaler.fit_transform(df[continuous_vars])
    
    log_print("✓ Z-score 标准化完成")
    log_print(f"  公式：x' = (x - μ) / σ")
    log_print(f"  应用于：{', '.join(continuous_vars)}\n")
    
    # 统计信息（规范化后）
    log_print("【规范化后的统计信息】")
    log_print("-" * 60)
    stats_after = []
    
    for var in continuous_vars:
        mean_val = df_normalized[var].mean()
        std_val = df_normalized[var].std()
        min_val = df_normalized[var].min()
        max_val = df_normalized[var].max()
        
        stats_after.append({
            '变量': var,
            '均值': mean_val,
            '标准差': std_val,
            '最小值': min_val,
            '最大值': max_val
        })
        
        log_print(f"\n{var}:")
        log_print(f"  均值 = {mean_val:.6f}  （≈ 0 ✓）")
        log_print(f"  标准差 = {std_val:.6f}  （≈ 1 ✓）")
        log_print(f"  范围 = [{min_val:.2f}, {max_val:.2f}]")
    
    # 生成对比表格
    log_print("\n\n【表格：规范化前后对比】")
    log_print("=" * 80)
    
    comparison = []
    for before, after in zip(stats_before, stats_after):
        comparison.append({
            '变量': before['变量'],
            '规范化前均值': before['均值'],
            '规范化后均值': after['均值'],
            '规范化前标准差': before['标准差'],
            '规范化后标准差': after['标准差'],
            '规范化前范围': f"[{before['最小值']}, {before['最大值']}]",
            '规范化后范围': f"[{after['最小值']:.2f}, {after['最大值']:.2f}]"
        })
    
    comparison_df = pd.DataFrame(comparison)
    log_print("\n" + comparison_df.to_string(index=False))
    log_print("")
    
    # 验证 education-num 未被改变
    log_print("\n【验证：education-num 保持原样】")
    log_print("-" * 60)
    
    if 'education-num' in df.columns:
        unchanged = (df['education-num'] == df_normalized['education-num']).all()
        if unchanged:
            log_print("✓ education-num 保持原始值 [1, 16]，未被规范化")
            log_print(f"  原始均值：{df['education-num'].mean():.2f}")
            log_print(f"  当前均值：{df_normalized['education-num'].mean():.2f}")
            log_print(f"  两者一致：{unchanged}\n")
    
    return df_normalized

# ============================================================
# 保存结果
# ============================================================
def save_normalized_data(df_original, df_normalized):
    """保存规范化后的数据"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 3.1.7】保存规范化后的数据")
    log_print("=" * 80)
    
    output_file = 'normalized_data.csv'
    df_normalized.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    log_print(f"\n✓ 规范化后的数据已保存：{output_file}")
    log_print(f"  数据规模：{df_normalized.shape[0]:,} 行 × {df_normalized.shape[1]} 列")
    log_print("")
    log_print("【数据变化对比】")
    log_print("-" * 60)
    log_print(f"规范化前：{df_original.shape[0]:,} 行 × {df_original.shape[1]} 列")
    log_print(f"规范化后：{df_normalized.shape[0]:,} 行 × {df_normalized.shape[1]} 列")
    log_print(f"")
    log_print(f"【变化说明】")
    log_print(f"  ✓ 样本数不变：{df_original.shape[0]:,}")
    log_print(f"  ✓ 特征数不变：{df_original.shape[1]}")
    log_print(f"  ✓ 5个连续变量已规范化（均值≈0，标准差≈1）")
    log_print(f"  ✓ education-num 保持原样")
    log_print(f"  ✓ 所有分类变量未改变\n")

# ============================================================
# 总结报告
# ============================================================
def summary_report():
    """生成总结报告"""
    log_print("\n" + "=" * 80)
    log_print("【数据规范化总结报告】")
    log_print("=" * 80)
    
    log_print("\n【实验概况】")
    log_print("-" * 60)
    log_print("目标：消除特征间的量纲差异，统一数据尺度")
    log_print("方法对比：Min-Max、Z-score、小数定标")
    log_print("最终选择：Z-score 标准化")
    log_print("")
    
    log_print("【关键决策】")
    log_print("-" * 60)
    log_print("✅ 规范化变量：5个（age, fnlwgt, capital-gain, capital-loss, hours-per-week）")
    log_print("❌ 不规范化变量：1个（education-num，保持等级编码）")
    log_print("✅ 统一方法：Z-score（均值0，标准差1）")
    log_print("")
    
    log_print("【选择 Z-score 的理由】")
    log_print("-" * 60)
    log_print("1. Adult 数据集存在严重偏态分布（capital-gain/loss）")
    log_print("2. Z-score 对极端值相对稳健，不会过度压缩正常值")
    log_print("3. 保留了数据的原始分布特征")
    log_print("4. 适合后续的机器学习建模")
    log_print("")
    log_print("【处理效果】")
    log_print("-" * 60)
    log_print("✓ 所有规范化变量均值 ≈ 0")
    log_print("✓ 所有规范化变量标准差 ≈ 1")
    log_print("✓ 消除了量纲差异（fnlwgt不再主导）")
    log_print("✓ 保留了数据分布特征（偏态依然存在，但尺度统一）")
    log_print("")
    log_print("【下一步工作】")
    log_print("-" * 60)
    log_print("1. 进行 PCA 主成分分析降维")
    log_print("2. 特征构造（交互特征、哑编码）")
    log_print("3. 准备最终建模数据集")
    log_print("")
    log_print("✅ 数据规范化完成！\n")

# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "=" * 80)
    print("【数据预处理实验 - 步骤3.1：数据规约 - 数据规范化】")
    print("=" * 80 + "\n")
    
    init_log()
    
    # 1. 加载数据
    df = load_integrated_data()
    if df is None:
        return
    
    df_original = df.copy()
    
    # 2. 理论说明
    normalization_theory()
    
    # 3. 识别需要规范化的变量
    continuous_vars = identify_variables_for_normalization(df)
    
    # 4. 数据特点分析
    analyze_data_characteristics(df, continuous_vars)
    
    # 5. 三种方法对比
    _, _, _ = compare_normalization_methods(df, var='capital-gain')
    
    # 6. 方法选择决策
    make_normalization_decision()
    
    # 7. 执行Z-score标准化
    df_normalized = apply_zscore_normalization(df, continuous_vars)
    
    # 8. 保存结果
    save_normalized_data(df_original, df_normalized)
    
    # 9. 总结报告
    summary_report()
    
    log_print(f"执行结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80 + "\n")
    
    print("\n" + "=" * 80)
    print("✅ 数据规范化完成！")
    print("=" * 80)
    print("\n生成的文件：")
    print("  1. step3_reduction_part1_log.txt - 详细分析日志")
    print("  2. 图4_规范化方法对比.png - 三种方法对比图")
    print("  3. normalized_data.csv - 规范化后的数据")
    print("\n下一步：进行 PCA 主成分分析降维\n")

if __name__ == "__main__":
    main()


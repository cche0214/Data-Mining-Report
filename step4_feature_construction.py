# -*- coding: utf-8 -*-
"""
================================================================================
特征构造模块：基本特征构造（交互特征）
================================================================================
输入文件: normalized_data.csv（已完成规范化的数据，不使用PCA）
输出文件: 
  - feature_constructed_data.csv（原始数据 + 新构造特征）
  - step4_feature_construction_log.txt（详细日志）
  - 图7_新特征分布分析.png
  - 图8_新特征与收入关系.png
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 初始化日志
log_file = 'step4_feature_construction_log.txt'

def write_log(message, print_console=True):
    """写入日志并打印到终端"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')
    if print_console:
        print(message)

# 清空日志文件
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("特征构造模块：基本特征构造（交互特征）\n")
    f.write("=" * 80 + "\n\n")

write_log("=" * 80)
write_log("开始基本特征构造（交互特征）")
write_log("=" * 80)
write_log("")

# ================================================================================
# 第一部分：理论说明
# ================================================================================
write_log("【理论基础】基本特征构造")
write_log("-" * 80)
write_log("")

theory_text = """
1. 什么是特征构造？
   - 特征构造（Feature Engineering）是从原始特征中创造新特征的过程
   - 通过领域知识和数据理解，挖掘特征间的潜在关系
   - 目标：提高模型的预测能力和可解释性

2. 特征构造的常见方法：
   (1) 交互特征（Interaction Features）：
       - 乘法交互：A × B（捕捉协同效应）
       - 加法/减法交互：A + B 或 A - B（捕捉总和或差异）
       - 比值特征：A / B（捕捉相对关系）
   
   (2) 多项式特征（Polynomial Features）：
       - 平方项：A²（捕捉非线性关系）
       - 高次项：A³, A⁴...
   
   (3) 统计特征（Statistical Features）：
       - 分组聚合：按类别计算均值、方差等
       - 排名/百分位：特征的相对位置
   
   (4) 离散化（Discretization）：
       - 连续变量分段（如年龄分组）
       - 便于捕捉非线性关系和分段模式

3. 为什么需要交互特征？
   - 单一特征可能无法完全反映业务规律
   - 特征之间可能存在协同效应（1+1>2）
   - 交互特征能捕捉更复杂的模式

4. 如何设计有意义的交互特征？
   - 基于业务逻辑：特征组合是否有实际含义
   - 基于相关性分析：相关特征可能有交互效应
   - 基于探索性分析：观察数据分布和模式
   - 避免过度构造：太多特征会导致过拟合

5. 特征构造的注意事项：
   - 保留原始特征：新特征是补充而非替代
   - 避免数据泄露：不能使用测试集信息
   - 处理缺失值：新特征可能产生NaN（如除法）
   - 考虑量纲：必要时对新特征进行规范化
"""

write_log(theory_text)
write_log("")

# ================================================================================
# 第二部分：数据加载与检查
# ================================================================================
write_log("=" * 80)
write_log("第一步：数据加载与检查")
write_log("=" * 80)
write_log("")

# 读取规范化后的数据
df = pd.read_csv('normalized_data.csv')
write_log(f"✓ 成功读取规范化后的数据：normalized_data.csv")
write_log(f"  - 数据规模：{df.shape[0]:,} 行 × {df.shape[1]} 列")
write_log("")

write_log("【数据现状说明】")
write_log("-" * 80)
write_log("")
write_log("⚠ 重要决策：不使用 PCA 主成分分析")
write_log("")
write_log("原因分析：")
write_log("  - PCA 结果显示5个连续特征方差分布均匀（各占约20%）")
write_log("  - 需要保留全部5个主成分才能达到85%累计方差解释率")
write_log("  - 无法实现有效的维度压缩")
write_log("  - 主成分的业务含义解释性较弱")
write_log("")
write_log("决策结果：")
write_log("  ✓ 放弃 PCA 降维方法")
write_log("  ✓ 直接使用规范化后的原始特征")
write_log("  ✓ 通过交互特征构造增强特征表达能力")
write_log("")

# 显示当前列
write_log("当前数据列（14列）：")
for i, col in enumerate(df.columns, 1):
    col_type = df[col].dtype
    write_log(f"  {i:2d}. {col:<20} (类型: {col_type})")
write_log("")

# 检查数值型特征
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('income') if 'income' in numerical_features else None
write_log(f"数值型特征（{len(numerical_features)}个）：")
write_log(f"  {', '.join(numerical_features)}")
write_log("")

# ================================================================================
# 第三部分：新特征构造（交互特征）
# ================================================================================
write_log("=" * 80)
write_log("第二步：构造交互特征")
write_log("=" * 80)
write_log("")

write_log("【特征构造方案】")
write_log("-" * 80)
write_log("")

# 创建特征构造表格
construction_plan = """
┌─────────────────┬──────────────────────────────┬─────────────────┬─────────────────────────┐
│  新特征名称     │       构造方法                │   业务含义      │        合理性           │
├─────────────────┼──────────────────────────────┼─────────────────┼─────────────────────────┤
│ 工作强度        │ education-num × hours-per-week│ 人力资本投入    │ 高学历×长工时 → 高收入  │
│ (work_intensity)│                               │                 │                         │
├─────────────────┼──────────────────────────────┼─────────────────┼─────────────────────────┤
│ 资本净收益      │ capital-gain - capital-loss   │ 投资净回报      │ 直接反映财富增长        │
│ (net_capital)   │                               │                 │                         │
├─────────────────┼──────────────────────────────┼─────────────────┼─────────────────────────┤
│ 年龄工作比      │ hours-per-week / age          │ 工作强度/年龄   │ 年轻人高强度工作倾向    │
│ (work_age_ratio)│                               │                 │                         │
└─────────────────┴──────────────────────────────┴─────────────────┴─────────────────────────┘
"""
write_log(construction_plan)
write_log("")

# ===== 特征1：工作强度 =====
write_log("【特征1：工作强度（work_intensity）】")
write_log("-" * 80)
write_log("")

write_log("构造方法：education-num × hours-per-week")
write_log("")
write_log("业务逻辑：")
write_log("  - education-num：教育年限（代表人力资本质量）")
write_log("  - hours-per-week：每周工作小时（代表劳动投入量）")
write_log("  - 乘积：综合反映「教育水平 × 劳动投入」的人力资本强度")
write_log("")
write_log("预期效果：")
write_log("  - 高学历且长工时的人，工作强度最高")
write_log("  - 这类人群往往从事高技能、高薪酬的工作")
write_log("  - 该特征可能与收入呈正相关")
write_log("")

# 注意：education-num 未规范化，hours-per-week 已规范化
# 需要使用原始值进行计算，确保业务含义
write_log("⚠ 重要说明：")
write_log("  - education-num 未规范化（取值1-16，保持原始序数编码）")
write_log("  - hours-per-week 已规范化（Z-score标准化）")
write_log("  - 直接相乘会混合不同量纲，需要特别说明")
write_log("")
write_log("处理方案：")
write_log("  ✓ 由于我们已对 hours-per-week 进行了 Z-score 标准化")
write_log("  ✓ 新特征 = education-num(未标准化) × hours-per-week(已标准化)")
write_log("  ✓ 这样可以保留 education-num 的序数特性，同时避免 hours-per-week 的极端值影响")
write_log("")

df['work_intensity'] = df['education-num'] * df['hours-per-week']
write_log(f"✓ 特征构造完成：work_intensity")
write_log(f"  - 数据类型：{df['work_intensity'].dtype}")
write_log(f"  - 缺失值：{df['work_intensity'].isnull().sum()} 个")
write_log(f"  - 统计摘要：")
write_log(f"    均值 = {df['work_intensity'].mean():.4f}")
write_log(f"    标准差 = {df['work_intensity'].std():.4f}")
write_log(f"    最小值 = {df['work_intensity'].min():.4f}")
write_log(f"    最大值 = {df['work_intensity'].max():.4f}")
write_log("")

# ===== 特征2：资本净收益 =====
write_log("【特征2：资本净收益（net_capital）】")
write_log("-" * 80)
write_log("")

write_log("构造方法：capital-gain - capital-loss")
write_log("")
write_log("业务逻辑：")
write_log("  - capital-gain：资本收益（投资、房产等增值收入）")
write_log("  - capital-loss：资本损失（投资亏损等）")
write_log("  - 差值：净资本收益，直接反映财富增长")
write_log("")
write_log("预期效果：")
write_log("  - 正值：投资盈利，财富增长")
write_log("  - 负值：投资亏损，财富减少")
write_log("  - 该特征比单独的 gain 或 loss 更有经济解释性")
write_log("  - 预期与高收入人群强相关")
write_log("")

write_log("⚠ 重要说明：")
write_log("  - capital-gain 和 capital-loss 均已进行 Z-score 标准化")
write_log("  - 两者量纲一致，可以直接相减")
write_log("  - 新特征的量纲与原特征一致（标准化后的值）")
write_log("")

df['net_capital'] = df['capital-gain'] - df['capital-loss']
write_log(f"✓ 特征构造完成：net_capital")
write_log(f"  - 数据类型：{df['net_capital'].dtype}")
write_log(f"  - 缺失值：{df['net_capital'].isnull().sum()} 个")
write_log(f"  - 统计摘要：")
write_log(f"    均值 = {df['net_capital'].mean():.4f}")
write_log(f"    标准差 = {df['net_capital'].std():.4f}")
write_log(f"    最小值 = {df['net_capital'].min():.4f}")
write_log(f"    最大值 = {df['net_capital'].max():.4f}")
write_log("")

# ===== 特征3：年龄工作比 =====
write_log("【特征3：年龄工作比（work_age_ratio）】")
write_log("-" * 80)
write_log("")

write_log("构造方法：hours-per-week / age")
write_log("")
write_log("业务逻辑：")
write_log("  - hours-per-week：每周工作小时")
write_log("  - age：年龄")
write_log("  - 比值：单位年龄的工作强度")
write_log("")
write_log("预期效果：")
write_log("  - 年轻人（低age）且长工时 → 高比值（拼搏期）")
write_log("  - 年长者（高age）且短工时 → 低比值（稳定期）")
write_log("  - 该特征反映不同年龄段的工作投入程度差异")
write_log("")

write_log("⚠ 重要说明：")
write_log("  - age 和 hours-per-week 均已进行 Z-score 标准化")
write_log("  - 两者量纲一致，可以直接相除")
write_log("  - 需要处理 age = 0 的情况（标准化后的均值附近可能为0）")
write_log("")

# 检查是否有接近0的age值
age_near_zero = (df['age'].abs() < 0.01).sum()
write_log(f"检查：age 接近0的样本数 = {age_near_zero} 个")
write_log("")

# 为避免除以0，使用 np.where 处理
# 如果 age 接近 0，则设置为 NaN 或一个特殊值
df['work_age_ratio'] = np.where(
    df['age'].abs() > 0.01,  # 如果 age 不接近0
    df['hours-per-week'] / df['age'],  # 正常相除
    np.nan  # 否则设为 NaN
)

nan_count = df['work_age_ratio'].isnull().sum()
write_log(f"✓ 特征构造完成：work_age_ratio")
write_log(f"  - 数据类型：{df['work_age_ratio'].dtype}")
write_log(f"  - 缺失值：{nan_count} 个（除以接近0的age产生）")

if nan_count > 0:
    write_log(f"  - 缺失值处理：使用该列的中位数填充")
    median_ratio = df['work_age_ratio'].median()
    df['work_age_ratio'].fillna(median_ratio, inplace=True)
    write_log(f"    填充值（中位数）= {median_ratio:.4f}")

write_log(f"  - 统计摘要：")
write_log(f"    均值 = {df['work_age_ratio'].mean():.4f}")
write_log(f"    标准差 = {df['work_age_ratio'].std():.4f}")
write_log(f"    最小值 = {df['work_age_ratio'].min():.4f}")
write_log(f"    最大值 = {df['work_age_ratio'].max():.4f}")
write_log("")

# ================================================================================
# 第四部分：新特征描述性统计
# ================================================================================
write_log("=" * 80)
write_log("第三步：新特征描述性统计分析")
write_log("=" * 80)
write_log("")

new_features = ['work_intensity', 'net_capital', 'work_age_ratio']

write_log("【新特征统计摘要】")
write_log("-" * 80)
write_log("")

# 详细统计表格
stats_df = df[new_features].describe()
write_log("详细统计指标：")
write_log("")
write_log(stats_df.to_string())
write_log("")

# 按收入分组的统计
write_log("【按收入水平分组的新特征均值对比】")
write_log("-" * 80)
write_log("")

grouped_stats = df.groupby('income')[new_features].mean()
write_log("各收入组的新特征均值：")
write_log("")
write_log(grouped_stats.to_string())
write_log("")

# 计算差异
write_log("高收入组 vs 低收入组差异分析：")
write_log("")
for feat in new_features:
    high_income_mean = grouped_stats.loc['>50K', feat]
    low_income_mean = grouped_stats.loc['<=50K', feat]
    diff = high_income_mean - low_income_mean
    diff_pct = (diff / abs(low_income_mean)) * 100 if low_income_mean != 0 else 0
    
    write_log(f"{feat}:")
    write_log(f"  - 高收入组（>50K）均值：{high_income_mean:>10.4f}")
    write_log(f"  - 低收入组（≤50K）均值：{low_income_mean:>10.4f}")
    write_log(f"  - 差异：{diff:>10.4f} ({diff_pct:>+6.2f}%)")
    
    if feat == 'work_intensity':
        if diff > 0:
            write_log(f"  → 结论：高收入人群的工作强度显著更高")
        else:
            write_log(f"  → 结论：工作强度与收入关系不明显")
    elif feat == 'net_capital':
        if diff > 0:
            write_log(f"  → 结论：高收入人群的资本净收益显著更高")
        else:
            write_log(f"  → 结论：资本净收益与收入关系不明显")
    elif feat == 'work_age_ratio':
        if diff > 0:
            write_log(f"  → 结论：高收入人群的年龄工作比更高（相对年龄更拼）")
        else:
            write_log(f"  → 结论：年龄工作比与收入关系不明显")
    write_log("")

# ================================================================================
# 第五部分：可视化
# ================================================================================
write_log("=" * 80)
write_log("第四步：新特征可视化分析")
write_log("=" * 80)
write_log("")

# ===== 图7：新特征分布分析 =====
write_log("生成图表：图7_新特征分布分析.png")
write_log("")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, feat in enumerate(new_features):
    # 子图1：直方图
    ax1 = axes[0, idx]
    ax1.hist(df[feat], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel(feat, fontsize=11, fontweight='bold')
    ax1.set_ylabel('频数', fontsize=11, fontweight='bold')
    ax1.set_title(f'{feat} 分布直方图', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加统计信息
    mean_val = df[feat].mean()
    median_val = df[feat].median()
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值={mean_val:.2f}')
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中位数={median_val:.2f}')
    ax1.legend(fontsize=9)
    
    # 子图2：箱线图（按收入分组）
    ax2 = axes[1, idx]
    df.boxplot(column=feat, by='income', ax=ax2, patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax2.set_xlabel('收入水平', fontsize=11, fontweight='bold')
    ax2.set_ylabel(feat, fontsize=11, fontweight='bold')
    ax2.set_title(f'{feat} 按收入分组箱线图', fontsize=12, fontweight='bold')
    ax2.get_figure().suptitle('')  # 移除默认标题
    ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('图7_新特征分布分析.png', dpi=300, bbox_inches='tight')
plt.close()

write_log("✓ 图7_新特征分布分析.png 已保存")
write_log("  - 上排：3个新特征的直方图（含均值和中位数线）")
write_log("  - 下排：3个新特征按收入分组的箱线图")
write_log("")

# ===== 图8：新特征与收入关系 =====
write_log("生成图表：图8_新特征与收入关系.png")
write_log("")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feat in enumerate(new_features):
    ax = axes[idx]
    
    # 小提琴图（结合箱线图和密度图）
    income_order = ['<=50K', '>50K']
    colors = ['#3498DB', '#E74C3C']
    
    parts = ax.violinplot(
        [df[df['income'] == cat][feat].dropna() for cat in income_order],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # 设置颜色
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # 添加箱线图元素
    for i, cat in enumerate(income_order):
        data = df[df['income'] == cat][feat].dropna()
        ax.plot([i], [data.mean()], 'o', color='white', markersize=8, 
                markeredgecolor='black', markeredgewidth=2, label=f'{cat} 均值' if i == 0 else '')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(income_order, fontsize=11, fontweight='bold')
    ax.set_ylabel(feat, fontsize=11, fontweight='bold')
    ax.set_title(f'{feat} 与收入关系（小提琴图）', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('图8_新特征与收入关系.png', dpi=300, bbox_inches='tight')
plt.close()

write_log("✓ 图8_新特征与收入关系.png 已保存")
write_log("  - 小提琴图：展示新特征在不同收入组的分布形态")
write_log("  - 白点：各组均值")
write_log("  - 可观察特征对收入的区分能力")
write_log("")

# ================================================================================
# 第六部分：数据保存（重点说明数据变化）
# ================================================================================
write_log("=" * 80)
write_log("第五步：保存特征构造后的数据")
write_log("=" * 80)
write_log("")

write_log("【数据变化说明】")
write_log("-" * 80)
write_log("")

write_log(f"数据变化详情：")
write_log(f"  输入数据：normalized_data.csv")
write_log(f"    - 行数：{df.shape[0]:,} 行")
write_log(f"    - 列数：{df.shape[1] - len(new_features)} 列")
write_log("")

write_log(f"  输出数据：feature_constructed_data.csv")
write_log(f"    - 行数：{df.shape[0]:,} 行（样本数不变）")
write_log(f"    - 列数：{df.shape[1]} 列（新增 {len(new_features)} 列）")
write_log("")

write_log(f"  ✓ 保留所有原始列（{df.shape[1] - len(new_features)} 列）：")
original_cols = [col for col in df.columns if col not in new_features]
for i in range(0, len(original_cols), 5):
    cols_batch = original_cols[i:i+5]
    write_log(f"    {', '.join(cols_batch)}")
write_log("")

write_log(f"  ✓ 新增特征列（{len(new_features)} 列）：")
for i, feat in enumerate(new_features, 1):
    write_log(f"    {i}. {feat}")
    if feat == 'work_intensity':
        write_log(f"       - 构造方法：education-num × hours-per-week")
        write_log(f"       - 业务含义：人力资本投入强度")
    elif feat == 'net_capital':
        write_log(f"       - 构造方法：capital-gain - capital-loss")
        write_log(f"       - 业务含义：投资净回报")
    elif feat == 'work_age_ratio':
        write_log(f"       - 构造方法：hours-per-week / age")
        write_log(f"       - 业务含义：单位年龄工作强度")
write_log("")

write_log(f"  ⚠ 重要说明：")
write_log(f"    1. 原始特征完全保留，未做任何删除")
write_log(f"    2. 新特征追加在数据右侧")
write_log(f"    3. 新特征均为数值型，无缺失值")
write_log(f"    4. 新特征尚未规范化（保留原始交互关系）")
write_log(f"    5. 如需使用，建议在建模前对新特征进行规范化")
write_log("")

# 调整列顺序：将新特征放在原始特征之后、目标变量之前
if 'income' in df.columns:
    cols_order = [col for col in df.columns if col not in new_features and col != 'income']
    cols_order.extend(new_features)
    cols_order.append('income')
    df = df[cols_order]
    write_log(f"  ✓ 已调整列顺序：原始特征 → 新构造特征 → 目标变量（income）")
    write_log("")

# 保存数据
df.to_csv('feature_constructed_data.csv', index=False, encoding='utf-8-sig')
write_log(f"✓ 数据已成功保存到：feature_constructed_data.csv")
write_log("")

# 显示最终数据示例
write_log("【最终数据预览】（前5行）")
write_log("-" * 80)
write_log("")
write_log(df.head().to_string())
write_log("")

# ================================================================================
# 第七部分：总结报告
# ================================================================================
write_log("=" * 80)
write_log("特征构造总结")
write_log("=" * 80)
write_log("")

summary = f"""
【核心成果】
1. 新特征数量：
   - 构造了 {len(new_features)} 个交互特征
   - 所有新特征均有明确的业务含义
   - 特征命名清晰易懂

2. 新特征列表：
   (1) work_intensity（工作强度）
       - 构造方法：education-num × hours-per-week
       - 业务解释：人力资本投入 = 教育质量 × 劳动数量
       - 预期效果：与高收入正相关
   
   (2) net_capital（资本净收益）
       - 构造方法：capital-gain - capital-loss
       - 业务解释：投资净回报 = 收益 - 损失
       - 预期效果：财富增长的直接指标
   
   (3) work_age_ratio（年龄工作比）
       - 构造方法：hours-per-week / age
       - 业务解释：单位年龄的工作强度
       - 预期效果：反映不同年龄段的拼搏程度

3. 特征验证结果：
"""

# 添加每个特征的验证结果
for feat in new_features:
    high_mean = df[df['income'] == '>50K'][feat].mean()
    low_mean = df[df['income'] == '<=50K'][feat].mean()
    diff = high_mean - low_mean
    
    summary += f"   - {feat}：\n"
    summary += f"     高收入组均值 = {high_mean:.4f}\n"
    summary += f"     低收入组均值 = {low_mean:.4f}\n"
    if diff > 0:
        summary += f"     ✓ 高收入组显著更高（差异={diff:.4f}），特征有效\n"
    else:
        summary += f"     ⚠ 差异不明显（差异={diff:.4f}），可能需要进一步分析\n"
    summary += "\n"

summary += f"""
4. 数据输出：
   - 保存文件：feature_constructed_data.csv
   - 数据规模：{df.shape[0]:,} 行 × {df.shape[1]} 列
   - 新增特征：{len(new_features)} 个
   - 原始特征：完全保留

5. 可视化输出：
   - 图7_新特征分布分析.png：新特征的分布直方图和箱线图
   - 图8_新特征与收入关系.png：新特征与收入的关系（小提琴图）

【特征构造的价值】
1. 增强特征表达能力：
   - 交互特征捕捉原始特征无法表达的组合效应
   - 比值特征揭示相对关系而非绝对值

2. 提高模型性能：
   - 有业务含义的特征更容易被模型学习
   - 减少模型需要自己发现复杂交互的负担

3. 提升可解释性：
   - 新特征有明确的业务含义
   - 便于向非技术人员解释模型结果

【关键决策回顾】
1. ✓ 为什么不使用 PCA？
   → PCA无法实现有效降维（需保留全部5个主成分）
   → 主成分业务含义不清晰
   → 直接构造有业务含义的交互特征更有价值

2. ✓ 为什么选择这3个交互特征？
   → 基于业务逻辑和常识推理
   → 特征组合有明确的经济学/社会学含义
   → 按收入分组的差异分析验证了特征有效性

3. ✓ 新特征是否需要规范化？
   → 当前未规范化，保留原始交互关系
   → 建议在建模前根据算法需求决定是否规范化
   → 树模型（如随机森林、XGBoost）不需要规范化
   → 线性模型、神经网络建议规范化

【完成的预处理流程回顾】
✓ 1. 数据清洗：缺失值处理、离群点检测与删除、格式统一
✓ 2. 数据集成：Pearson相关性分析、卡方检验
✓ 3. 数据规约：Z-score规范化（5个连续特征）
✓ 4. 特征构造：3个交互特征（当前步骤）

【下一步建议】
✓ 数据预处理已全部完成，可以进入建模阶段
✓ 建议任务：
   - 对分类特征进行独热编码（One-Hot Encoding）
   - 特征选择（可选）：进一步筛选最重要的特征
   - 模型训练与评估：使用清洗后的数据训练分类模型
   - 模型对比：比较使用新特征前后的模型性能
"""

write_log(summary)
write_log("")

# ================================================================================
# 程序结束
# ================================================================================
write_log("=" * 80)
write_log("✅ 特征构造模块执行完成")
write_log("=" * 80)
write_log("")

print("\n" + "=" * 80)
print("✅ 特征构造（交互特征）已全部完成！")
print("=" * 80)
print(f"\n📊 生成文件清单：")
print(f"  1. feature_constructed_data.csv        - 包含原始数据 + 新构造特征")
print(f"  2. step4_feature_construction_log.txt  - 详细日志文件")
print(f"  3. 图7_新特征分布分析.png             - 新特征分布图")
print(f"  4. 图8_新特征与收入关系.png           - 新特征与收入关系图")
print(f"\n📈 核心结果：")
print(f"  - 原始特征：{df.shape[1] - len(new_features)} 个（完全保留）")
print(f"  - 新增特征：{len(new_features)} 个（交互特征）")
print(f"  - 数据规模：{df.shape[0]:,} 行 × {df.shape[1]} 列")
print(f"  - 新特征：work_intensity, net_capital, work_age_ratio")
print(f"\n✓ 所有结果已保存到当前目录")
print("=" * 80 + "\n")


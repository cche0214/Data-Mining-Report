# -*- coding: utf-8 -*-
"""
【数据预处理实验 - 步骤2.2：数据集成 - 分类变量卡方检验】
目标：识别与目标变量显著相关的特征，检测特征间的冗余性
方法：卡方独立性检验 + Cramér's V 关联系数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ============================================================
# 日志系统
# ============================================================
log_file = 'step2_integration_part2_log.txt'

def init_log():
    """初始化日志"""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("【数据预处理实验 - 步骤2.2：数据集成】\n")
        f.write("第二部分：分类变量卡方检验（独立性检验）\n")
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
    log_print("【阶段 2.2.0】数据加载")
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
# 卡方检验理论
# ============================================================
def chi_square_theory():
    """卡方检验理论说明"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.1】卡方检验理论基础")
    log_print("=" * 80)
    
    log_print("\n【什么是卡方检验？】")
    log_print("-" * 60)
    log_print("卡方检验（Chi-Square Test）用于检验两个分类变量之间是否独立。")
    log_print("")
    log_print("【原假设与备择假设】")
    log_print("  H0（原假设）：两个变量相互独立（无关联）")
    log_print("  H1（备择假设）：两个变量不独立（存在关联）")
    log_print("")
    log_print("【卡方统计量计算公式】")
    log_print("  χ² = Σ [(观察频数 - 期望频数)² / 期望频数]")
    log_print("")
    log_print("  期望频数 = (行总计 × 列总计) / 总样本数")
    log_print("")
    log_print("【判断标准】")
    log_print("  - p < 0.001：高度显著（***），强烈拒绝独立性假设")
    log_print("  - p < 0.01 ：显著（**），拒绝独立性假设")
    log_print("  - p < 0.05 ：显著（*），拒绝独立性假设")
    log_print("  - p ≥ 0.05 ：不显著，接受独立性假设（变量独立）")
    log_print("")
    log_print("【Cramér's V 关联系数】")
    log_print("-" * 60)
    log_print("用于衡量两个分类变量之间的关联强度（类似于相关系数）")
    log_print("")
    log_print("计算公式：")
    log_print("  V = √[χ² / (n × min(r-1, c-1))]")
    log_print("  其中：n=样本量, r=行数, c=列数")
    log_print("")
    log_print("取值范围与含义：")
    log_print("  V ∈ [0, 1]")
    log_print("  V ≥ 0.7  ：强关联（高度冗余，考虑删除）⚠️")
    log_print("  0.3 < V < 0.7：中等关联")
    log_print("  V ≤ 0.3  ：弱关联")
    log_print("")
    log_print("【本实验的两个应用场景】")
    log_print("-" * 60)
    log_print("场景A：特征与目标变量的关联性检验（特征选择）")
    log_print("  - 检验哪些特征与 income（收入）显著相关")
    log_print("  - 保留显著特征，删除不显著特征")
    log_print("")
    log_print("场景B：特征之间的冗余性检验（冗余分析）")
    log_print("  - 检验特征对之间是否高度关联")
    log_print("  - 识别冗余特征对，删除其中一个\n")

def calculate_cramers_v(chi2, n, r, c):
    """计算 Cramér's V 系数"""
    return np.sqrt(chi2 / (n * min(r-1, c-1)))

# ============================================================
# 识别分类变量
# ============================================================
def identify_categorical_features(df):
    """识别分类变量"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.2】识别分类变量")
    log_print("=" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    log_print(f"\n【分类变量列表】共 {len(categorical_cols)} 个")
    log_print("-" * 60)
    
    for idx, col in enumerate(categorical_cols, 1):
        unique_count = df[col].nunique()
        log_print(f"  {idx}. {col:20s} - {unique_count:3d} 个类别")
    
    log_print("\n【变量分类】")
    log_print("-" * 60)
    
    # 目标变量
    target = 'income'
    log_print(f"目标变量（1个）：")
    log_print(f"  ⭐ {target} - 收入水平（<=50K 或 >50K）")
    log_print("")
    
    # 特征变量
    features = [col for col in categorical_cols if col != target]
    log_print(f"特征变量（{len(features)}个）：")
    for idx, feat in enumerate(features, 1):
        log_print(f"  {idx}. {feat}")
    
    log_print("")
    return features, target

# ============================================================
# 场景A：特征与目标变量的关联性检验
# ============================================================
def test_features_vs_target(df, features, target):
    """检验所有特征与目标变量的关联性"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.3】场景A：特征与目标变量关联性检验")
    log_print("=" * 80)
    
    log_print("\n【检验目的】")
    log_print("-" * 60)
    log_print("识别哪些特征与收入水平（income）显著相关")
    log_print("保留有预测价值的特征，删除无关特征\n")
    
    log_print(f"【检验对象】{len(features)} 个特征 × income")
    log_print("-" * 60)
    
    results = []
    
    for feat in features:
        log_print(f"\n正在检验：{feat} ↔ {target}")
        
        # 构造列联表
        contingency_table = pd.crosstab(df[feat], df[target])
        
        # 卡方检验
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # 计算 Cramér's V
        n = contingency_table.sum().sum()
        r, c = contingency_table.shape
        cramers_v = calculate_cramers_v(chi2, n, r, c)
        
        # 显著性判断
        if p_value < 0.001:
            significance = "***（高度显著）"
        elif p_value < 0.01:
            significance = "**（显著）"
        elif p_value < 0.05:
            significance = "*（显著）"
        else:
            significance = "不显著"
        
        # 关联强度
        if cramers_v >= 0.7:
            strength = "强关联"
        elif cramers_v >= 0.3:
            strength = "中等关联"
        else:
            strength = "弱关联"
        
        log_print(f"  χ² = {chi2:.2f}, p = {p_value:.6f}, df = {dof}")
        log_print(f"  Cramér's V = {cramers_v:.4f} [{strength}]")
        log_print(f"  显著性：{significance}")
        
        results.append({
            '特征': feat,
            '卡方值(χ²)': chi2,
            'p值': p_value,
            '自由度': dof,
            "Cramér's V": cramers_v,
            '显著性': significance,
            '关联强度': strength
        })
    
    results_df = pd.DataFrame(results)
    
    # 按卡方值降序排序
    results_df = results_df.sort_values('卡方值(χ²)', ascending=False).reset_index(drop=True)
    
    log_print("\n\n【表格1：特征与目标变量关联性检验结果汇总】")
    log_print("=" * 80)
    log_print("（按卡方值降序排列，卡方值越大表示关联性越强）\n")
    
    # 格式化输出表格
    table_str = results_df.to_string(index=False, 
                                     float_format=lambda x: f'{x:.2f}' if abs(x) > 1 else f'{x:.6f}')
    log_print(table_str)
    log_print("")
    
    return results_df

def analyze_feature_importance(results_df):
    """分析特征重要性"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.4】特征重要性分析")
    log_print("=" * 80)
    
    log_print("\n【特征重要性排名】（基于卡方值）")
    log_print("-" * 60)
    
    for idx, row in results_df.iterrows():
        rank = idx + 1
        feat = row['特征']
        chi2 = row['卡方值(χ²)']
        v = row["Cramér's V"]
        sig = row['显著性']
        
        log_print(f"第 {rank} 名：{feat:20s} (χ²={chi2:10.2f}, V={v:.4f}) {sig}")
    
    # 统计显著特征
    log_print("\n【显著性统计】")
    log_print("-" * 60)
    
    high_sig = len(results_df[results_df['p值'] < 0.001])
    sig = len(results_df[results_df['p值'] < 0.01]) - high_sig
    weak_sig = len(results_df[results_df['p值'] < 0.05]) - high_sig - sig
    not_sig = len(results_df[results_df['p值'] >= 0.05])
    
    log_print(f"  高度显著（p < 0.001）：{high_sig} 个")
    log_print(f"  显著（p < 0.01）    ：{sig} 个")
    log_print(f"  弱显著（p < 0.05）  ：{weak_sig} 个")
    log_print(f"  不显著（p ≥ 0.05）  ：{not_sig} 个")
    
    log_print("\n【特征选择建议】")
    log_print("-" * 60)
    
    insignificant_features = results_df[results_df['p值'] >= 0.05]['特征'].tolist()
    
    if len(insignificant_features) > 0:
        log_print(f"⚠️  发现 {len(insignificant_features)} 个不显著特征，建议删除：")
        for feat in insignificant_features:
            log_print(f"    - {feat}")
        log_print("")
    else:
        log_print("✓ 所有特征均与目标变量显著相关（p < 0.05）")
        log_print("  建议保留所有特征用于后续建模\n")
    
    return insignificant_features

# ============================================================
# 场景B：特征间冗余性检验
# ============================================================
def test_feature_redundancy(df, features):
    """检验特征对之间的冗余性"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.5】场景B：特征间冗余性检验")
    log_print("=" * 80)
    
    log_print("\n【检验目的】")
    log_print("-" * 60)
    log_print("识别高度关联的特征对（Cramér's V ≥ 0.7）")
    log_print("这些特征对可能携带重复信息，需要考虑删除其中一个\n")
    
    # 重点检验的特征对
    test_pairs = [
        ('education', 'marital-status', '教育程度可能与婚姻状况相关（高学历晚婚）'),
        ('relationship', 'marital-status', '家庭关系与婚姻状况高度相关（如Husband vs Married）'),
        ('occupation', 'education', '职业与教育程度可能相关（高学历→专业职业）'),
        ('workclass', 'occupation', '工作类型与职业可能相关')
    ]
    
    log_print(f"【选定检验的特征对】{len(test_pairs)} 对")
    log_print("-" * 60)
    for idx, (feat1, feat2, reason) in enumerate(test_pairs, 1):
        log_print(f"{idx}. {feat1} ↔ {feat2}")
        log_print(f"   理由：{reason}")
    log_print("")
    
    results = []
    
    for feat1, feat2, reason in test_pairs:
        log_print(f"\n【检验 {feat1} ↔ {feat2}】")
        log_print("-" * 60)
        
        # 构造列联表
        contingency_table = pd.crosstab(df[feat1], df[feat2])
        
        log_print(f"列联表大小：{contingency_table.shape[0]} × {contingency_table.shape[1]}")
        log_print(f"（{feat1} 的 {contingency_table.shape[0]} 个类别 × {feat2} 的 {contingency_table.shape[1]} 个类别）")
        log_print("")
        
        # 显示列联表（前5×5）
        log_print("列联表预览（前5行×前5列）：")
        log_print(contingency_table.iloc[:5, :5].to_string())
        log_print("")
        
        # 卡方检验
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # 计算 Cramér's V
        n = contingency_table.sum().sum()
        r, c = contingency_table.shape
        cramers_v = calculate_cramers_v(chi2, n, r, c)
        
        # 判断
        if p_value < 0.001:
            significance = "***（高度显著）"
        elif p_value < 0.01:
            significance = "**（显著）"
        elif p_value < 0.05:
            significance = "*（显著）"
        else:
            significance = "不显著"
        
        if cramers_v >= 0.7:
            strength = "⚠️  强关联（高度冗余）"
            redundancy = "高度冗余"
        elif cramers_v >= 0.3:
            strength = "中等关联"
            redundancy = "中等相关"
        else:
            strength = "弱关联"
            redundancy = "弱相关"
        
        log_print("检验结果：")
        log_print(f"  卡方值（χ²）：{chi2:.2f}")
        log_print(f"  p 值：{p_value:.6f}")
        log_print(f"  自由度：{dof}")
        log_print(f"  Cramér's V：{cramers_v:.4f}")
        log_print(f"  显著性：{significance}")
        log_print(f"  关联强度：{strength}")
        log_print("")
        
        results.append({
            '特征对': f"{feat1} ↔ {feat2}",
            '特征1': feat1,
            '特征2': feat2,
            '卡方值(χ²)': chi2,
            'p值': p_value,
            '自由度': dof,
            "Cramér's V": cramers_v,
            '冗余程度': redundancy,
            '显著性': significance
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Cramér's V", ascending=False).reset_index(drop=True)
    
    log_print("\n【表格2：特征间冗余性检验结果汇总】")
    log_print("=" * 80)
    log_print("（按 Cramér's V 降序排列，V 值越大表示冗余度越高）\n")
    
    # 格式化输出
    display_df = results_df[['特征对', '卡方值(χ²)', 'p值', "Cramér's V", '冗余程度', '显著性']]
    log_print(display_df.to_string(index=False))
    log_print("")
    
    return results_df

def identify_redundant_pairs(results_df, threshold=0.7):
    """识别高度冗余的特征对"""
    log_print("\n" + "=" * 80)
    log_print(f"【阶段 2.2.6】识别高度冗余特征对（V ≥ {threshold}）")
    log_print("=" * 80)
    
    redundant_pairs = results_df[results_df["Cramér's V"] >= threshold]
    
    log_print(f"\n筛选标准：Cramér's V ≥ {threshold}")
    log_print(f"筛选结果：发现 {len(redundant_pairs)} 对高度冗余特征\n")
    
    if len(redundant_pairs) > 0:
        log_print("【高度冗余特征对列表】")
        log_print("-" * 60)
        
        for idx, row in redundant_pairs.iterrows():
            log_print(f"\n特征对 {idx+1}：{row['特征1']} ↔ {row['特征2']}")
            v_value = row["Cramér's V"]
            log_print(f"  Cramér's V = {v_value:.4f} （V ≥ {threshold}，高度关联）")
            log_print(f"  ⚠️  警告：这两个特征携带高度重复的信息")
            log_print(f"  建议：删除其中一个特征，保留业务意义更强或更易建模的特征")
        log_print("")
    else:
        log_print("✓ 未发现高度冗余特征对（V < 0.7）")
        log_print("  说明：检验的特征对之间关联性不强，信息冗余度可接受\n")
    
    return redundant_pairs

# ============================================================
# 特殊案例：education vs education-num
# ============================================================
def analyze_education_redundancy(df):
    """分析 education 与 education-num 的冗余性"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.7】特殊案例：education vs education-num")
    log_print("=" * 80)
    
    log_print("\n【特殊说明】")
    log_print("-" * 60)
    log_print("虽然 education-num 是数值型变量，但它是 education 的编码表示")
    log_print("这两者是【语义冗余】：同一信息的不同表达形式\n")
    
    log_print("【编码对应关系示例】")
    log_print("-" * 60)
    
    # 显示对应关系
    if 'education' in df.columns and 'education-num' in df.columns:
        mapping = df[['education', 'education-num']].drop_duplicates().sort_values('education-num')
        log_print(mapping.to_string(index=False))
        log_print("")
    
    log_print("【冗余性分析】")
    log_print("-" * 60)
    log_print("冗余程度：100%（完全冗余）")
    log_print("  - education：文本型分类变量（如 'Bachelors', 'HS-grad'）")
    log_print("  - education-num：数值型编码（如 13, 9）")
    log_print("  - 二者一一对应，携带完全相同的信息")
    log_print("")
    log_print("【决策建议】")
    log_print("-" * 60)
    log_print("❌ 删除：education（文本型）")
    log_print("✅ 保留：education-num（数值型）")
    log_print("")
    log_print("理由：")
    log_print("  1. 数值型变量更便于机器学习建模")
    log_print("  2. 数值型可直接用于线性模型、树模型等")
    log_print("  3. 文本型需要额外的编码处理（如One-Hot），增加特征维度")
    log_print("  4. education-num 保留了教育程度的序关系（9 < 13 < 16）")
    log_print("")

# ============================================================
# 综合决策
# ============================================================
def make_final_decision(df, results_feature_target, redundant_pairs, insignificant_features):
    """生成最终的特征保留/删除决策"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.8】特征选择综合决策")
    log_print("=" * 80)
    
    log_print("\n【决策依据】")
    log_print("-" * 60)
    log_print("1. 与目标变量的相关性（p < 0.05 为显著）")
    log_print("2. 特征间的冗余性（Cramér's V ≥ 0.7 为高度冗余）")
    log_print("3. 业务含义和建模便利性")
    log_print("")
    
    # 构建决策表
    decisions = []
    
    # 所有分类特征
    all_categorical = df.select_dtypes(include=['object']).columns.tolist()
    all_categorical.remove('income')  # 移除目标变量
    
    # 判断每个特征
    for feat in all_categorical:
        # 从场景A获取信息
        feat_row = results_feature_target[results_feature_target['特征'] == feat]
        if len(feat_row) > 0:
            p_val = feat_row['p值'].values[0]
            v_val = feat_row["Cramér's V"].values[0]
            sig = feat_row['显著性'].values[0]
        else:
            p_val, v_val, sig = None, None, None
        
        # 判断是否冗余
        is_redundant = False
        redundant_with = ""
        
        # 检查是否在冗余对中
        for _, row in redundant_pairs.iterrows():
            if feat in [row['特征1'], row['特征2']]:
                is_redundant = True
                redundant_with = row['特征2'] if feat == row['特征1'] else row['特征1']
                break
        
        # 特殊处理 education（与 education-num 冗余）
        if feat == 'education':
            is_redundant = True
            redundant_with = "education-num（数值型编码）"
        
        # 决策
        if feat in insignificant_features:
            decision = "❌ 删除"
            reason = "与目标变量无显著关联"
        elif feat == 'education':
            decision = "❌ 删除"
            reason = f"与 education-num 100%冗余，保留数值型"
        elif is_redundant:
            decision = "⚠️  待定"
            reason = f"与 {redundant_with} 高度冗余，需进一步决策"
        else:
            decision = "✅ 保留"
            reason = "有预测价值且无冗余"
        
        decisions.append({
            '特征': feat,
            '类别数': df[feat].nunique(),
            '显著性': sig if sig else 'N/A',
            "Cramér's V": f"{v_val:.4f}" if v_val is not None else 'N/A',
            '冗余情况': redundant_with if is_redundant else '无',
            '决策': decision,
            '理由': reason
        })
    
    decision_df = pd.DataFrame(decisions)
    
    log_print("【表格3：特征保留/删除决策表】")
    log_print("=" * 80)
    log_print("")
    log_print(decision_df.to_string(index=False))
    log_print("")
    
    # 统计
    to_keep = decision_df[decision_df['决策'] == '✅ 保留']
    to_remove = decision_df[decision_df['决策'] == '❌ 删除']
    pending = decision_df[decision_df['决策'] == '⚠️  待定']
    
    log_print("\n【决策统计】")
    log_print("-" * 60)
    log_print(f"保留特征：{len(to_keep)} 个")
    for feat in to_keep['特征'].tolist():
        log_print(f"  ✅ {feat}")
    
    log_print(f"\n删除特征：{len(to_remove)} 个")
    for feat in to_remove['特征'].tolist():
        log_print(f"  ❌ {feat}")
    
    if len(pending) > 0:
        log_print(f"\n待定特征：{len(pending)} 个（需进一步分析）")
        for feat in pending['特征'].tolist():
            log_print(f"  ⚠️  {feat}")
    
    log_print("")
    
    return decision_df, to_remove['特征'].tolist()

# ============================================================
# 执行特征删除
# ============================================================
def remove_features(df, features_to_remove):
    """执行特征删除操作"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.9】执行特征删除操作")
    log_print("=" * 80)
    
    if len(features_to_remove) == 0:
        log_print("\n✓ 无需删除任何特征")
        log_print("  所有特征均有价值且无严重冗余\n")
        return df
    
    log_print(f"\n【准备删除 {len(features_to_remove)} 个特征】")
    log_print("-" * 60)
    
    for feat in features_to_remove:
        log_print(f"  ❌ {feat}")
    
    # 执行删除
    df_reduced = df.drop(columns=features_to_remove)
    
    log_print(f"\n【删除操作完成】")
    log_print("-" * 60)
    log_print(f"删除前：{df.shape[0]:,} 行 × {df.shape[1]} 列")
    log_print(f"删除后：{df_reduced.shape[0]:,} 行 × {df_reduced.shape[1]} 列")
    log_print(f"减少列数：{df.shape[1] - df_reduced.shape[1]} 列\n")
    
    # 保存
    output_file = 'integrated_data.csv'
    df_reduced.to_csv(output_file, index=False, encoding='utf-8-sig')
    log_print(f"✓ 集成后的数据已保存：{output_file}\n")
    
    return df_reduced

# ============================================================
# 可视化
# ============================================================
def plot_feature_importance(results_df, save_path='图3_特征重要性排序.png'):
    """绘制特征重要性柱状图"""
    log_print("\n" + "=" * 80)
    log_print("【阶段 2.2.10】绘制特征重要性图")
    log_print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按卡方值排序
    data = results_df.sort_values('卡方值(χ²)', ascending=True)
    
    # 绘制水平柱状图
    bars = ax.barh(data['特征'], data['卡方值(χ²)'], color='steelblue', edgecolor='black')
    
    # 添加数值标签
    for i, (idx, row) in enumerate(data.iterrows()):
        chi2 = row['卡方值(χ²)']
        ax.text(chi2, i, f' {chi2:.0f}', va='center', fontsize=9)
    
    ax.set_xlabel('卡方值（χ²）', fontsize=12, fontweight='bold')
    ax.set_ylabel('特征', fontsize=12, fontweight='bold')
    ax.set_title('特征与目标变量（income）的关联强度排序\n（卡方值越大，关联性越强）', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log_print(f"\n✓ 特征重要性图已保存：{save_path}\n")

# ============================================================
# 总结报告
# ============================================================
def summary_report(df_original, df_reduced, results_feature_target, redundant_pairs, features_removed):
    """生成总结报告"""
    log_print("\n" + "=" * 80)
    log_print("【数据集成（分类变量）总结报告】")
    log_print("=" * 80)
    
    log_print("\n【分析概况】")
    log_print("-" * 60)
    log_print(f"分析的分类特征数：{len(results_feature_target)} 个")
    log_print(f"检验的特征对数：{len(redundant_pairs)} 对")
    log_print(f"分析方法：卡方独立性检验 + Cramér's V 系数")
    log_print("")
    
    log_print("【关键发现】")
    log_print("-" * 60)
    
    # 最重要的特征
    top1 = results_feature_target.iloc[0]
    v_val = top1["Cramér's V"]
    log_print(f"最重要特征：{top1['特征']} (χ²={top1['卡方值(χ²)']:.2f}, V={v_val:.4f})")
    
    # 显著特征统计
    sig_count = len(results_feature_target[results_feature_target['p值'] < 0.05])
    log_print(f"显著特征数：{sig_count}/{len(results_feature_target)} 个")
    
    # 冗余特征对
    high_redundant = len(redundant_pairs)
    log_print(f"高度冗余特征对：{high_redundant} 对")
    
    log_print("")
    
    log_print("【数据变化】")
    log_print("-" * 60)
    log_print(f"集成前：{df_original.shape[0]:,} 行 × {df_original.shape[1]} 列")
    log_print(f"集成后：{df_reduced.shape[0]:,} 行 × {df_reduced.shape[1]} 列")
    log_print(f"删除特征：{len(features_removed)} 个")
    if len(features_removed) > 0:
        log_print(f"  具体为：{', '.join(features_removed)}")
    log_print("")
    
    log_print("【主要结论】")
    log_print("-" * 60)
    log_print("✓ 完成了分类特征的关联性和冗余性分析")
    log_print("✓ 识别并删除了冗余和无关特征")
    log_print("✓ 保留的特征均与目标变量显著相关")
    log_print("✓ 数据集成后的特征更精简、信息更集中")
    log_print("")
    
    log_print("【下一步工作】")
    log_print("-" * 60)
    log_print("1. 进行数据规约（标准化、PCA降维）")
    log_print("2. 特征构造（交互特征、哑编码）")
    log_print("3. 准备用于建模的最终数据集\n")
    
    log_print("✅ 分类变量卡方检验分析完成！\n")

# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "=" * 80)
    print("【数据预处理实验 - 步骤2.2：数据集成 - 分类变量卡方检验】")
    print("=" * 80 + "\n")
    
    init_log()
    
    # 1. 加载数据
    df = load_cleaned_data()
    if df is None:
        return
    
    df_original = df.copy()
    
    # 2. 理论说明
    chi_square_theory()
    
    # 3. 识别分类变量
    features, target = identify_categorical_features(df)
    
    # 4. 场景A：特征与目标变量检验
    results_feature_target = test_features_vs_target(df, features, target)
    insignificant_features = analyze_feature_importance(results_feature_target)
    
    # 5. 场景B：特征间冗余性检验
    results_redundancy = test_feature_redundancy(df, features)
    redundant_pairs = identify_redundant_pairs(results_redundancy, threshold=0.7)
    
    # 6. 特殊案例：education vs education-num
    analyze_education_redundancy(df)
    
    # 7. 综合决策
    decision_df, features_to_remove = make_final_decision(df, results_feature_target, 
                                                          redundant_pairs, insignificant_features)
    
    # 8. 执行删除
    df_reduced = remove_features(df, features_to_remove)
    
    # 9. 可视化
    plot_feature_importance(results_feature_target)
    
    # 10. 总结报告
    summary_report(df_original, df_reduced, results_feature_target, 
                  redundant_pairs, features_to_remove)
    
    log_print(f"执行结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80 + "\n")
    
    print("\n" + "=" * 80)
    print("✅ 分类变量卡方检验分析完成！")
    print("=" * 80)
    print("\n生成的文件：")
    print("  1. step2_integration_part2_log.txt - 详细分析日志")
    print("  2. 图3_特征重要性排序.png - 特征重要性柱状图")
    print("  3. integrated_data.csv - 集成后的数据")
    print("\n下一步：进行数据规约（标准化、PCA降维）\n")

if __name__ == "__main__":
    main()


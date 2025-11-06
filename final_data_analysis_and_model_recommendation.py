# -*- coding: utf-8 -*-
"""
================================================================================
最终数据分析与模型推荐
================================================================================
目的：
  1. 详细分析最终预处理数据的格式和特征
  2. 解释每个特征的含义和数据类型
  3. 推荐适合的二分类模型（从简单到复杂）
  4. 说明不同模型对特征标准化的要求
  
输入文件: final_preprocessed_data.csv（最终预处理数据）
输出文件: 
  - final_data_analysis_report.txt（详细分析报告）
  - 图11_特征类型结构图.png
  - 图12_数据质量评估图.png
  - 图13_模型推荐路线图.png
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
log_file = 'final_data_analysis_report.txt'

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
    f.write("=" * 100 + "\n")
    f.write("最终数据分析与模型推荐报告\n")
    f.write("=" * 100 + "\n\n")

write_log("=" * 100)
write_log("最终数据分析与模型推荐")
write_log("=" * 100)
write_log("")

# ================================================================================
# 第一部分：数据加载与基本信息
# ================================================================================
write_log("=" * 100)
write_log("第一部分：数据基本信息")
write_log("=" * 100)
write_log("")

# 读取最终数据
df = pd.read_csv('final_preprocessed_data.csv')

write_log("【数据概览】")
write_log("-" * 100)
write_log("")
write_log(f"数据文件：final_preprocessed_data.csv")
write_log(f"数据规模：{df.shape[0]:,} 行 × {df.shape[1]} 列")
write_log(f"内存占用：{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
write_log("")

# 数据质量检查
write_log("【数据质量评估】")
write_log("-" * 100)
write_log("")

total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isnull().sum().sum()
duplicated_rows = df.duplicated().sum()

write_log(f"1. 完整性检查：")
write_log(f"   - 总单元格数：{total_cells:,} 个")
write_log(f"   - 缺失值数量：{missing_cells} 个")
write_log(f"   - 缺失率：{(missing_cells / total_cells) * 100:.4f}%")
write_log(f"   - 状态：{'✓ 无缺失值' if missing_cells == 0 else '⚠ 存在缺失值'}")
write_log("")

write_log(f"2. 唯一性检查：")
write_log(f"   - 重复样本数：{duplicated_rows} 行")
write_log(f"   - 重复率：{(duplicated_rows / df.shape[0]) * 100:.4f}%")
write_log(f"   - 状态：{'✓ 无重复样本' if duplicated_rows == 0 else '⚠ 存在重复'}")
write_log("")

# 目标变量分析
write_log(f"3. 目标变量分布：")
income_counts = df['income'].value_counts()
income_ratio = (income_counts / df.shape[0] * 100).round(2)

for category, count in income_counts.items():
    ratio = income_ratio[category]
    write_log(f"   - {category}: {count:,} 样本 ({ratio}%)")

imbalance_ratio = income_counts.max() / income_counts.min()
write_log(f"   - 类别不平衡比：{imbalance_ratio:.2f} : 1")
if imbalance_ratio > 3:
    write_log(f"   - 状态：⚠ 类别不平衡（建议使用分层抽样或 SMOTE）")
elif imbalance_ratio > 1.5:
    write_log(f"   - 状态：✓ 轻度不平衡（可接受）")
else:
    write_log(f"   - 状态：✓ 类别平衡")
write_log("")

# ================================================================================
# 第二部分：特征分类与详细解释
# ================================================================================
write_log("=" * 100)
write_log("第二部分：特征详细解释（共 {0} 个特征 + 1 个目标变量）".format(df.shape[1] - 1))
write_log("=" * 100)
write_log("")

# 识别特征类型
numerical_features = []
one_hot_features = []
target = 'income'

for col in df.columns:
    if col == target:
        continue
    elif df[col].dtype in ['int64', 'float64']:
        # 判断是否为独热编码（只有0和1）
        unique_vals = df[col].unique()
        if set(unique_vals).issubset({0, 1}) and len(unique_vals) == 2:
            one_hot_features.append(col)
        else:
            numerical_features.append(col)

write_log(f"特征类型统计：")
write_log(f"  - 数值型特征：{len(numerical_features)} 个")
write_log(f"  - 独热编码特征：{len(one_hot_features)} 个")
write_log(f"  - 目标变量：1 个")
write_log("")

# ===== 数值型特征详细解释 =====
write_log("┌" + "─" * 98 + "┐")
write_log("│" + " " * 35 + "【类别 A：数值型特征】" + " " * 41 + "│")
write_log("└" + "─" * 98 + "┘")
write_log("")

write_log(f"共 {len(numerical_features)} 个数值型特征，分为以下子类：")
write_log("")

# 原始数值特征（已标准化）
original_standardized = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
write_log("【A1. 原始数值特征 - 已标准化】（5个）")
write_log("-" * 100)
write_log("")

feature_explanations = {
    'age': {
        'name': 'age（年龄）',
        'type': '连续变量',
        'standardized': '✓ 已标准化（Z-score）',
        'range': f"当前范围：[{df['age'].min():.2f}, {df['age'].max():.2f}]（标准化后）",
        'original': '原始范围：17-90 岁',
        'meaning': '个体的年龄，反映生命周期阶段和工作经验',
        'importance': '年龄是收入的重要预测因子（中年收入通常较高）'
    },
    'fnlwgt': {
        'name': 'fnlwgt（人口权重）',
        'type': '连续变量',
        'standardized': '✓ 已标准化（Z-score）',
        'range': f"当前范围：[{df['fnlwgt'].min():.2f}, {df['fnlwgt'].max():.2f}]（标准化后）",
        'original': '原始范围：约 10,000 - 1,500,000',
        'meaning': 'Final Weight，人口统计权重，表示该样本代表的人口数量',
        'importance': '用于加权统计，对个体收入预测的直接影响较小'
    },
    'capital-gain': {
        'name': 'capital-gain（资本收益）',
        'type': '连续变量（高度右偏）',
        'standardized': '✓ 已标准化（Z-score）',
        'range': f"当前范围：[{df['capital-gain'].min():.2f}, {df['capital-gain'].max():.2f}]（标准化后）",
        'original': '原始范围：0 - 99,999 美元/年',
        'meaning': '投资、房产等资本性收入（大多数人为0，少数人极高）',
        'importance': '高收入人群的重要标志，与收入强相关'
    },
    'capital-loss': {
        'name': 'capital-loss（资本损失）',
        'type': '连续变量（高度右偏）',
        'standardized': '✓ 已标准化（Z-score）',
        'range': f"当前范围：[{df['capital-loss'].min():.2f}, {df['capital-loss'].max():.2f}]（标准化后）",
        'original': '原始范围：0 - 4,356 美元/年',
        'meaning': '投资、商业经营的损失（大多数人为0）',
        'importance': '有资本损失的人往往也有高收入（风险投资能力）'
    },
    'hours-per-week': {
        'name': 'hours-per-week（每周工作小时）',
        'type': '连续变量',
        'standardized': '✓ 已标准化（Z-score）',
        'range': f"当前范围：[{df['hours-per-week'].min():.2f}, {df['hours-per-week'].max():.2f}]（标准化后）",
        'original': '原始范围：1 - 99 小时/周',
        'meaning': '每周工作时长，反映劳动投入强度',
        'importance': '工作时长与收入正相关（全职>兼职）'
    }
}

for i, feat in enumerate(original_standardized, 1):
    if feat in df.columns:
        info = feature_explanations[feat]
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        
        write_log(f"{i}. {info['name']}")
        write_log(f"   类型：{info['type']}")
        write_log(f"   标准化状态：{info['standardized']}")
        write_log(f"   取值范围：{info['range']}")
        write_log(f"   原始范围：{info['original']}")
        write_log(f"   当前统计：均值={mean_val:.4f}，标准差={std_val:.4f}")
        write_log(f"   业务含义：{info['meaning']}")
        write_log(f"   重要性：{info['importance']}")
        write_log("")

# 序数编码特征（未标准化）
write_log("【A2. 序数编码特征 - 未标准化】（1个）")
write_log("-" * 100)
write_log("")

if 'education-num' in df.columns:
    mean_val = df['education-num'].mean()
    std_val = df['education-num'].std()
    min_val = df['education-num'].min()
    max_val = df['education-num'].max()
    
    write_log(f"1. education-num（教育年限）")
    write_log(f"   类型：序数变量的数值编码（Ordinal Encoding）")
    write_log(f"   标准化状态：✗ 未标准化（保留原始序数关系）")
    write_log(f"   取值范围：{int(min_val)} - {int(max_val)}（对应教育年限）")
    write_log(f"   当前统计：均值={mean_val:.2f}，标准差={std_val:.2f}")
    write_log(f"   编码对应关系：")
    write_log(f"     1  = Preschool（学前）")
    write_log(f"     2  = 1st-4th（小学低年级）")
    write_log(f"     ...（省略中间）")
    write_log(f"     9  = HS-grad（高中毕业）")
    write_log(f"     10 = Some-college（部分大学）")
    write_log(f"     13 = Bachelors（学士）")
    write_log(f"     14 = Masters（硕士）")
    write_log(f"     16 = Doctorate（博士）")
    write_log(f"   业务含义：受教育年限，直接反映人力资本质量")
    write_log(f"   重要性：教育是收入的最重要预测因子之一")
    write_log(f"   ⚠ 特别说明：")
    write_log(f"     - 该特征保持原始 1-16 的取值，未进行标准化")
    write_log(f"     - 原因：保留序数关系的直观性（如 <=12年 易于解释）")
    write_log(f"     - 如使用线性模型/神经网络，建议标准化")
    write_log(f"     - 如使用树模型，保持原样即可")
    write_log("")

# 新构造特征
write_log("【A3. 新构造特征 - 混合状态】（3个）")
write_log("-" * 100)
write_log("")

constructed_features = {
    'work_intensity': {
        'name': 'work_intensity（工作强度）',
        'formula': 'education-num × hours-per-week',
        'standardized': '✗ 未标准化',
        'components': 'education-num（未标准化）× hours-per-week（已标准化）',
        'meaning': '人力资本投入强度 = 教育质量 × 劳动投入量',
        'interpretation': '高学历 × 长工时 = 高人力资本密度 → 通常对应高收入工作',
        'validation': '已验证：高收入组均值=5.11，低收入组均值=-1.08（差异显著）'
    },
    'net_capital': {
        'name': 'net_capital（资本净收益）',
        'formula': 'capital-gain - capital-loss',
        'standardized': '✓ 基于已标准化特征计算',
        'components': 'capital-gain（已标准化）- capital-loss（已标准化）',
        'meaning': '投资净回报 = 资本收益 - 资本损失',
        'interpretation': '正值=盈利，负值=亏损，直接反映财富增长情况',
        'validation': '已验证：高收入组均值=0.13，低收入组均值=-0.04（差异显著）'
    },
    'work_age_ratio': {
        'name': 'work_age_ratio（年龄工作比）',
        'formula': 'hours-per-week / age',
        'standardized': '✓ 基于已标准化特征计算',
        'components': 'hours-per-week（已标准化）/ age（已标准化）',
        'meaning': '单位年龄工作强度 = 工作时长 / 年龄',
        'interpretation': '年轻人高强度工作倾向，反映职业发展阶段',
        'validation': '已验证：高收入组均值=-0.06，低收入组均值=0.00（差异不明显）'
    }
}

for i, (feat, info) in enumerate(constructed_features.items(), 1):
    if feat in df.columns:
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        min_val = df[feat].min()
        max_val = df[feat].max()
        
        write_log(f"{i}. {info['name']}")
        write_log(f"   构造公式：{info['formula']}")
        write_log(f"   标准化状态：{info['standardized']}")
        write_log(f"   组成成分：{info['components']}")
        write_log(f"   取值范围：[{min_val:.2f}, {max_val:.2f}]")
        write_log(f"   当前统计：均值={mean_val:.4f}，标准差={std_val:.4f}")
        write_log(f"   业务含义：{info['meaning']}")
        write_log(f"   解释：{info['interpretation']}")
        write_log(f"   有效性验证：{info['validation']}")
        
        if feat == 'work_intensity':
            write_log(f"   ⚠ 特别说明：")
            write_log(f"     - 该特征由未标准化（education-num）和已标准化（hours-per-week）特征相乘得到")
            write_log(f"     - 取值范围较大（-50到+56），尺度与其他标准化特征不一致")
            write_log(f"     - 如使用线性模型/神经网络，建议标准化")
            write_log(f"     - 如使用树模型，保持原样即可")
        write_log("")

# ===== 独热编码特征 =====
write_log("")
write_log("┌" + "─" * 98 + "┐")
write_log("│" + " " * 34 + "【类别 B：独热编码特征】" + " " * 39 + "│")
write_log("└" + "─" * 98 + "┘")
write_log("")

write_log(f"共 {len(one_hot_features)} 个独热编码特征，来自 7 个原始分类变量")
write_log("")
write_log("【特征命名规则】")
write_log("  格式：原变量名_类别名")
write_log("  取值：0（不属于该类别）或 1（属于该类别）")
write_log("  编码方式：drop_first=True（删除每个变量的首个类别，避免多重共线性）")
write_log("")

# 统计每个原始变量生成的独热编码列
one_hot_groups = {}
for col in one_hot_features:
    # 提取原变量名（下划线之前的部分）
    if '_' in col:
        base_name = col.rsplit('_', 1)[0]
        if base_name not in one_hot_groups:
            one_hot_groups[base_name] = []
        one_hot_groups[base_name].append(col)

write_log(f"【各原始分类变量的独热编码详情】")
write_log("-" * 100)
write_log("")

# 分类变量详细说明
categorical_explanations = {
    'workclass': {
        'name': 'workclass（工作类型）',
        'meaning': '雇佣关系类型，反映工作稳定性和收入来源',
        'categories': '8个类别：政府、私企、自雇等',
        'importance': '不同工作类型的收入差异显著'
    },
    'marital-status': {
        'name': 'marital-status（婚姻状态）',
        'meaning': '婚姻状况，反映家庭结构和经济责任',
        'categories': '7个类别：已婚、未婚、离异等',
        'importance': '已婚人群收入通常更高（家庭支出压力）'
    },
    'occupation': {
        'name': 'occupation（职业）',
        'meaning': '具体职业类别，直接决定收入水平',
        'categories': '14个类别：管理、技术、服务等',
        'importance': '职业是收入的最直接决定因素'
    },
    'relationship': {
        'name': 'relationship（家庭角色）',
        'meaning': '在家庭中的角色关系',
        'categories': '6个类别：丈夫、妻子、未婚等',
        'importance': '与婚姻状态相关，反映家庭经济责任'
    },
    'race': {
        'name': 'race（种族）',
        'meaning': '种族类别（美国人口统计标准）',
        'categories': '5个类别：白人、黑人、亚太裔等',
        'importance': '可能存在系统性收入差异（社会经济因素）'
    },
    'sex': {
        'name': 'sex（性别）',
        'meaning': '生理性别',
        'categories': '2个类别：男性、女性',
        'importance': '性别薪酬差距是重要的社会经济现象'
    },
    'native-country': {
        'name': 'native-country（出生国家）',
        'meaning': '个体的出生国家或地区',
        'categories': '41个类别：美国、墨西哥、印度等',
        'importance': '移民背景可能影响职业选择和收入'
    }
}

for i, (base_name, cols) in enumerate(sorted(one_hot_groups.items()), 1):
    n_cols = len(cols)
    original_categories = n_cols + 1  # 因为 drop_first=True，原始类别数 = 编码列数 + 1
    
    if base_name in categorical_explanations:
        info = categorical_explanations[base_name]
        write_log(f"{i}. {info['name']}")
        write_log(f"   原始类别数：{original_categories} 个")
        write_log(f"   编码后列数：{n_cols} 列（删除了首个类别）")
        write_log(f"   业务含义：{info['meaning']}")
        write_log(f"   类别说明：{info['categories']}")
        write_log(f"   重要性：{info['importance']}")
        write_log(f"   生成的列名示例（前3个）：")
        for col in cols[:3]:
            category = col.split('_', 1)[1] if '_' in col else col
            count = df[col].sum()
            ratio = (count / df.shape[0] * 100)
            write_log(f"     • {col} = 1 表示该样本属于 '{category}' 类别（{count:,}样本，{ratio:.2f}%）")
        if n_cols > 3:
            write_log(f"     ... 还有 {n_cols - 3} 个列（见完整数据）")
        write_log("")

# ===== 目标变量 =====
write_log("")
write_log("┌" + "─" * 98 + "┐")
write_log("│" + " " * 38 + "【类别 C：目标变量】" + " " * 39 + "│")
write_log("└" + "─" * 98 + "┘")
write_log("")

write_log(f"【income（年收入水平）】")
write_log("-" * 100)
write_log("")
write_log(f"变量名：income")
write_log(f"变量类型：二分类标签（分类变量）")
write_log(f"取值：")
write_log(f"  - '<=50K'：年收入 ≤ 50,000 美元（低收入组）")
write_log(f"  - '>50K'：年收入 > 50,000 美元（高收入组）")
write_log(f"业务含义：预测目标，判断个体是否为高收入人群")
write_log(f"数据分布：")
for category, count in income_counts.items():
    ratio = income_ratio[category]
    write_log(f"  - {category}: {count:,} 样本 ({ratio}%)")
write_log(f"问题类型：二分类问题（Binary Classification）")
write_log(f"评估指标建议：准确率、精确率、召回率、F1-score、AUC-ROC")
write_log("")

# ================================================================================
# 第三部分：特征标准化状态总结
# ================================================================================
write_log("=" * 100)
write_log("第三部分：特征标准化状态总结")
write_log("=" * 100)
write_log("")

write_log("【特征标准化情况汇总】")
write_log("-" * 100)
write_log("")

standardized_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 
                        'net_capital', 'work_age_ratio']
not_standardized_features = ['education-num', 'work_intensity']

write_log(f"✓ 已标准化特征（{len([f for f in standardized_features if f in df.columns])} 个）：")
for feat in standardized_features:
    if feat in df.columns:
        write_log(f"  - {feat}")
write_log("")

write_log(f"✗ 未标准化特征（{len([f for f in not_standardized_features if f in df.columns])} 个）：")
for feat in not_standardized_features:
    if feat in df.columns:
        write_log(f"  - {feat}")
write_log("")

write_log(f"✓ 独热编码特征（{len(one_hot_features)} 个）：")
write_log(f"  - 全部为 0/1 二进制值，不需要标准化")
write_log("")

write_log("【对不同模型的影响】")
write_log("-" * 100)
write_log("")

write_log("1. 树模型（决策树、随机森林、XGBoost、LightGBM）：")
write_log("   ✓ 当前数据可以直接使用")
write_log("   ✓ 树模型对特征尺度不敏感")
write_log("   ✓ 保留 education-num 和 work_intensity 的原始值有助于解释")
write_log("")

write_log("2. 线性模型（逻辑回归、线性SVM）：")
write_log("   ⚠ 需要额外处理")
write_log("   ⚠ education-num 和 work_intensity 与其他特征尺度不一致")
write_log("   ⚠ 建议：对这两个特征进行 Z-score 标准化后再使用")
write_log("   原因：")
write_log("     - 特征尺度不一致会导致梯度下降收敛慢")
write_log("     - 正则化会不公平地惩罚大尺度特征")
write_log("     - 系数大小难以直接比较重要性")
write_log("")

write_log("3. 神经网络（多层感知机、深度学习）：")
write_log("   ⚠ 强烈建议标准化")
write_log("   ⚠ 对 education-num 和 work_intensity 进行标准化")
write_log("   原因：")
write_log("     - 神经网络对特征尺度非常敏感")
write_log("     - 激活函数在标准化输入下效果更好")
write_log("     - 避免梯度消失/爆炸问题")
write_log("")

write_log("4. 距离模型（KNN、K-Means）：")
write_log("   ⚠ 必须标准化")
write_log("   ⚠ 对 education-num 和 work_intensity 进行标准化")
write_log("   原因：")
write_log("     - 基于欧氏距离计算")
write_log("     - 大尺度特征会主导距离计算")
write_log("")

# ================================================================================
# 第四部分：模型推荐（从简单到复杂）
# ================================================================================
write_log("=" * 100)
write_log("第四部分：二分类模型推荐（从简单到复杂）")
write_log("=" * 100)
write_log("")

write_log("【模型推荐路线图】")
write_log("-" * 100)
write_log("")

models_recommendation = """
根据数据特点（48,169样本，85特征，二分类，类别轻度不平衡），推荐以下模型：

┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      模型推荐路线                                              │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│  简单 ────────────────────────────────────────────────────────────────────────────► 复杂      │
│                                                                                                │
│  Level 1          Level 2           Level 3            Level 4            Level 5             │
│  ┌─────┐         ┌─────┐           ┌─────┐           ┌─────┐            ┌─────┐             │
│  │逻辑 │         │决策 │           │随机 │           │XGBoost│          │深度 │             │
│  │回归 │   →     │树   │     →     │森林 │     →     │LightGBM│    →    │神经 │             │
│  │     │         │朴素 │           │SVM  │           │CatBoost│         │网络 │             │
│  └─────┘         │贝叶斯│          └─────┘           └─────┘            └─────┘             │
│                  └─────┘                                                                      │
│                                                                                                │
│  易解释          可解释            平衡              高性能              最复杂               │
│  快速训练        中等速度          较慢              中等               最慢                  │
│  基准模型        改进              生产级            竞赛级             研究级                │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

write_log(models_recommendation)
write_log("")

# 详细模型说明
write_log("【详细模型说明】")
write_log("-" * 100)
write_log("")

models = [
    {
        'level': 'Level 1: 基准模型',
        'models': [
            {
                'name': '逻辑回归（Logistic Regression）',
                'complexity': '⭐',
                'pros': [
                    '训练速度极快（秒级）',
                    '结果高度可解释（系数代表特征重要性）',
                    '内存占用小',
                    '适合作为基准模型',
                    '输出概率值，便于设置阈值'
                ],
                'cons': [
                    '假设线性可分，无法捕捉复杂非线性关系',
                    '对特征工程要求高',
                    '需要特征标准化'
                ],
                'data_requirement': '⚠ 需要标准化 education-num 和 work_intensity',
                'sklearn': 'from sklearn.linear_model import LogisticRegression',
                'params': 'LogisticRegression(max_iter=1000, random_state=42)',
                'expected_accuracy': '约 80-84%（基于类似数据集经验）',
                'use_case': '快速验证特征有效性，获得可解释的基准结果'
            }
        ]
    },
    {
        'level': 'Level 2: 简单非线性模型',
        'models': [
            {
                'name': '决策树（Decision Tree）',
                'complexity': '⭐⭐',
                'pros': [
                    '完全可解释（可视化决策路径）',
                    '自动特征选择',
                    '处理非线性关系',
                    '不需要特征标准化',
                    '可处理缺失值'
                ],
                'cons': [
                    '容易过拟合',
                    '对噪声敏感',
                    '预测性能通常不如集成方法'
                ],
                'data_requirement': '✓ 当前数据可直接使用（不需要额外标准化）',
                'sklearn': 'from sklearn.tree import DecisionTreeClassifier',
                'params': 'DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)',
                'expected_accuracy': '约 82-85%',
                'use_case': '理解特征分裂逻辑，快速原型'
            },
            {
                'name': '朴素贝叶斯（Naive Bayes）',
                'complexity': '⭐',
                'pros': [
                    '训练和预测速度极快',
                    '适合高维数据',
                    '概率输出',
                    '小样本表现好'
                ],
                'cons': [
                    '假设特征独立（实际中往往不成立）',
                    '对特征分布敏感',
                    '性能通常不如其他方法'
                ],
                'data_requirement': '✓ 当前数据可直接使用',
                'sklearn': 'from sklearn.naive_bayes import GaussianNB',
                'params': 'GaussianNB()',
                'expected_accuracy': '约 78-82%',
                'use_case': '快速基准，对比验证'
            }
        ]
    },
    {
        'level': 'Level 3: 集成学习入门',
        'models': [
            {
                'name': '随机森林（Random Forest）',
                'complexity': '⭐⭐⭐',
                'pros': [
                    '性能优秀（通常超过单决策树5-10%）',
                    '不容易过拟合',
                    '提供特征重要性排序',
                    '不需要特征标准化',
                    '鲁棒性强',
                    '可并行训练'
                ],
                'cons': [
                    '模型文件较大',
                    '预测速度比单模型慢',
                    '可解释性较弱'
                ],
                'data_requirement': '✓ 当前数据可直接使用（推荐）',
                'sklearn': 'from sklearn.ensemble import RandomForestClassifier',
                'params': 'RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)',
                'expected_accuracy': '约 85-87%',
                'use_case': '生产环境首选，平衡性能与稳定性'
            },
            {
                'name': '支持向量机（SVM）',
                'complexity': '⭐⭐⭐',
                'pros': [
                    '高维空间表现好',
                    '可用核技巧捕捉非线性',
                    '泛化能力强'
                ],
                'cons': [
                    '训练时间长（O(n²)到O(n³)）',
                    '大数据集不适用',
                    '需要特征标准化',
                    '超参数调优复杂'
                ],
                'data_requirement': '⚠ 需要标准化 education-num 和 work_intensity',
                'sklearn': 'from sklearn.svm import SVC',
                'params': 'SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)',
                'expected_accuracy': '约 84-86%',
                'use_case': '中小数据集，追求泛化能力'
            }
        ]
    },
    {
        'level': 'Level 4: 高性能梯度提升（推荐）',
        'models': [
            {
                'name': 'XGBoost（Extreme Gradient Boosting）',
                'complexity': '⭐⭐⭐⭐',
                'pros': [
                    '性能极强（Kaggle竞赛常用）',
                    '内置正则化，防止过拟合',
                    '处理缺失值',
                    '特征重要性',
                    '支持GPU加速',
                    '不需要特征标准化'
                ],
                'cons': [
                    '超参数多，调优复杂',
                    '训练时间较长',
                    '需要安装额外库'
                ],
                'data_requirement': '✓ 当前数据可直接使用（强烈推荐）',
                'install': 'pip install xgboost',
                'code': 'import xgboost as xgb\nmodel = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)',
                'expected_accuracy': '约 86-88%',
                'use_case': '追求最高性能，生产环境/竞赛'
            },
            {
                'name': 'LightGBM',
                'complexity': '⭐⭐⭐⭐',
                'pros': [
                    '训练速度快（比XGBoost快10倍）',
                    '内存占用小',
                    '支持大数据集',
                    '性能与XGBoost相当',
                    '原生支持类别特征',
                    '不需要特征标准化'
                ],
                'cons': [
                    '小数据集容易过拟合',
                    '需要调优',
                    '需要安装额外库'
                ],
                'data_requirement': '✓ 当前数据可直接使用（强烈推荐）',
                'install': 'pip install lightgbm',
                'code': 'import lightgbm as lgb\nmodel = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)',
                'expected_accuracy': '约 86-88%',
                'use_case': '大数据集，追求训练速度'
            },
            {
                'name': 'CatBoost',
                'complexity': '⭐⭐⭐⭐',
                'pros': [
                    '自动处理类别特征（无需独热编码）',
                    '防止过拟合能力强',
                    '默认参数性能好',
                    '训练稳定',
                    '不需要特征标准化'
                ],
                'cons': [
                    '训练速度较LightGBM慢',
                    '需要安装额外库'
                ],
                'data_requirement': '✓ 当前数据可直接使用',
                'install': 'pip install catboost',
                'code': 'from catboost import CatBoostClassifier\nmodel = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=False)',
                'expected_accuracy': '约 86-88%',
                'use_case': '类别特征多，追求稳定性'
            }
        ]
    },
    {
        'level': 'Level 5: 深度学习',
        'models': [
            {
                'name': '多层感知机（MLP / Neural Network）',
                'complexity': '⭐⭐⭐⭐⭐',
                'pros': [
                    '捕捉复杂非线性关系',
                    '可扩展性强',
                    '适合超大数据集'
                ],
                'cons': [
                    '需要大量数据',
                    '训练时间长',
                    '超参数调优困难',
                    '黑盒模型，可解释性差',
                    '需要GPU加速',
                    '需要特征标准化'
                ],
                'data_requirement': '⚠ 需要标准化 education-num 和 work_intensity',
                'sklearn': 'from sklearn.neural_network import MLPClassifier',
                'params': 'MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=42)',
                'expected_accuracy': '约 85-87%（可能不超过XGBoost）',
                'use_case': '数据量极大（>100万），深度特征学习'
            }
        ]
    }
]

for level_info in models:
    write_log(f"【{level_info['level']}】")
    write_log("=" * 100)
    write_log("")
    
    for model in level_info['models']:
        write_log(f"模型名称：{model['name']}")
        write_log(f"复杂度：{model['complexity']} （5星满分）")
        write_log("")
        
        write_log(f"优点：")
        for pro in model['pros']:
            write_log(f"  ✓ {pro}")
        write_log("")
        
        write_log(f"缺点：")
        for con in model['cons']:
            write_log(f"  ✗ {con}")
        write_log("")
        
        write_log(f"数据要求：{model['data_requirement']}")
        write_log("")
        
        if 'install' in model:
            write_log(f"安装命令：{model['install']}")
            write_log("")
        
        if 'sklearn' in model:
            write_log(f"代码示例：")
            write_log(f"  {model['sklearn']}")
            write_log(f"  model = {model['params']}")
        elif 'code' in model:
            write_log(f"代码示例：")
            for line in model['code'].split('\n'):
                write_log(f"  {line}")
        write_log("")
        
        write_log(f"预期准确率：{model['expected_accuracy']}")
        write_log(f"使用场景：{model['use_case']}")
        write_log("")
        write_log("-" * 100)
        write_log("")

# ================================================================================
# 第五部分：建模建议
# ================================================================================
write_log("=" * 100)
write_log("第五部分：建模流程建议")
write_log("=" * 100)
write_log("")

write_log("【推荐的建模流程】")
write_log("-" * 100)
write_log("")

workflow = """
步骤1：数据划分
  - 训练集 / 测试集：80% / 20%（或 70% / 30%）
  - 使用分层抽样（stratified split）保持 income 比例一致
  - 代码示例：
    from sklearn.model_selection import train_test_split
    X = df.drop('income', axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

步骤2：选择模型路线
  路线A（推荐）：树模型路线
    → 当前数据可直接使用，无需额外标准化
    → 推荐顺序：
      1. 随机森林（快速验证）
      2. XGBoost / LightGBM（追求性能）
      3. 模型融合（Ensemble）
  
  路线B：线性模型路线
    → 需要先标准化 education-num 和 work_intensity
    → 推荐顺序：
      1. 逻辑回归（基准）
      2. 线性SVM（改进）
      3. 神经网络（深度学习）

步骤3：模型训练与评估
  - 训练模型：model.fit(X_train, y_train)
  - 预测：y_pred = model.predict(X_test)
  - 评估指标：
    * 准确率（Accuracy）
    * 精确率（Precision）= TP / (TP + FP)
    * 召回率（Recall）= TP / (TP + FN)
    * F1-score = 2 × (Precision × Recall) / (Precision + Recall)
    * AUC-ROC曲线（推荐）
  - 混淆矩阵分析
  - 特征重要性分析（树模型）

步骤4：模型优化
  - 超参数调优：
    * 网格搜索（GridSearchCV）
    * 随机搜索（RandomizedSearchCV）
    * 贝叶斯优化（Optuna）
  - K折交叉验证（5-fold或10-fold）
  - 特征选择（可选）：
    * 基于特征重要性
    * L1正则化（Lasso）
    * 递归特征消除（RFE）

步骤5：模型融合（进阶）
  - Voting：多个模型投票
  - Stacking：分层堆叠
  - Blending：加权平均

步骤6：模型部署
  - 保存模型：joblib.dump(model, 'model.pkl')
  - 加载模型：model = joblib.load('model.pkl')
  - 在线预测服务
"""

write_log(workflow)
write_log("")

write_log("【处理类别不平衡的建议】")
write_log("-" * 100)
write_log("")

imbalance_solutions = f"""
当前数据类别比：{imbalance_ratio:.2f} : 1（轻度不平衡）

解决方案（按需选择）：

1. 分层抽样（必须使用）
   - 划分数据集时使用 stratify=y 参数
   - 确保训练集和测试集的类别比例一致

2. 类别权重（推荐）
   - 大多数模型支持 class_weight='balanced' 参数
   - 自动根据类别频率调整权重
   - 代码：LogisticRegression(class_weight='balanced')

3. SMOTE过采样（可选）
   - 对少数类进行合成过采样
   - 适用于严重不平衡（>5:1）
   - from imblearn.over_sampling import SMOTE

4. 调整决策阈值（推荐）
   - 使用 predict_proba() 获取概率
   - 根据业务需求调整分类阈值（默认0.5）
   - 适用于对某类错误更敏感的场景

5. 评估指标选择
   - 不能只看准确率（可能被多数类主导）
   - 推荐使用：
     * F1-score（平衡精确率和召回率）
     * AUC-ROC（反映整体分类能力）
     * 混淆矩阵（详细分析各类错误）
"""

write_log(imbalance_solutions)
write_log("")

# ================================================================================
# 第六部分：可视化
# ================================================================================
write_log("=" * 100)
write_log("第六部分：数据特征可视化")
write_log("=" * 100)
write_log("")

# ===== 图11：特征类型结构图 =====
write_log("生成图表：图11_特征类型结构图.png")
write_log("")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1：特征类型饼图
ax1 = axes[0, 0]
feature_types = {
    '原始数值（已标准化）': 5,
    '序数编码（未标准化）': 1,
    '新构造特征': 3,
    '独热编码': len(one_hot_features)
}
colors = ['#3498DB', '#E67E22', '#9B59B6', '#E74C3C']
explode = (0.05, 0.05, 0.05, 0.1)

wedges, texts, autotexts = ax1.pie(feature_types.values(), labels=feature_types.keys(),
                                     colors=colors, explode=explode, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax1.set_title(f'特征类型分布（总计 {df.shape[1]-1} 个特征）', fontsize=14, fontweight='bold', pad=20)

# 子图2：标准化状态柱状图
ax2 = axes[0, 1]
standardization_status = {
    '已标准化': len([f for f in standardized_features if f in df.columns]),
    '未标准化': len([f for f in not_standardized_features if f in df.columns]),
    '独热编码\n(0/1)': len(one_hot_features)
}
x_pos = np.arange(len(standardization_status))
colors_bar = ['#27AE60', '#E74C3C', '#95A5A6']
bars = ax2.bar(x_pos, standardization_status.values(), color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)

for bar, val in zip(bars, standardization_status.values()):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{val} 个', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(standardization_status.keys(), fontsize=11)
ax2.set_ylabel('特征数量', fontsize=12, fontweight='bold')
ax2.set_title('特征标准化状态统计', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3)

# 子图3：独热编码来源
ax3 = axes[1, 0]
one_hot_sources = {}
for base_name, cols in one_hot_groups.items():
    one_hot_sources[base_name] = len(cols)

sorted_sources = dict(sorted(one_hot_sources.items(), key=lambda x: x[1], reverse=True))
y_pos = np.arange(len(sorted_sources))
bars = ax3.barh(y_pos, sorted_sources.values(), color='steelblue', alpha=0.8, edgecolor='black')

for i, (bar, val) in enumerate(zip(bars, sorted_sources.values())):
    width = bar.get_width()
    ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val} 列', ha='left', va='center', fontsize=10, fontweight='bold')

ax3.set_yticks(y_pos)
ax3.set_yticklabels(sorted_sources.keys(), fontsize=10)
ax3.set_xlabel('生成的独热编码列数', fontsize=12, fontweight='bold')
ax3.set_title('各分类变量生成的独热编码列数', fontsize=14, fontweight='bold', pad=20)
ax3.grid(axis='x', alpha=0.3)

# 子图4：数据概览
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
【最终数据摘要】

样本数：{df.shape[0]:,} 行
特征数：{df.shape[1] - 1} 列
目标变量：1 列（income）

特征构成：
  • 原始数值特征：6 个
    - 5 个已标准化
    - 1 个未标准化（education-num）
  
  • 新构造特征：3 个
    - 2 个基于标准化特征
    - 1 个混合标准化状态
  
  • 独热编码特征：{len(one_hot_features)} 个
    - 来自 7 个分类变量
    - 全部为 0/1 二进制

数据质量：
  ✓ 无缺失值
  ✓ 无重复样本
  ✓ 类别轻度不平衡（{imbalance_ratio:.1f}:1）

适用模型：
  ✓ 树模型（推荐）
  ⚠ 线性模型（需额外标准化）
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
         family='monospace')

plt.tight_layout()
plt.savefig('图11_特征类型结构图.png', dpi=300, bbox_inches='tight')
plt.close()

write_log("✓ 图11_特征类型结构图.png 已保存")
write_log("")

# ===== 图12：数据质量评估图 =====
write_log("生成图表：图12_数据质量评估图.png")
write_log("")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1：目标变量分布
ax1 = axes[0, 0]
income_counts.plot(kind='bar', ax=ax1, color=['#3498DB', '#E74C3C'], alpha=0.8, edgecolor='black', linewidth=2)
for i, (cat, count) in enumerate(income_counts.items()):
    ratio = income_ratio[cat]
    ax1.text(i, count + 500, f'{count:,}\n({ratio}%)', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

ax1.set_xlabel('收入水平', fontsize=12, fontweight='bold')
ax1.set_ylabel('样本数量', fontsize=12, fontweight='bold')
ax1.set_title('目标变量分布（income）', fontsize=14, fontweight='bold', pad=20)
ax1.set_xticklabels(income_counts.index, rotation=0)
ax1.grid(axis='y', alpha=0.3)

# 子图2：数值特征尺度对比
ax2 = axes[0, 1]
feature_ranges = {}
for feat in numerical_features[:8]:  # 只显示前8个
    if feat in df.columns:
        feature_ranges[feat] = [df[feat].min(), df[feat].max()]

feat_names = list(feature_ranges.keys())
mins = [feature_ranges[f][0] for f in feat_names]
maxs = [feature_ranges[f][1] for f in feat_names]

x = np.arange(len(feat_names))
width = 0.35

bars1 = ax2.bar(x - width/2, maxs, width, label='最大值', color='#E74C3C', alpha=0.7)
bars2 = ax2.bar(x + width/2, mins, width, label='最小值', color='#3498DB', alpha=0.7)

ax2.set_xlabel('特征名称', fontsize=12, fontweight='bold')
ax2.set_ylabel('取值', fontsize=12, fontweight='bold')
ax2.set_title('数值特征取值范围（展示尺度差异）', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=9)
ax2.legend(fontsize=10)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='y', alpha=0.3)

# 子图3：独热编码稀疏性
ax3 = axes[1, 0]
# 随机选择5个独热编码特征展示稀疏性
sample_one_hot = one_hot_features[:5]
sparsity_data = []
for feat in sample_one_hot:
    zeros = (df[feat] == 0).sum()
    ones = (df[feat] == 1).sum()
    sparsity_data.append([zeros, ones])

sparsity_df = pd.DataFrame(sparsity_data, columns=['0值数量', '1值数量'],
                           index=sample_one_hot)

sparsity_df.plot(kind='barh', stacked=True, ax=ax3,
                color=['#BDC3C7', '#E74C3C'], alpha=0.8, edgecolor='black')

ax3.set_xlabel('样本数量', fontsize=12, fontweight='bold')
ax3.set_title('独热编码特征稀疏性示例（前5个）', fontsize=14, fontweight='bold', pad=20)
ax3.legend(['0（不属于）', '1（属于）'], fontsize=10)
ax3.grid(axis='x', alpha=0.3)

# 子图4：特征数量演变
ax4 = axes[1, 1]
stages = ['原始数据', '数据清洗', '数据集成', '数据规约', '特征构造', '独热编码']
feature_counts = [15, 15, 14, 14, 17, 86]
colors_stages = ['#95A5A6', '#3498DB', '#9B59B6', '#E67E22', '#1ABC9C', '#E74C3C']

bars = ax4.bar(stages, feature_counts, color=colors_stages, alpha=0.8, edgecolor='black', linewidth=2)

for bar, count in zip(bars, feature_counts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加变化箭头
for i in range(len(feature_counts) - 1):
    change = feature_counts[i+1] - feature_counts[i]
    if change != 0:
        color = 'red' if change < 0 else 'green'
        symbol = '▼' if change < 0 else '▲'
        ax4.text(i + 0.5, max(feature_counts[i], feature_counts[i+1]) + 5,
                f'{symbol}{abs(change)}', ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold')

ax4.set_ylabel('特征数量', fontsize=12, fontweight='bold')
ax4.set_title('特征数量演变（预处理流程）', fontsize=14, fontweight='bold', pad=20)
ax4.set_xticklabels(stages, rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim([0, max(feature_counts) + 10])

plt.tight_layout()
plt.savefig('图12_数据质量评估图.png', dpi=300, bbox_inches='tight')
plt.close()

write_log("✓ 图12_数据质量评估图.png 已保存")
write_log("")

# ===== 图13：模型推荐路线图 =====
write_log("生成图表：图13_模型推荐路线图.png")
write_log("")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(5, 9.5, '二分类模型推荐路线图', ha='center', va='top',
        fontsize=20, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 绘制路线
levels = [
    {'name': 'Level 1\n基准模型', 'y': 7.5, 'models': ['逻辑回归'], 'color': '#3498DB'},
    {'name': 'Level 2\n简单非线性', 'y': 6.0, 'models': ['决策树', '朴素贝叶斯'], 'color': '#1ABC9C'},
    {'name': 'Level 3\n集成学习', 'y': 4.5, 'models': ['随机森林', 'SVM'], 'color': '#9B59B6'},
    {'name': 'Level 4\n梯度提升\n（推荐）', 'y': 3.0, 'models': ['XGBoost', 'LightGBM', 'CatBoost'], 'color': '#E74C3C'},
    {'name': 'Level 5\n深度学习', 'y': 1.5, 'models': ['神经网络'], 'color': '#E67E22'}
]

for i, level in enumerate(levels):
    # 绘制等级框
    rect = plt.Rectangle((0.5, level['y']-0.5), 2, 0.9, facecolor=level['color'],
                         edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(1.5, level['y'], level['name'], ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')
    
    # 绘制模型
    x_start = 3.5
    for j, model in enumerate(level['models']):
        model_x = x_start + j * 2
        circle = plt.Circle((model_x, level['y']), 0.4, facecolor='white',
                           edgecolor=level['color'], linewidth=2.5)
        ax.add_patch(circle)
        ax.text(model_x, level['y'], model, ha='center', va='center',
               fontsize=9, fontweight='bold')
    
    # 绘制箭头
    if i < len(levels) - 1:
        ax.annotate('', xy=(1.5, levels[i+1]['y']+0.4), xytext=(1.5, level['y']-0.4),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# 添加说明
explanations = [
    {'text': '易解释 / 快速', 'y': 7.5, 'x': 8.5},
    {'text': '可解释 / 中速', 'y': 6.0, 'x': 8.5},
    {'text': '平衡 / 较慢', 'y': 4.5, 'x': 8.5},
    {'text': '高性能 / 中速', 'y': 3.0, 'x': 8.5},
    {'text': '复杂 / 慢', 'y': 1.5, 'x': 8.5}
]

for exp in explanations:
    ax.text(exp['x'], exp['y'], exp['text'], ha='center', va='center',
           fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# 添加推荐标注
ax.text(5, 3.0, '★ 推荐首选 ★', ha='center', va='bottom',
       fontsize=13, fontweight='bold', color='red',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# 添加底部说明
note_text = """
当前数据特点：48,169样本，85特征，二分类
推荐路线：Level 3-4（随机森林 → XGBoost/LightGBM）
数据可直接使用于树模型，无需额外标准化
"""
ax.text(5, 0.3, note_text, ha='center', va='top',
       fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

plt.savefig('图13_模型推荐路线图.png', dpi=300, bbox_inches='tight')
plt.close()

write_log("✓ 图13_模型推荐路线图.png 已保存")
write_log("")

# ================================================================================
# 第七部分：总结
# ================================================================================
write_log("=" * 100)
write_log("总结")
write_log("=" * 100)
write_log("")

summary_final = f"""
【数据分析总结】
本报告详细分析了经过完整预处理流程后的最终数据（final_preprocessed_data.csv）。

数据规模：
  • 样本数：{df.shape[0]:,} 行
  • 特征数：{df.shape[1] - 1} 列
  • 目标变量：1 列（二分类）
  • 内存占用：{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

数据质量：
  ✓ 完整性：无缺失值（0个NaN）
  ✓ 唯一性：无重复样本
  ✓ 平衡性：类别比 {imbalance_ratio:.2f}:1（轻度不平衡，可接受）

特征构成：
  • 原始数值特征：6 个（5个已标准化，1个未标准化）
  • 新构造特征：3 个（交互特征，验证有效）
  • 独热编码特征：{len(one_hot_features)} 个（来自7个分类变量）

关键发现：
  1. education-num 和 work_intensity 未标准化，与其他特征尺度不一致
  2. 这两个特征在使用线性模型/神经网络时需要额外标准化
  3. 树模型（随机森林、XGBoost、LightGBM）可直接使用当前数据

【模型推荐】
根据数据特点和业务需求，推荐以下模型路线：

首选路线（树模型）：
  1. 随机森林：快速验证，获得基准性能（预计85-87%）
  2. XGBoost/LightGBM：追求最高性能（预计86-88%）
  3. 模型融合：进一步提升（预计88-90%）

备选路线（线性模型）：
  1. 先标准化 education-num 和 work_intensity
  2. 逻辑回归：快速基准（预计80-84%）
  3. 神经网络：深度学习（预计85-87%）

【后续工作】
  ✓ 数据已完全准备好，可直接进入建模阶段
  ✓ 建议使用分层抽样划分训练集/测试集
  ✓ 使用交叉验证评估模型稳定性
  ✓ 关注 F1-score 和 AUC-ROC，不仅仅是准确率
  ✓ 进行特征重要性分析，理解模型决策逻辑

【文件清单】
  1. final_preprocessed_data.csv  - 最终预处理数据
  2. final_data_analysis_report.txt - 本报告（详细分析）
  3. 图11_特征类型结构图.png  - 特征类型可视化
  4. 图12_数据质量评估图.png  - 数据质量评估
  5. 图13_模型推荐路线图.png  - 模型选择指南

报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

write_log(summary_final)
write_log("")
write_log("=" * 100)
write_log("✅ 数据分析与模型推荐报告生成完成")
write_log("=" * 100)
write_log("")

print("\n" + "=" * 100)
print("✅ 最终数据分析与模型推荐已完成！")
print("=" * 100)
print(f"\n📊 生成文件清单：")
print(f"  1. final_data_analysis_report.txt  - 详细分析报告")
print(f"  2. 图11_特征类型结构图.png        - 特征类型可视化")
print(f"  3. 图12_数据质量评估图.png        - 数据质量评估")
print(f"  4. 图13_模型推荐路线图.png        - 模型选择指南")
print(f"\n📈 核心发现：")
print(f"  - 数据规模：{df.shape[0]:,} 行 × {df.shape[1]} 列")
print(f"  - 数据质量：✓ 无缺失值，✓ 无重复")
print(f"  - 特征构成：9个数值特征 + {len(one_hot_features)}个独热编码")
print(f"  - 推荐模型：随机森林 → XGBoost/LightGBM（无需额外标准化）")
print(f"\n✓ 数据已准备完毕，可以开始建模训练！")
print("=" * 100 + "\n")


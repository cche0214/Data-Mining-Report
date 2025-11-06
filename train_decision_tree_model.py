# -*- coding: utf-8 -*-
"""
================================================================================
决策树模型训练与评估
================================================================================
模型：Decision Tree Classifier
数据划分：训练集70% / 测试集30%（分层抽样）
类别不平衡处理：class_weight='balanced'
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

# 设置matplotlib参数（使用英文，避免中文乱码）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 初始化日志
log_file = 'model_training_log.txt'

def write_log(message, print_console=True):
    """写入日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')
    if print_console:
        print(message)

# 清空日志
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Decision Tree Model Training Report\n")
    f.write("=" * 80 + "\n\n")

print("=" * 80)
print("Decision Tree Model Training")
print("=" * 80)
print()

# ================================================================================
# Step 1: Load Data
# ================================================================================
write_log("=" * 80)
write_log("Step 1: Loading Data")
write_log("=" * 80)
write_log("")

df = pd.read_csv('final_preprocessed_data.csv')
write_log(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
write_log("")

# Separate features and target
X = df.drop('income', axis=1)
y = df['income']

write_log(f"Features: {X.shape[1]} columns")
write_log(f"Target distribution:")
for label, count in y.value_counts().items():
    write_log(f"  {label}: {count} ({count/len(y)*100:.2f}%)")
write_log("")

# ================================================================================
# Step 2: Train-Test Split (70-30, Stratified)
# ================================================================================
write_log("=" * 80)
write_log("Step 2: Train-Test Split")
write_log("=" * 80)
write_log("")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

write_log("Split configuration:")
write_log("  - Train size: 70%")
write_log("  - Test size: 30%")
write_log("  - Stratified: Yes (maintain class distribution)")
write_log("")

write_log(f"Training set: {X_train.shape[0]} samples")
write_log(f"Test set: {X_test.shape[0]} samples")
write_log("")

write_log("Training set distribution:")
for label, count in y_train.value_counts().items():
    write_log(f"  {label}: {count} ({count/len(y_train)*100:.2f}%)")
write_log("")

write_log("Test set distribution:")
for label, count in y_test.value_counts().items():
    write_log(f"  {label}: {count} ({count/len(y_test)*100:.2f}%)")
write_log("")

# ================================================================================
# Step 3: Handle Class Imbalance
# ================================================================================
write_log("=" * 80)
write_log("Step 3: Handling Class Imbalance")
write_log("=" * 80)
write_log("")

write_log("Method: class_weight='balanced'")
write_log("")
write_log("Explanation:")
write_log("  The 'balanced' mode uses the values of y to automatically adjust")
write_log("  weights inversely proportional to class frequencies:")
write_log("  weight = n_samples / (n_classes * np.bincount(y))")
write_log("")

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
write_log("Computed class weights:")
for label, weight in zip(np.unique(y_train), class_weights):
    write_log(f"  {label}: {weight:.4f}")
write_log("")
write_log("This ensures the minority class (>50K) has more influence during training.")
write_log("")

# ================================================================================
# Step 4: Train Decision Tree Model
# ================================================================================
write_log("=" * 80)
write_log("Step 4: Training Decision Tree Model")
write_log("=" * 80)
write_log("")

write_log("Model configuration:")
write_log("  - Algorithm: Decision Tree Classifier")
write_log("  - max_depth: 15 (prevent overfitting)")
write_log("  - min_samples_split: 50 (minimum samples to split a node)")
write_log("  - min_samples_leaf: 20 (minimum samples in a leaf)")
write_log("  - class_weight: 'balanced' (handle imbalance)")
write_log("  - random_state: 42 (reproducibility)")
write_log("")

model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)

write_log("Training started...")
start_time = datetime.now()
model.fit(X_train, y_train)
end_time = datetime.now()
training_time = (end_time - start_time).total_seconds()

write_log(f"Training completed in {training_time:.2f} seconds")
write_log("")

write_log("Model structure:")
write_log(f"  - Total nodes: {model.tree_.node_count}")
write_log(f"  - Max depth: {model.get_depth()}")
write_log(f"  - Number of leaves: {model.get_n_leaves()}")
write_log("")

# ================================================================================
# Step 5: Make Predictions
# ================================================================================
write_log("=" * 80)
write_log("Step 5: Making Predictions")
write_log("=" * 80)
write_log("")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[:, 1]

write_log("Predictions completed")
write_log("")

# ================================================================================
# Step 6: Evaluate Model Performance
# ================================================================================
write_log("=" * 80)
write_log("Step 6: Model Performance Evaluation")
write_log("=" * 80)
write_log("")

# Training set metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, pos_label='>50K')
train_recall = recall_score(y_train, y_train_pred, pos_label='>50K')
train_f1 = f1_score(y_train, y_train_pred, pos_label='>50K')

write_log("Training Set Performance:")
write_log(f"  Accuracy:  {train_accuracy:.4f}")
write_log(f"  Precision: {train_precision:.4f}")
write_log(f"  Recall:    {train_recall:.4f}")
write_log(f"  F1-Score:  {train_f1:.4f}")
write_log("")

# Test set metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, pos_label='>50K')
test_recall = recall_score(y_test, y_test_pred, pos_label='>50K')
test_f1 = f1_score(y_test, y_test_pred, pos_label='>50K')

write_log("Test Set Performance:")
write_log(f"  Accuracy:  {test_accuracy:.4f}")
write_log(f"  Precision: {test_precision:.4f}")
write_log(f"  Recall:    {test_recall:.4f}")
write_log(f"  F1-Score:  {test_f1:.4f}")
write_log("")

# ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_test_proba, pos_label='>50K')
roc_auc = auc(fpr, tpr)
write_log(f"ROC-AUC Score: {roc_auc:.4f}")
write_log("")

# Average Precision
avg_precision = average_precision_score(y_test, y_test_proba, pos_label='>50K')
write_log(f"Average Precision Score: {avg_precision:.4f}")
write_log("")

# Detailed classification report
write_log("Detailed Classification Report (Test Set):")
write_log("")
report = classification_report(y_test, y_test_pred)
write_log(report)
write_log("")

# ================================================================================
# Step 7: Visualizations
# ================================================================================
write_log("=" * 80)
write_log("Step 7: Generating Visualizations")
write_log("=" * 80)
write_log("")

# Figure 14: Confusion Matrix
write_log("Generating Figure 14: Confusion Matrix...")
cm = confusion_matrix(y_test, y_test_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'],
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('Figure_14_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.close()
write_log("  Saved: Figure_14_Confusion_Matrix.png")
write_log("")

# Figure 15: ROC Curve
write_log("Generating Figure 15: ROC Curve...")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figure_15_ROC_Curve.png', dpi=300, bbox_inches='tight')
plt.close()
write_log("  Saved: Figure_15_ROC_Curve.png")
write_log("")

# Figure 16: Precision-Recall Curve
write_log("Generating Figure 16: Precision-Recall Curve...")
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba, pos_label='>50K')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall_curve, precision_curve, color='darkgreen', lw=2, 
        label=f'PR curve (AP = {avg_precision:.4f})')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc="lower left", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Figure_16_Precision_Recall_Curve.png', dpi=300, bbox_inches='tight')
plt.close()
write_log("  Saved: Figure_16_Precision_Recall_Curve.png")
write_log("")

# Figure 17: Feature Importance
write_log("Generating Figure 17: Feature Importance (Top 20)...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

top_20 = feature_importance.head(20)

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(top_20))
ax.barh(y_pos, top_20['importance'].values, color='steelblue', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_20['feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Feature Importance', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('Figure_17_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.close()
write_log("  Saved: Figure_17_Feature_Importance.png")
write_log("")

# Save full feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
write_log("  Feature importance data saved to: feature_importance.csv")
write_log("")

# Figure 18: Performance Metrics Comparison
write_log("Generating Figure 18: Performance Metrics Comparison...")
metrics_train = [train_accuracy, train_precision, train_recall, train_f1]
metrics_test = [test_accuracy, test_precision, test_recall, test_f1]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, metrics_train, width, label='Training Set', 
               color='skyblue', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, metrics_test, width, label='Test Set', 
               color='lightcoral', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Figure_18_Performance_Metrics.png', dpi=300, bbox_inches='tight')
plt.close()
write_log("  Saved: Figure_18_Performance_Metrics.png")
write_log("")

# Figure 19: Decision Tree Structure (partial visualization)
write_log("Generating Figure 19: Decision Tree Structure (partial)...")
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(model, 
          max_depth=3,  # Only show top 3 levels
          feature_names=X.columns,
          class_names=['<=50K', '>50K'],
          filled=True,
          rounded=True,
          fontsize=8,
          ax=ax)
ax.set_title('Decision Tree Structure (Top 3 Levels)', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('Figure_19_Tree_Structure.png', dpi=300, bbox_inches='tight')
plt.close()
write_log("  Saved: Figure_19_Tree_Structure.png")
write_log("  Note: Only top 3 levels shown for clarity (full tree has depth {})".format(model.get_depth()))
write_log("")

# Figure 20: Model Complexity Analysis
write_log("Generating Figure 20: Model Complexity Analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Tree depth vs nodes
depths = list(range(1, model.get_depth() + 1))
nodes_at_depth = []
for d in depths:
    # Count nodes at each depth (approximate)
    nodes_at_depth.append(2**d if d < 10 else model.tree_.node_count)

ax1.plot(depths[:min(15, len(depths))], nodes_at_depth[:15], 
         marker='o', linewidth=2, markersize=6, color='darkblue')
ax1.set_xlabel('Tree Depth', fontsize=12, fontweight='bold')
ax1.set_ylabel('Approximate Nodes', fontsize=12, fontweight='bold')
ax1.set_title('Tree Growth by Depth', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)

# Class distribution in predictions
pred_dist = pd.Series(y_test_pred).value_counts()
ax2.bar(['<=50K', '>50K'], 
        [pred_dist.get('<=50K', 0), pred_dist.get('>50K', 0)],
        color=['skyblue', 'lightcoral'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Test Set Predictions Distribution', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for i, (label, count) in enumerate(pred_dist.items()):
    ax2.text(i, count + 100, f'{count}\n({count/len(y_test_pred)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('Figure_20_Model_Complexity.png', dpi=300, bbox_inches='tight')
plt.close()
write_log("  Saved: Figure_20_Model_Complexity.png")
write_log("")

# ================================================================================
# Step 8: Summary
# ================================================================================
write_log("=" * 80)
write_log("Step 8: Training Summary")
write_log("=" * 80)
write_log("")

summary = f"""
Training Summary:
-----------------
Model: Decision Tree Classifier
Training Time: {training_time:.2f} seconds
Tree Depth: {model.get_depth()}
Total Nodes: {model.tree_.node_count}
Number of Leaves: {model.get_n_leaves()}

Performance (Test Set):
-----------------------
Accuracy:  {test_accuracy:.4f}
Precision: {test_precision:.4f}
Recall:    {test_recall:.4f}
F1-Score:  {test_f1:.4f}
ROC-AUC:   {roc_auc:.4f}

Class Imbalance Handling:
-------------------------
Method: class_weight='balanced'
Effect: Minority class (>50K) weighted {class_weights[1]/class_weights[0]:.2f}x higher

Generated Files:
----------------
1. Figure_14_Confusion_Matrix.png
2. Figure_15_ROC_Curve.png
3. Figure_16_Precision_Recall_Curve.png
4. Figure_17_Feature_Importance.png
5. Figure_18_Performance_Metrics.png
6. Figure_19_Tree_Structure.png
7. Figure_20_Model_Complexity.png
8. feature_importance.csv
9. model_training_log.txt

Conclusion:
-----------
The Decision Tree model has been successfully trained and evaluated.
Test accuracy: {test_accuracy:.2%}
The model shows {'good' if test_accuracy > 0.80 else 'reasonable'} performance on the test set.
{'Overfitting is minimal.' if abs(train_accuracy - test_accuracy) < 0.05 else 'Some overfitting detected (train-test gap: ' + f'{abs(train_accuracy - test_accuracy):.2%}' + ').'}
"""

write_log(summary)
write_log("")

write_log("=" * 80)
write_log("Training Complete!")
write_log("=" * 80)
write_log("")

print("\n" + "=" * 80)
print("Decision Tree Training Complete!")
print("=" * 80)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"\nAll figures and logs saved successfully.")
print("=" * 80 + "\n")


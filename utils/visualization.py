import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
import pandas as pd

def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """绘制混淆矩阵
    
    参数:
        y_true: array-like, 真实标签
        y_pred: array-like, 预测标签
        labels: list, 标签名称列表
        output_path: str/Path, 保存路径
    """
    set_chinese_font()
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, labels, output_path):
    """绘制ROC曲线
    
    参数:
        y_true: array-like, 真实标签
        y_pred_proba: array-like, 预测概率
        labels: list, 标签名称列表
        output_path: str/Path, 保存路径
    """
    set_chinese_font()
    plt.figure(figsize=(10, 8))
    
    # 将真实标签转换为one-hot编码
    y_true_bin = pd.get_dummies(y_true)
    
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin.iloc[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(feature_names, importance_scores, output_path, top_n=20):
    """绘制特征重要性
    
    参数:
        feature_names: array-like, 特征名称列表
        importance_scores: array-like, 特征重要性分数
        output_path: str/Path, 保存路径
        top_n: int, 显示前n个重要特征
    """
    set_chinese_font()
    # 获取前N个最重要的特征
    indices = np.argsort(importance_scores)[-top_n:]
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), importance_scores[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title(f'前{top_n}个最重要特征')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_probability_distribution(y_pred_proba, labels, output_path):
    """绘制预测概率分布
    
    参数:
        y_pred_proba: array-like, 预测概率矩阵
        labels: list, 标签名称列表
        output_path: str/Path, 保存路径
    """
    set_chinese_font()
    plt.figure(figsize=(12, 6))
    
    for i, label in enumerate(labels):
        # 使用直方图替代密度图
        plt.hist(y_pred_proba[:, i], bins=50, alpha=0.5, label=label, density=True)
    
    plt.xlabel('预测概率')
    plt.ylabel('密度')
    plt.title('各类别预测概率分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_class_distribution(y, labels, output_path):
    """绘制类别分布
    
    参数:
        y: array-like, 标签数组
        labels: list, 标签名称列表
        output_path: str/Path, 保存路径
    """
    set_chinese_font()
    plt.figure(figsize=(10, 6))
    
    # 计算每个类别的数量
    value_counts = pd.Series(y).value_counts()
    
    # 创建柱状图
    plt.bar(range(len(labels)), [value_counts.get(label, 0) for label in labels])
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('数据集类别分布')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_analysis_report(
    y_true, y_pred, y_pred_proba, 
    labels, feature_names, importance_scores,
    output_dir
):
    """生成综合分析报告
    
    参数:
        y_true: array-like, 真实标签
        y_pred: array-like, 预测标签
        y_pred_proba: array-like, 预测概率
        labels: list, 标签名称列表
        feature_names: array-like, 特征名称列表
        importance_scores: array-like, 特征重要性分数
        output_dir: str/Path, 输出目录
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制各种可视化图表
    plot_confusion_matrix(y_true, y_pred, labels, output_dir / 'confusion_matrix.png')
    plot_roc_curves(y_true, y_pred_proba, labels, output_dir / 'roc_curves.png')
    plot_feature_importance(feature_names, importance_scores, output_dir / 'feature_importance.png')
    plot_probability_distribution(y_pred_proba, labels, output_dir / 'probability_distribution.png')
    plot_class_distribution(y_true, labels, output_dir / 'class_distribution.png') 
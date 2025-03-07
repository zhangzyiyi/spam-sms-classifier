import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 尝试使用微软雅黑
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
except:
    try:
        # 尝试使用黑体
        font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
    except:
        # 如果都不存在，使用系统默认字体
        font = FontProperties()

class SMSClassifier:
    def __init__(self, alpha=1.0, max_features=5000):
        """初始化分类器
        
        参数:
            alpha: float, 拉普拉斯平滑参数
            max_features: int, 最大特征数
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.classifier = MultinomialNB(alpha=alpha)
        self.is_fitted = False
    
    def preprocess_features(self, texts):
        """将文本转换为TF-IDF特征"""
        if not self.is_fitted:
            return self.vectorizer.fit_transform(texts)
        return self.vectorizer.transform(texts)
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """训练模型并评估
        
        参数:
            X: array-like, 文本数据
            y: array-like, 标签
            test_size: float, 测试集比例
            random_state: int, 随机种子
        """
        # 特征提取
        X_features = self.preprocess_features(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=random_state
        )
        
        # 训练模型
        self.classifier.fit(X_train, y_train)
        self.is_fitted = True
        
        # 预测
        y_pred = self.classifier.predict(X_test)
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 计算ROC曲线
        y_prob = self.classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'roc': {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            },
            'test_data': {
                'X': X_test,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
        }
    
    def predict(self, texts):
        """预测新文本的类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_features = self.preprocess_features(texts)
        return self.classifier.predict(X_features)
    
    def predict_proba(self, texts):
        """预测新文本的概率分布"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_features = self.preprocess_features(texts)
        return self.classifier.predict_proba(X_features)
    
    def cross_validate(self, X, y, cv=5):
        """交叉验证
        
        参数:
            X: array-like, 文本数据
            y: array-like, 标签
            cv: int, 交叉验证折数
        """
        X_features = self.preprocess_features(X)
        scores = cross_val_score(self.classifier, X_features, y, cv=cv)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def save_model(self, model_dir='models'):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存向量器
        joblib.dump(self.vectorizer, model_dir / 'vectorizer.joblib')
        # 保存分类器
        joblib.dump(self.classifier, model_dir / 'classifier.joblib')
    
    @classmethod
    def load_model(cls, model_dir='models'):
        """加载模型"""
        model_dir = Path(model_dir)
        
        instance = cls()
        # 加载向量器
        instance.vectorizer = joblib.load(model_dir / 'vectorizer.joblib')
        # 加载分类器
        instance.classifier = joblib.load(model_dir / 'classifier.joblib')
        instance.is_fitted = True
        
        return instance
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵热力图"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵', fontproperties=font, fontsize=14)
        plt.ylabel('真实标签', fontproperties=font, fontsize=12)
        plt.xlabel('预测标签', fontproperties=font, fontsize=12)
        
        # 添加标签说明
        plt.text(-0.1, -0.1, '0: 正常短信', fontproperties=font, ha='right')
        plt.text(-0.1, -0.2, '1: 垃圾短信', fontproperties=font, ha='right')
        
        return plt
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率', fontproperties=font, fontsize=12)
        plt.ylabel('真阳性率', fontproperties=font, fontsize=12)
        plt.title('接收者操作特征(ROC)曲线', fontproperties=font, fontsize=14)
        plt.legend(loc="lower right", prop=font)
        return plt
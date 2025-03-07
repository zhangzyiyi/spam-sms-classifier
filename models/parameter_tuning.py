from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTuner:
    def __init__(self):
        """初始化模型调优器"""
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
    
    def create_pipeline(self):
        """创建模型管道"""
        return Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])
    
    def get_param_grid(self):
        """定义参数网格"""
        return {
            'vectorizer__max_features': [1000, 3000, 5000, 7000],
            'vectorizer__min_df': [1, 2, 3],
            'vectorizer__max_df': [0.9, 0.95, 0.99],
            'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
        }
    
    def tune_parameters(self, X, y, cv=5, n_jobs=-1):
        """使用网格搜索调优参数
        
        参数:
            X: array-like, 训练数据
            y: array-like, 标签
            cv: int, 交叉验证折数
            n_jobs: int, 并行作业数
        """
        # 创建管道
        pipeline = self.create_pipeline()
        
        # 获取参数网格
        param_grid = self.get_param_grid()
        
        # 创建网格搜索对象
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring='f1',
            verbose=2
        )
        
        # 执行网格搜索
        print("开始参数调优...")
        grid_search.fit(X, y)
        
        # 保存结果
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.cv_results_ = grid_search.cv_results_
        
        return grid_search.best_estimator_
    
    def plot_param_scores(self, output_dir='plots'):
        """绘制参数调优结果图
        
        参数:
            output_dir: str, 输出目录
        """
        if self.cv_results_ is None:
            raise ValueError("请先运行参数调优")
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取参数和分数
        results = pd.DataFrame(self.cv_results_)
        
        # 绘制alpha参数的影响
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='param_classifier__alpha', y='mean_test_score', data=results)
        plt.title('Alpha参数对模型性能的影响')
        plt.xlabel('Alpha值')
        plt.ylabel('F1分数')
        plt.savefig(output_dir / 'alpha_scores.png')
        plt.close()
        
        # 绘制max_features参数的影响
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='param_vectorizer__max_features', y='mean_test_score', data=results)
        plt.title('特征数量对模型性能的影响')
        plt.xlabel('最大特征数')
        plt.ylabel('F1分数')
        plt.savefig(output_dir / 'max_features_scores.png')
        plt.close()
    
    def get_best_params_summary(self):
        """获取最佳参数摘要"""
        if self.best_params_ is None:
            raise ValueError("请先运行参数调优")
        
        # 计算每个参数的重要性（使用交叉验证分数的标准差）
        param_importance = {}
        for param in self.best_params_:
            # 使用列表推导式而不是生成器表达式
            scores = [self.cv_results_[f'split{i}_test_score'] for i in range(5)]
            param_importance[param] = float(np.std(scores))  # 确保转换为Python float
        
        return {
            'best_params': self.best_params_,
            'best_score': float(self.best_score_),  # 确保转换为Python float
            'param_importance': param_importance
        }
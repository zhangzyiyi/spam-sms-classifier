import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import json

class SpamClassifier:
    def __init__(self, config_path=None, skip_vectorization=False):
        """初始化分类器
        
        参数:
            config_path: str, 配置文件路径，包含阈值和类型映射
            skip_vectorization: bool, 是否跳过向量化步骤（当输入已经是向量时设为True）
        """
        # 加载配置
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # 创建分类器管道
        if skip_vectorization:
            self.pipeline = Pipeline([
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=50,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                ))
            ])
        else:
            self.pipeline = Pipeline([
                ('vectorizer', TfidfVectorizer(
                    max_features=10000,  # 增加特征数量
                    min_df=2,
                    max_df=0.9,
                    ngram_range=(1, 2)  # 添加二元词组特征
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=50,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                ))
            ])
        
        # 标签编码器
        self.label_encoder = LabelEncoder()
        
        # 训练状态
        self.is_fitted = False
    
    def _default_config(self):
        """默认配置"""
        return {
            'spam_types': {
                0: '正常短信',
                1: '广告营销',
                2: '诈骗短信',
                3: '违法信息'
            },
            'thresholds': {
                'spam': 0.4,  # 降低基础垃圾短信阈值
                'fraud': 0.3,  # 大幅降低诈骗短信阈值以提高召回率
                'illegal': 0.3  # 大幅降低违法信息阈值以提高召回率
            }
        }
    
    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_config(self, config_path):
        """保存配置"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def fit(self, X, y):
        """训练模型
        
        参数:
            X: array-like, 文本数据或向量
            y: array-like, 标签
        """
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 训练模型
        self.pipeline.fit(X, y_encoded)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 获取预测概率
        probas = self.pipeline.predict_proba(X)
        
        # 应用阈值进行多级分类
        predictions = []
        for proba in probas:
            # 获取每个类别的概率
            normal_prob = proba[0]
            spam_probs = proba[1:]  # 所有垃圾短信类别的概率
            
            # 如果正常短信概率足够高，直接判定为正常
            if normal_prob >= (1 - self.config['thresholds']['spam']):
                final_class = 0
            else:
                # 对于垃圾短信，优先检测诈骗和违法信息
                fraud_prob = proba[2]  # 诈骗短信概率
                illegal_prob = proba[3]  # 违法信息概率
                
                if fraud_prob >= self.config['thresholds']['fraud']:
                    final_class = 2  # 诈骗短信
                elif illegal_prob >= self.config['thresholds']['illegal']:
                    final_class = 3  # 违法信息
                else:
                    # 如果不是诈骗或违法信息，判断是否为广告
                    final_class = 1 if proba[1] == max(spam_probs) else 0
            
            predictions.append(final_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """预测概率分布"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        return self.pipeline.predict_proba(X)
    
    def get_prediction_details(self, X):
        """获取详细的预测结果
        
        返回：
            list of dict, 每个字典包含预测类别、概率和风险等级
        """
        probas = self.predict_proba(X)
        predictions = self.predict(X)
        
        results = []
        for proba, pred in zip(probas, predictions):
            max_prob = np.max(proba)
            
            # 确定风险等级
            if pred == 0:
                risk_level = '安全'
            elif pred == 1:
                risk_level = '低风险'
            elif pred == 2:
                risk_level = '高风险'
            else:
                risk_level = '严重风险'
            
            results.append({
                'prediction': self.config['spam_types'][int(pred)],
                'probability': float(max_prob),
                'risk_level': risk_level,
                'class_probabilities': {
                    self.config['spam_types'][i]: float(p)
                    for i, p in enumerate(proba)
                }
            })
        
        return results
    
    def save_model(self, model_dir='models'):
        """保存模型"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型组件
        joblib.dump(self.pipeline, model_dir / 'spam_pipeline.joblib')
        joblib.dump(self.label_encoder, model_dir / 'label_encoder.joblib')
        
        # 保存配置
        self.save_config(model_dir / 'config.json')
    
    @classmethod
    def load_model(cls, model_dir='models', skip_vectorization=False):
        """加载模型"""
        model_dir = Path(model_dir)
        
        # 创建实例
        instance = cls(config_path=model_dir / 'config.json', skip_vectorization=skip_vectorization)
        
        # 加载模型组件
        instance.pipeline = joblib.load(model_dir / 'spam_pipeline.joblib')
        instance.label_encoder = joblib.load(model_dir / 'label_encoder.joblib')
        instance.is_fitted = True
        
        return instance
    
    def optimize_thresholds(self, X_val, y_val, metric='f1'):
        """优化预测阈值
        
        参数:
            X_val: array-like, 验证集特征
            y_val: array-like, 验证集标签
            metric: str, 优化指标 ('f1', 'precision', 'recall')
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # 获取预测概率
        probas = self.predict_proba(X_val)
        
        # 定义候选阈值
        thresholds = np.linspace(0.1, 0.9, num=9)
        
        best_score = 0
        best_thresholds = self.config['thresholds'].copy()
        
        # 网格搜索最佳阈值组合
        for t1 in thresholds:  # spam阈值
            for t2 in thresholds[thresholds >= t1]:  # fraud阈值
                for t3 in thresholds[thresholds >= t2]:  # illegal阈值
                    # 临时更新阈值
                    self.config['thresholds'].update({
                        'spam': t1,
                        'fraud': t2,
                        'illegal': t3
                    })
                    
                    # 使用当前阈值进行预测
                    y_pred = self.predict(X_val)
                    
                    # 计算得分
                    if metric == 'f1':
                        score = f1_score(y_val, y_pred, average='weighted')
                    elif metric == 'precision':
                        score = precision_score(y_val, y_pred, average='weighted')
                    else:
                        score = recall_score(y_val, y_pred, average='weighted')
                    
                    # 更新最佳阈值
                    if score > best_score:
                        best_score = score
                        best_thresholds = self.config['thresholds'].copy()
        
        # 恢复最佳阈值
        self.config['thresholds'] = best_thresholds
        
        return {
            'best_thresholds': best_thresholds,
            'best_score': best_score
        }
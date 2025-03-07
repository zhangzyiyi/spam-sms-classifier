import jieba.analyse
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

class FeatureEngineer:
    def __init__(self, custom_dict_path=None, stop_words_path=None):
        """初始化特征工程类
        
        参数:
            custom_dict_path: str, 自定义词典路径
            stop_words_path: str, 停用词表路径
        """
        self.keywords_extractor = jieba.analyse.TFIDF()
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
        
        self.stop_words = set()
        if stop_words_path:
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                self.stop_words = set(line.strip() for line in f)
    
    def extract_keywords(self, text, topK=20):
        """提取文本关键词
        
        参数:
            text: str, 输入文本
            topK: int, 返回的关键词数量
        """
        return self.keywords_extractor.extract_tags(text, topK=topK)
    
    def extract_patterns(self, text):
        """提取特征模式
        
        提取：
        - 数字模式（如电话号码、金额等）
        - 特殊符号模式
        - URL模式
        - 邮箱模式
        """
        patterns = {
            'has_phone': bool(re.search(r'\d{11}|\d{3,4}[-\s]?\d{4}', text)),  # 手机号或座机号
            'has_url': bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),  # URL
            'has_email': bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)),  # 邮箱
            'has_price': bool(re.search(r'¥?\d+\.?\d*[万千百十]?[元块]', text)),  # 金额
            'has_date': bool(re.search(r'\d{1,2}[月日号]|\d{4}年', text)),  # 日期
            'has_time': bool(re.search(r'\d{1,2}[点时分]', text)),  # 时间
            'special_chars_count': len(re.findall(r'[!！?？。，,;；]', text)),  # 特殊字符数量
            'number_count': len(re.findall(r'\d', text)),  # 数字数量
        }
        return patterns
    
    def build_custom_features(self, text):
        """构建自定义特征"""
        # 提取关键词
        keywords = self.extract_keywords(text)
        # 提取模式特征
        patterns = self.extract_patterns(text)
        # 计算文本长度特征
        length_features = {
            'text_length': len(text),
            'word_count': len(list(jieba.cut(text))),
            'avg_word_length': np.mean([len(w) for w in jieba.cut(text)]) if text else 0
        }
        
        # 合并所有特征
        features = {**patterns, **length_features}
        return features, keywords
    
    def select_features(self, X, y, k=1000):
        """使用卡方检验选择最重要的特征
        
        参数:
            X: 特征矩阵
            y: 标签
            k: 选择的特征数量
        """
        selector = SelectKBest(chi2, k=k)
        X_new = selector.fit_transform(X, y)
        return X_new, selector
    
    def analyze_spam_patterns(self, texts, labels):
        """分析垃圾短信的特征模式
        
        参数:
            texts: list, 文本列表
            labels: list, 标签列表
        """
        spam_texts = [text for text, label in zip(texts, labels) if label == 1]
        normal_texts = [text for text, label in zip(texts, labels) if label == 0]
        
        # 分析关键词
        spam_keywords = []
        for text in spam_texts:
            spam_keywords.extend(self.extract_keywords(text))
        
        normal_keywords = []
        for text in normal_texts:
            normal_keywords.extend(self.extract_keywords(text))
        
        # 统计频率
        spam_word_freq = Counter(spam_keywords)
        normal_word_freq = Counter(normal_keywords)
        
        # 分析模式特征
        spam_patterns = [self.extract_patterns(text) for text in spam_texts]
        normal_patterns = [self.extract_patterns(text) for text in normal_texts]
        
        # 计算模式特征的统计信息
        spam_pattern_stats = {
            key: np.mean([p[key] for p in spam_patterns]) 
            for key in spam_patterns[0].keys()
        }
        
        normal_pattern_stats = {
            key: np.mean([p[key] for p in normal_patterns])
            for key in normal_patterns[0].keys()
        }
        
        return {
            'spam_keywords': spam_word_freq.most_common(20),
            'normal_keywords': normal_word_freq.most_common(20),
            'spam_patterns': spam_pattern_stats,
            'normal_patterns': normal_pattern_stats
        }
    
    def get_feature_importance(self, vectorizer, selector):
        """获取特征重要性
        
        参数:
            vectorizer: TfidfVectorizer实例
            selector: SelectKBest实例
        """
        feature_names = vectorizer.get_feature_names_out()
        if hasattr(selector, 'scores_'):
            scores = selector.scores_
            feature_scores = list(zip(feature_names, scores))
            return sorted(feature_scores, key=lambda x: x[1], reverse=True)
        return [] 
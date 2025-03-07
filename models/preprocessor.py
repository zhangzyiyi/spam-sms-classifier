import pandas as pd
import re
import jieba
import numpy as np
from pathlib import Path
from sklearn.utils import resample

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set()
        self._load_stopwords()
    
    def _load_stopwords(self):
        """加载停用词表"""
        stopwords_path = Path(__file__).parent.parent / 'data' / 'stopwords.txt'
        if stopwords_path.exists():
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f)
    
    def load_data(self, file_path):
        """加载原始数据并转换为DataFrame格式"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        label, content = line.split('\t', 1)
                        data.append({
                            'label': int(label),
                            'content': content.strip()
                        })
                    except ValueError:
                        continue  # 跳过格式不正确的行
        
        return pd.DataFrame(data)
    
    def clean_text(self, text):
        """清洗文本内容"""
        # 保留xxx模式作为特征
        text = re.sub(r'x+', 'xxx', text)
        # 统一数字格式
        text = re.sub(r'\d+', 'NUM', text)
        # 去除特殊字符和多余空格
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def segment_text(self, text):
        """对文本进行分词"""
        words = jieba.cut(text)
        # 过滤停用词
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)
    
    def balance_data(self, data, method='undersample', random_state=42):
        """平衡数据集
        
        参数:
            data: DataFrame, 原始数据
            method: str, 'undersample' 或 'oversample'
            random_state: int, 随机种子
        """
        # 分离多数类和少数类
        data_majority = data[data['label'] == 0]
        data_minority = data[data['label'] == 1]
        
        # 打印原始数据分布
        print(f"\n原始数据分布:")
        print(f"正常短信数量: {len(data_majority)}")
        print(f"垃圾短信数量: {len(data_minority)}")
        
        if method == 'undersample':
            # 对多数类进行欠采样
            data_majority_downsampled = resample(
                data_majority,
                replace=False,
                n_samples=len(data_minority),
                random_state=random_state
            )
            # 合并数据
            data_balanced = pd.concat([data_majority_downsampled, data_minority])
        
        elif method == 'oversample':
            # 对少数类进行过采样
            data_minority_upsampled = resample(
                data_minority,
                replace=True,
                n_samples=len(data_majority),
                random_state=random_state
            )
            # 合并数据
            data_balanced = pd.concat([data_majority, data_minority_upsampled])
        
        # 打印平衡后的数据分布
        print(f"\n平衡后的数据分布:")
        print(f"正常短信数量: {sum(data_balanced['label'] == 0)}")
        print(f"垃圾短信数量: {sum(data_balanced['label'] == 1)}")
        
        return data_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    def preprocess(self, data, balance_method=None):
        """完整的预处理流程"""
        # 清洗文本
        print("正在清洗文本...")
        data['cleaned_content'] = data['content'].apply(self.clean_text)
        
        # 分词
        print("正在进行分词...")
        data['segmented_content'] = data['cleaned_content'].apply(self.segment_text)
        
        # 数据平衡（如果指定了平衡方法）
        if balance_method:
            print(f"正在使用{balance_method}方法平衡数据...")
            data = self.balance_data(data, method=balance_method)
        
        return data
    
    def save_processed_data(self, data, output_path):
        """保存处理后的数据"""
        data.to_csv(output_path, index=False, encoding='utf-8')
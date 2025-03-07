from flask import Flask, render_template, request, jsonify
from models.classifier import SMSClassifier
import numpy as np
from pathlib import Path
import pandas as pd
import os

app = Flask(__name__)

def train_and_save_model():
    """训练并保存模型"""
    print("正在加载数据...")
    # 加载数据
    data = pd.read_csv('data/processed_data.csv')
    
    # 处理空值
    print("正在处理数据...")
    data = data.dropna(subset=['segmented_content', 'label'])
    data['segmented_content'] = data['segmented_content'].fillna('')
    data['label'] = data['label'].astype(int)
    
    print(f"处理后的数据大小: {len(data)}")
    
    # 创建分类器实例
    print("正在训练模型...")
    classifier = SMSClassifier(alpha=1.0, max_features=5000)
    
    # 训练模型
    results = classifier.train(
        X=data['segmented_content'],
        y=data['label']
    )
    
    # 保存模型
    print("正在保存模型...")
    classifier.save_model()
    print("模型训练完成并保存")
    
    return classifier

# 检查模型文件是否存在
model_dir = Path('models')
vectorizer_path = model_dir / 'vectorizer.joblib'
classifier_path = model_dir / 'classifier.joblib'

# 如果模型文件存在但可能损坏，先删除它们
if vectorizer_path.exists():
    os.remove(vectorizer_path)
if classifier_path.exists():
    os.remove(classifier_path)

# 训练新模型
classifier = train_and_save_model()

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    # 获取输入文本
    text = request.form.get('text', '')
    
    if not text:
        return jsonify({
            'error': '请输入短信内容'
        }), 400
    
    # 进行预测
    try:
        # 获取预测概率
        probas = classifier.predict_proba([text])[0]
        # 获取预测标签
        label = classifier.predict([text])[0]
        
        # 计算置信度
        confidence = float(probas[label])
        
        # 准备响应数据
        result = {
            'label': int(label),
            'confidence': confidence,
            'probabilities': {
                'normal': float(probas[0]),
                'spam': float(probas[1])
            },
            'text': text,
            'prediction': '垃圾短信' if label == 1 else '正常短信'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'预测出错：{str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
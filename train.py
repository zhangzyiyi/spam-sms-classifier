import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.spam_classifier import SpamClassifier
from pathlib import Path
import json
import sys
import traceback
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from utils.visualization import create_analysis_report

def load_fraud_keywords():
    """加载诈骗短信关键词配置"""
    with open('config/fraud_keywords.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 合并所有关键词
    keywords = []
    for category in config.values():
        keywords.extend(category['keywords'])
    
    return list(set(keywords))  # 去重

def balance_dataset(X, y):
    """平衡数据集
    
    参数:
        X: 特征矩阵
        y: 标签
    """
    # 使用SMOTE进行过采样
    sampler = SMOTE(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return X_resampled, y_resampled

def evaluate_model(y_true, y_pred, y_pred_proba):
    """评估模型性能
    
    参数:
        y_true: array-like, 真实标签
        y_pred: array-like, 预测标签
        y_pred_proba: array-like, 预测概率
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 计算详细指标
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
    }
    
    # 打印评估报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['正常短信', '诈骗短信']))
    
    print("\n详细指标:")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"特异度 (Specificity): {metrics['specificity']:.4f}")
    print(f"F1分数 (F1-Score): {metrics['f1_score']:.4f}")
    print(f"假阳性率 (FPR): {metrics['false_positive_rate']:.4f}")
    print(f"假阴性率 (FNR): {metrics['false_negative_rate']:.4f}")
    
    return metrics

def main():
    try:
        # 加载数据
        print("加载数据...")
        data_path = Path('data/processed_data.csv')
        if not data_path.exists():
            raise FileNotFoundError(f"找不到数据文件: {data_path}")
        
        data = pd.read_csv(data_path)
        print(f"原始数据大小: {len(data)}")
        
        # 检查必要的列
        required_columns = ['segmented_content', 'label', 'content']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据文件缺少必要的列: {missing_columns}")
        
        # 处理空值
        data = data.dropna(subset=['segmented_content', 'label'])
        print(f"处理空值后的数据大小: {len(data)}")
        
        # 加载诈骗短信关键词
        print("\n加载诈骗短信关键词...")
        fraud_keywords = load_fraud_keywords()
        print(f"加载了 {len(fraud_keywords)} 个关键词")
        
        # 将标签转换为二分类
        print("\n转换为二分类标签...")
        
        def is_fraud(row):
            """判断是否为诈骗短信"""
            if row['label'] == 0:
                return 0  # 正常短信
            else:
                # 检查是否包含诈骗相关关键词
                text = str(row['content']).lower()
                return 1 if any(word in text for word in fraud_keywords) else 0
        
        print("正在识别诈骗短信...")
        data['fraud_label'] = data.apply(is_fraud, axis=1)
        
        # 打印标签分布
        print("\n标签分布:")
        label_dist = data['fraud_label'].value_counts()
        for label, count in label_dist.items():
            label_name = {0: '正常短信', 1: '诈骗短信'}[label]
            print(f"{label_name}: {count} ({count/len(data)*100:.2f}%)")
        
        # 分割数据集
        print("\n分割数据集...")
        X_train, X_test, y_train, y_test = train_test_split(
            data['segmented_content'].values,
            data['fraud_label'].values,
            test_size=0.2,
            random_state=42,
            stratify=data['fraud_label']
        )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 转换特征为TF-IDF向量
        print("\n转换特征...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        
        # 平衡数据集
        print("\n平衡数据集...")
        X_train_resampled_vec, y_train_resampled = balance_dataset(
            X_train_vec, y_train
        )
        
        print("\n平衡后的标签分布:")
        for label, count in Counter(y_train_resampled).items():
            label_name = {0: '正常短信', 1: '诈骗短信'}[label]
            print(f"{label_name}: {count} ({count/len(y_train_resampled)*100:.2f}%)")
        
        # 创建并训练分类器
        print("\n训练模型...")
        classifier = SpamClassifier(skip_vectorization=True)  # 跳过向量化，因为数据已经向量化
        
        # 使用重采样后的特征矩阵训练模型
        classifier.fit(X_train_resampled_vec, y_train_resampled)
        
        # 优化阈值
        print("\n优化阈值...")
        X_test_vec = vectorizer.transform(X_test)
        results = classifier.optimize_threshold(X_test_vec, y_test)
        print("最佳阈值:", results['best_threshold'])
        print(f"最佳F1分数: {results['best_score']:.4f}")
        
        # 在测试集上评估
        print("\n模型评估:")
        y_pred = classifier.predict(X_test_vec)
        y_pred_proba = classifier.predict_proba(X_test_vec)
        
        # 评估模型性能
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # 获取特征名称和重要性分数
        feature_names = vectorizer.get_feature_names_out()
        feature_importance = classifier.get_feature_importance(feature_names)
        
        # 打印最重要的特征
        print("\n最重要的特征词:")
        for item in feature_importance[:20]:
            print(f"{item['feature']}: {item['importance']:.4f}")
        
        # 创建分析报告
        print("\n生成分析报告...")
        create_analysis_report(
            y_test,
            y_pred,
            y_pred_proba,
            ['正常短信', '诈骗短信'],
            feature_names,
            np.array([item['importance'] for item in feature_importance]),
            'analysis'
        )
        
        # 保存一些预测示例
        print("\n保存预测示例...")
        predictions = classifier.get_prediction_details(X_test_vec)
        examples = []
        for text, true_label, pred in zip(X_test[:10], y_test[:10], predictions[:10]):
            examples.append({
                'text': text,
                'true_label': {0: '正常短信', 1: '诈骗短信'}[true_label],
                'predicted': pred
            })
        
        # 保存结果
        output_dir = Path('analysis')
        output_dir.mkdir(exist_ok=True)
        
        # 保存预测示例
        with open(output_dir / 'prediction_examples.json', 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        # 保存评估指标
        with open(output_dir / 'evaluation_metrics.json', 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'model_params': classifier.get_model_params(),
                'feature_importance': feature_importance[:50]  # 保存前50个最重要的特征
            }, f, ensure_ascii=False, indent=2)
        
        # 保存模型
        print("\n保存模型...")
        classifier.save_model()
        
        print("\n训练完成！")
        print(f"\n分析报告已保存到 {output_dir} 目录")
        
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        print("\n详细错误信息:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
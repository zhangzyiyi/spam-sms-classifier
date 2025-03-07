# 基于朴素贝叶斯算法的中文诈骗短信分类系统

## 项目概述
本项目是一个基于朴素贝叶斯算法的中文诈骗短信分类系统，通过机器学习技术实现对诈骗短信的自动识别和分类，帮助用户有效防范短信诈骗。系统采用Python开发，结合Flask框架提供Web界面，使用jieba进行中文分词，scikit-learn实现机器学习算法。

## 功能特点
### 核心功能
- **朴素贝叶斯分类器**
  - 多项式朴素贝叶斯实现
  - 拉普拉斯平滑
  - 多类别诈骗短信分类
  - 参数自动优化

- **中文文本预处理**
  - 专业中文分词（jieba）
  - 停用词过滤
  - 文本清洗
  - TF-IDF特征提取

- **模型评估与优化**
  - 准确率、精确率、召回率计算
  - 混淆矩阵分析
  - 交叉验证
  - 特征选择优化

### 辅助功能
- **风险评估**：风险等级划分、概率评分
- **可视化分析**：分类结果展示、特征分析

## 技术架构
- **后端**：Python 3.8+, scikit-learn, Flask, jieba
- **前端**：Bootstrap 5, HTML5/CSS3/JavaScript

## 开发进度
- [x] 第一阶段：基础功能实现
  - [x] 数据预处理系统
  - [x] 朴素贝叶斯分类器
  - [x] Web界面框架

- [ ] 第二阶段：功能优化（进行中）
  - [x] 特征工程优化
  - [x] 参数调优系统
  - [ ] 多类别分类完善
  - [ ] 性能评估与优化

- [ ] 第三阶段：系统完善（待开始）
  - [ ] 可视化功能
  - [ ] 风险评估系统
  - [ ] 系统测试与优化

## 安装步骤
1. 克隆项目仓库
```bash
git clone https://github.com/yourusername/spam-sms-classifier.git
cd spam-sms-classifier
```

2. 创建并激活虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

4. 运行应用
```bash
python app.py
```

5. 在浏览器中访问应用
```
http://localhost:5000
```

## 使用方法
1. 在Web界面输入或粘贴需要分析的短信内容
2. 点击"分析"按钮
3. 查看分类结果和风险评估

## 项目结构
```
spam-sms-classifier/
├── app.py                    # Web应用入口
├── preprocess_data.py        # 数据预处理
├── train_model.py            # 模型训练
├── train_multiclass.py       # 多分类支持
├── optimize_model.py         # 模型优化
├── requirements.txt          # 项目依赖
├── models/                   # 模型相关
│   ├── classifier.py         # 基础分类器
│   ├── spam_classifier.py    # 诈骗短信分类器
│   ├── preprocessor.py       # 文本预处理
│   ├── feature_engineering.py# 特征工程
│   ├── parameter_tuning.py   # 参数调优
│   └── config.json          # 配置文件
├── data/                     # 数据文件
├── analysis/                 # 分析结果
├── utils/                    # 工具函数
├── static/                   # 静态资源
├── templates/                # HTML模板
└── plots/                   # 可视化图表
```

## 贡献指南
欢迎贡献代码、报告问题或提出新功能建议。请遵循以下步骤：
1. Fork项目仓库
2. 创建特性分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

## 许可证
本项目采用MIT许可证 - 详情请参阅LICENSE文件 
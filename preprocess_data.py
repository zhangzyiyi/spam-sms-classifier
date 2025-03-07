from models.preprocessor import TextPreprocessor
from pathlib import Path
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='预处理短信数据')
    parser.add_argument('--balance', choices=['undersample', 'oversample'], 
                      help='数据平衡方法: undersample(欠采样) 或 oversample(过采样)')
    args = parser.parse_args()
    
    # 创建预处理器实例
    preprocessor = TextPreprocessor()
    
    # 设置输入输出路径
    input_file = Path('data/raw/带标签短信.txt')
    output_file = Path('data/processed_data.csv')
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("正在加载数据...")
    data = preprocessor.load_data(input_file)
    print(f"加载完成，共有{len(data)}条数据")
    
    # 数据预处理
    print("正在进行数据预处理...")
    processed_data = preprocessor.preprocess(data, balance_method=args.balance)
    print("预处理完成")
    
    # 保存处理后的数据
    print("正在保存处理后的数据...")
    preprocessor.save_processed_data(processed_data, output_file)
    print(f"数据已保存到 {output_file}")

if __name__ == '__main__':
    main() 
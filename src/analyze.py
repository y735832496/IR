import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data():
    # 读取数据
    df = pd.read_csv('data/processed/processed_data.csv')
    
    # 打印基本信息
    print("\n=== 数据集基本信息 ===")
    print(f"总样本数: {len(df)}")
    print("\n=== 标签分布 ===")
    print(df['label'].value_counts())
    print("\n=== 标签分布百分比 ===")
    print(df['label'].value_counts(normalize=True) * 100)
    
    # 分析文本长度
    df['text_length'] = df['text'].str.len()
    print("\n=== 文本长度统计 ===")
    print(df['text_length'].describe())
    
    # 绘制标签分布图
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label')
    plt.title('标签分布')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/analysis/label_distribution.png')
    
    # 绘制文本长度分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_length', bins=50)
    plt.title('文本长度分布')
    plt.tight_layout()
    plt.savefig('data/analysis/text_length_distribution.png')
    
    # 分析每个标签的文本长度
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='label', y='text_length')
    plt.title('各标签文本长度分布')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/analysis/label_text_length.png')
    
    # 分析文本内容
    print("\n=== 各标签的示例文本 ===")
    for label in df['label'].unique():
        print(f"\n标签 '{label}' 的示例:")
        examples = df[df['label'] == label]['text'].head(3)
        for i, text in enumerate(examples, 1):
            print(f"{i}. {text[:100]}...")

if __name__ == "__main__":
    analyze_data() 
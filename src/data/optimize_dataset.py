import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import re
from tqdm import tqdm
import random

def clean_text(text):
    """清理文本"""
    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_duplicates(df, similarity_threshold=0.85):
    """去除重复样本"""
    print("开始去除重复样本...")
    
    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)))
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    # 计算余弦相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # 找出重复样本
    duplicates = set()
    for i in tqdm(range(len(df))):
        if i in duplicates:
            continue
        for j in range(i + 1, len(df)):
            if j in duplicates:
                continue
            if similarity_matrix[i, j] > similarity_threshold:
                duplicates.add(j)
    
    # 移除重复样本
    df_cleaned = df.drop(index=list(duplicates)).reset_index(drop=True)
    print(f"移除了 {len(duplicates)} 个重复样本")
    return df_cleaned

def generate_weather_samples():
    """生成更多样化的天气相关样本"""
    print("生成更多样化的天气相关样本...")
    
    # 天气相关模板
    templates = [
        "今天{weather}吗",
        "明天{weather}吗",
        "请问{weather}",
        "帮我看看{weather}",
        "我想知道{weather}",
        "能否告诉我{weather}",
        "麻烦{weather}",
        "{weather}怎么样",
        "{weather}如何",
        "{weather}情况"
    ]
    
    # 天气相关词汇
    weather_terms = [
        "天气", "气温", "温度", "湿度", "降雨", "下雨", "下雪", "刮风",
        "空气质量", "雾霾", "紫外线", "气压", "能见度", "风速", "风向",
        "体感温度", "穿衣指数", "洗车指数", "防晒指数", "感冒指数"
    ]
    
    # 生成新样本
    new_samples = []
    for template in templates:
        for term in weather_terms:
            text = template.format(weather=term)
            new_samples.append({
                'text': text,
                'label': 2  # 天气相关标签
            })
    
    return pd.DataFrame(new_samples)

def generate_news_samples():
    """生成更多样化的新闻相关样本"""
    print("生成更多样化的新闻相关样本...")
    
    # 新闻相关模板
    templates = [
        "有什么{news}新闻",
        "最近{news}有什么消息",
        "帮我看看{news}的新闻",
        "我想了解{news}的最新情况",
        "能否告诉我{news}的进展",
        "麻烦查询{news}的报道",
        "{news}有什么新动态",
        "{news}的最新消息",
        "{news}的新闻资讯",
        "{news}的热点话题"
    ]
    
    # 新闻相关词汇
    news_terms = [
        "国内", "国际", "财经", "科技", "体育", "娱乐", "教育", "健康",
        "社会", "文化", "军事", "政治", "环保", "交通", "房产", "汽车",
        "互联网", "人工智能", "医疗", "旅游"
    ]
    
    # 生成新样本
    new_samples = []
    for template in templates:
        for term in news_terms:
            text = template.format(news=term)
            new_samples.append({
                'text': text,
                'label': 1  # 新闻相关标签
            })
    
    return pd.DataFrame(new_samples)

def generate_time_samples():
    """生成更多样化的时间相关样本"""
    print("生成更多样化的时间相关样本...")
    
    # 时间相关模板
    templates = [
        "现在{time}了",
        "请问{time}",
        "帮我看看{time}",
        "我想知道{time}",
        "能否告诉我{time}",
        "麻烦查询{time}",
        "{time}是什么时候",
        "{time}的具体时间",
        "{time}的准确时间",
        "{time}的时间点"
    ]
    
    # 时间相关词汇
    time_terms = [
        "几点", "时间", "日期", "星期", "月份", "年份", "季节", "节日",
        "上午", "下午", "晚上", "凌晨", "中午", "傍晚", "凌晨", "午夜",
        "工作日", "周末", "假期", "开学", "放假"
    ]
    
    # 生成新样本
    new_samples = []
    for template in templates:
        for term in time_terms:
            text = template.format(time=term)
            new_samples.append({
                'text': text,
                'label': 0  # 时间相关标签
            })
    
    return pd.DataFrame(new_samples)

def ensure_category_boundaries(df):
    """确保类别边界清晰"""
    print("确保类别边界清晰...")
    
    # 定义关键词映射
    keyword_mapping = {
        0: ['时间', '日期', '星期', '几点', '多久', '分钟', '小时', '天', '周', '月', '年'],
        1: ['新闻', '报道', '事件', '发生', '最新', '消息', '资讯', '热点', '头条'],
        2: ['天气', '气温', '温度', '下雨', '下雪', '刮风', '湿度', '空气质量']
    }
    
    # 检查每个样本的标签是否与关键词匹配
    misclassified = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        label = row['label']
        
        # 检查文本中的关键词
        keywords = keyword_mapping[label]
        if not any(keyword in text for keyword in keywords):
            misclassified.append(idx)
    
    # 移除可能错误分类的样本
    df_cleaned = df.drop(index=misclassified).reset_index(drop=True)
    print(f"移除了 {len(misclassified)} 个可能错误分类的样本")
    return df_cleaned

def balance_dataset(df, target_samples=200):
    """平衡数据集"""
    print("平衡数据集...")
    
    # 获取每个类别的样本数
    label_counts = df['label'].value_counts()
    
    # 对每个类别进行采样或复制
    balanced_samples = []
    for label in range(3):
        label_df = df[df['label'] == label]
        if len(label_df) < target_samples:
            # 如果样本不足，进行复制
            samples_needed = target_samples - len(label_df)
            additional_samples = label_df.sample(n=samples_needed, replace=True)
            balanced_samples.append(pd.concat([label_df, additional_samples]))
        else:
            # 如果样本过多，进行采样
            balanced_samples.append(label_df.sample(n=target_samples))
    
    # 合并平衡后的数据集
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    print("平衡后的数据集分布：")
    print(balanced_df['label'].value_counts())
    return balanced_df

def main():
    # 读取原始数据集
    print("读取原始数据集...")
    train_df = pd.read_csv('src/data/train_clean.csv')
    val_df = pd.read_csv('src/data/val_clean.csv')
    
    # 清理文本
    print("清理文本...")
    train_df['text'] = train_df['text'].apply(clean_text)
    val_df['text'] = val_df['text'].apply(clean_text)
    
    # 去除重复样本
    train_df = remove_duplicates(train_df)
    val_df = remove_duplicates(val_df)
    
    # 生成更多样本
    weather_samples = generate_weather_samples()
    news_samples = generate_news_samples()
    time_samples = generate_time_samples()
    
    # 合并新生成的样本
    train_df = pd.concat([train_df, weather_samples, news_samples, time_samples], ignore_index=True)
    
    # 确保类别边界清晰
    train_df = ensure_category_boundaries(train_df)
    val_df = ensure_category_boundaries(val_df)
    
    # 平衡数据集
    train_df = balance_dataset(train_df, target_samples=200)
    val_df = balance_dataset(val_df, target_samples=50)
    
    # 保存优化后的数据集
    print("保存优化后的数据集...")
    train_df.to_csv('src/data/train_optimized.csv', index=False)
    val_df.to_csv('src/data/val_optimized.csv', index=False)
    
    # 打印数据集统计信息
    print("\n优化后的数据集统计：")
    print("训练集：")
    print(train_df['label'].value_counts())
    print("\n验证集：")
    print(val_df['label'].value_counts())

if __name__ == "__main__":
    main() 
import os
import pandas as pd
import random
import jieba

# 读取原始数据
raw_path = 'src/data/train.csv'
df = pd.read_csv(raw_path)

# 数据清洗：去除空行、重复、异常标签
allowed_labels = {0, 1, 2}
df = df.drop_duplicates()
df = df.dropna(subset=['text', 'label'])
df = df[df['label'].apply(lambda x: x in allowed_labels)]
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'] != '']

def word_shuffle(text):
    """词语打乱"""
    words = list(jieba.cut(text))
    if len(words) <= 2:
        return text
    # 随机打乱中间的词
    middle_words = words[1:-1]
    random.shuffle(middle_words)
    return words[0] + ''.join(middle_words) + words[-1]

def augment(row):
    """数据增强主函数"""
    text = row['text']
    label = row['label']
    aug_texts = [text]
    
    # 1. 添加前缀
    prefixes = ['请问', '帮我', '能否', '麻烦', '我想知道', '可以告诉我', '请告诉我']
    for prefix in prefixes:
        aug_texts.append(prefix + text)
    
    # 2. 词语打乱
    for _ in range(2):  # 每个文本生成2个打乱版本
        aug_texts.append(word_shuffle(text))
    
    # 3. 添加语气词
    particles = ['啊', '呢', '吧', '哦', '啦']
    for particle in particles:
        aug_texts.append(text + particle)
    
    return pd.DataFrame({'text': aug_texts, 'label': [label]*len(aug_texts)})

# 数据增强
print("开始数据增强...")
augmented = pd.concat([augment(row) for _, row in df.iterrows()], ignore_index=True)

# 随机打乱
augmented = augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# 分割数据集（8:2）
train_size = int(len(augmented) * 0.8)
train_df = augmented.iloc[:train_size]
val_df = augmented.iloc[train_size:]

# 保存到绝对路径
save_dir = os.path.join(os.path.dirname(__file__))
train_path = os.path.join(save_dir, 'train_clean.csv')
val_path = os.path.join(save_dir, 'val_clean.csv')
train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

print(f'原始样本数: {len(df)}')
print(f'增强后总样本数: {len(augmented)}')
print(f'清洗增强后训练集样本数: {len(train_df)}')
print(f'验证集样本数: {len(val_df)}')
print(f'训练集保存路径: {train_path}')
print(f'验证集保存路径: {val_path}') 
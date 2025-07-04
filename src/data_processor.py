import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

class DataProcessor:
    def __init__(self, max_vocab_size=25000):
        self.max_vocab_size = max_vocab_size
        self.vocab = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path):
        """加载数据"""
        df = pd.read_csv(file_path)
        return df
        
    def preprocess_text(self, text):
        """文本预处理"""
        # 这里可以添加更多的文本预处理步骤
        return text.lower().split()
        
    def build_vocab(self, texts):
        """构建词汇表"""
        words = []
        for text in texts:
            words.extend(self.preprocess_text(text))
            
        word_counts = Counter(words)
        self.vocab = {word: idx+2 for idx, (word, count) in 
                     enumerate(word_counts.most_common(self.max_vocab_size-2))}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        
    def encode_text(self, text):
        """将文本转换为索引序列"""
        words = self.preprocess_text(text)
        return [self.vocab.get(word, self.vocab['<unk>']) for word in words]
        
    def prepare_data(self, df, text_column, label_column):
        """准备训练数据"""
        # 构建词汇表
        self.build_vocab(df[text_column])
        
        # 编码标签
        labels = self.label_encoder.fit_transform(df[label_column])
        
        # 编码文本
        encoded_texts = [self.encode_text(text) for text in df[text_column]]
        
        return encoded_texts, labels
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与反向传播）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x) 
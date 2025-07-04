import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent len, batch size, emb dim]
        output, (hidden, cell) = self.lstm(embedded)
        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)
    def forward(self, text):
    # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))  # [sent len, batch size, emb dim]
        
        # 传递给LSTM
        output, (hidden, cell) = self.lstm(embedded)  # output: [sent len, batch size, hid dim * num directions]
        
        # 选择LSTM的最后一个时间步的隐藏状态作为文本表示
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))  # [batch size, hid dim * num directions]
        
        # 通过全连接层
        return self.fc(hidden)  # 输出: [batch size, output_dim]
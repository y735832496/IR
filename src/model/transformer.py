import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=4, d_ff=2048, dropout=0.3, num_classes=3):
        super().__init__()
        
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        # 冻结BERT参数
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),  # BERT的隐藏维度是768
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # 获取BERT的输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用[CLS]标记的输出进行分类
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 通过分类头
        logits = self.classifier(pooled_output)
        
        return logits
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import BertTokenizer
from src.model.transformer import Transformer
from src.data.dataset import create_dataloader
from src.train.trainer import Trainer

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建数据加载器（使用清洗增强后的数据）
    train_dataloader = create_dataloader(
        csv_path='src/data/train_clean.csv',
        tokenizer=tokenizer,
        batch_size=32,
        max_length=128
    )
    val_dataloader = create_dataloader(
        csv_path='src/data/val_clean.csv',
        tokenizer=tokenizer,
        batch_size=32,
        max_length=128
    )
    
    # 创建模型
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        dropout=0.1
    )
    
    # 创建训练器
    trainer = Trainer(model, tokenizer, device)
    
    # 开始训练
    trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=30,
        learning_rate=2e-4
    )
    # 预留：后续可在此处添加验证集评估

if __name__ == '__main__':
    main() 
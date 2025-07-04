import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model.transformer import Transformer
from data.dataset import create_dataloader
from train.trainer import Trainer
from torch.nn import CrossEntropyLoss
import pandas as pd


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    # 创建模型
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        num_classes=3
    )

    # 创建数据加载器
    train_dataloader = create_dataloader(
        csv_path='src/data/train_optimized.csv',
        tokenizer=tokenizer,
        batch_size=16,
        max_length=128
    )

    # 创建优化器
    optimizer = AdamW( ## 自适应学习率优化算法
        model.parameters(),
        lr=2e-5,  # BERT推荐的学习率
        eps=1e-8  #添加eps防止除零
    )

    # 创建学习率调度器
    num_training_steps = len(train_dataloader) * 50  # 50个epoch
    num_warmup_steps = num_training_steps // 10  # 10%的warmup步数
    scheduler = get_linear_schedule_with_warmup(  ##
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 创建训练器 交叉熵损失函数
    trainer = Trainer(  
        model=model,
        tokenizer=tokenizer,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # 开始训练
    trainer.train(
        train_dataloader=train_dataloader,
        num_epochs=50,
        learning_rate=2e-5
    )

    # 检查训练集标签分布
    train_df = pd.read_csv('src/data/train_optimized.csv')
    print('训练集标签分布：\n', train_df['label'].value_counts())

    # 检查验证集标签分布
    val_df = pd.read_csv('src/data/val_optimized.csv')
    print('\n验证集标签分布：\n', val_df['label'].value_counts())

    # # 检查训练集"天气相关"样本
    # train_df = pd.read_csv('src/data/train_optimized.csv')
    # print('训练集"天气相关"样本：\n', train_df[train_df['label'] == 2]['text'].tolist())

    # # 检查验证集"天气相关"样本
    # val_df = pd.read_csv('src/data/val_optimized.csv')
    # print('\n验证集"天气相关"样本：\n', val_df[val_df['label'] == 2]['text'].tolist())


if __name__ == "__main__":
    main()

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
import numpy as np
from transformers import BertTokenizer
from model.transformer import Transformer
from evaluate import ModelEvaluator
import pandas as pd

def load_test_data():
    """加载测试数据"""
    df = pd.read_csv('src/data/val_clean.csv')
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def build_label_map():
    """构建标签映射"""
    return {
        0: "时间相关",
        1: "新闻相关",
        2: "天气相关"
    }

def batch_predict(model, tokenizer, texts, batch_size=32, device='cpu'):
    """批量预测"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 对文本进行编码
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # 将数据移到设备上
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(outputs, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = Transformer(
        vocab_size=21128,  # BERT中文词表大小
        d_model=512,  # 使用较小的模型维度
        num_heads=8,  # 使用较少的注意力头
        num_layers=6,  # 使用较少的层数
        d_ff=1024,  # 使用较小的前馈网络维度
        dropout=0.1
    )
    
    # 加载模型权重
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint)  # 直接加载权重
    model = model.to(device)
    
    # 优化模型用于推理
    # print('正在优化模型...')
    # model = model.optimize_for_inference()
    
    # 模型预热
    print('模型预热中...')
    dummy_input = torch.randint(0, 21128, (1, 512)).to(device)
    dummy_attention_mask = torch.ones((1, 512), dtype=torch.long).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input, dummy_attention_mask)
    
    # 加载测试数据
    texts, true_labels = load_test_data()
    label_map = build_label_map()
    
    # 批量预测
    print('开始批量预测...')
    start_time = time.time()
    predictions, probabilities = batch_predict(model, tokenizer, texts, batch_size=32, device=device)
    end_time = time.time()
    
    # 计算准确率
    accuracy = np.mean(predictions == true_labels)
    print(f'准确率: {accuracy:.4f}')
    print(f'推理时间: {end_time - start_time:.2f}秒')
    
    # 生成评估报告
    print('生成评估报告...')
    evaluator = ModelEvaluator(model, tokenizer, device, label_map)
    evaluator.generate_evaluation_report(texts, true_labels)
    
    # 交互式预测
    print('\n开始交互式预测 (输入 q 退出):')
    while True:
        text = input('\n请输入文本: ')
        if text.lower() == 'q':
            break
            
        # 对输入文本进行编码
        encoding = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        
        # 将数据移到设备上
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 预测
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probabilities = torch.softmax(output, dim=-1)
            prediction = torch.argmax(output, dim=-1)
        
        # 输出预测结果
        predicted_label = prediction.item()
        confidence = probabilities[0][predicted_label].item()
        print(f'预测标签: {label_map[predicted_label]}')
        print(f'置信度: {confidence:.4f}')
        print('各类别概率:')
        for label, prob in enumerate(probabilities[0]):
            if label in label_map:
                print(f'{label_map[label]}: {prob.item():.4f}')

if __name__ == "__main__":
    main()
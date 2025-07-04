import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report

class Trainer:
    def __init__(self, model, tokenizer, device, optimizer=None, scheduler=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CrossEntropyLoss()
        self.best_accuracy = 0
        self.patience = 5  # 增加patience到5
        self.patience_counter = 0
        self.min_delta = 0.001  # 降低最小改善阈值
        
    def train(self, train_dataloader, num_epochs=50, learning_rate=2e-5): ## hard to modify lr
        """训练模型"""
        self.model.to(self.device)
        self.model.train()
        
        # 如果没有提供优化器，创建一个
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW( ## Adam算法，求的局部最优解
                self.model.parameters(),
                lr=learning_rate,
                eps=1e-8
            )
        
        for epoch in range(num_epochs):
            total_loss = 0
            all_preds = []
            all_labels = []
            
            # 训练一个epoch
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch in progress_bar:
                # 准备数据
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels) ## 计算损失
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪 防止梯度爆炸
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # 统计
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.2f}',
                    'accuracy': f'{accuracy_score(labels.cpu().numpy(), preds) * 100:.2f}%',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # 计算epoch的平均损失和准确率
            avg_loss = total_loss / len(train_dataloader)
            accuracy = accuracy_score(all_labels, all_preds) * 100
            
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Average Loss: {avg_loss:.4f}')
            print(f'Accuracy: {accuracy:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.2e}')
            
            # 早停检查
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'checkpoints/best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('checkpoints/best_model.pt'))
        
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)
        
        return accuracy, report
    
    def save_model(self, path):
        # 创建保存目录
        os.makedirs('checkpoints', exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer
        }, os.path.join('checkpoints', path))
    
    def load_model(self, path):
        # 加载模型
        if not os.path.exists(path):
            path = os.path.join('checkpoints', path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
    
    def predict(self, text):
        # 对输入文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 将数据移到设备上
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        padding_mask = (input_ids != self.tokenizer.pad_token_id).float().to(self.device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, padding_mask)
            predictions = torch.argmax(outputs, dim=-1)
            
        return predictions.item()

    def generate_text(self, prompt, max_length=100, temperature=1.0):
        # 对输入文本进行编码
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # 生成文本
        output_ids = self.model.generate(input_ids, max_length=max_length, temperature=temperature)
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return generated_text 
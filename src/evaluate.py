import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model, tokenizer, device, label_map: Dict[int, str]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = label_map
        
    def predict_batch(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """预测一批文本并返回概率和预测结果"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                padding_mask = (input_ids != self.tokenizer.pad_token_id).float().to(self.device)
                
                outputs = self.model(input_ids, padding_mask)
                probs = torch.softmax(outputs, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                
                predictions.append(pred.cpu().numpy()[0])
                probabilities.append(probs.cpu().numpy().squeeze(0))
        
        return np.array(predictions).flatten(), np.array(probabilities)
    
    def evaluate(self, texts: List[str], true_labels: List[int]) -> Dict:
        """计算所有评估指标"""
        predictions, probabilities = self.predict_batch(texts)
        true_labels = np.array(true_labels)  # 修正：确保为np数组
        
        # 1. 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        # 2. 分类报告
        report = classification_report(
            true_labels, 
            predictions,
            labels=list(self.label_map.keys()),
            target_names=list(self.label_map.values()),
            output_dict=True
        )
        
        # 3. ROC曲线和AUC分数（对每个类别）
        roc_curves = {}
        for i, label in enumerate(self.label_map.keys()):
            y_true_binary = (true_labels == label).astype(int)
            y_score = probabilities[:, i]
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            roc_curves[label] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        
        # 4. 精确率-召回率曲线
        pr_curves = {}
        for i, label in enumerate(self.label_map.keys()):
            y_true_binary = (true_labels == label).astype(int)
            y_score = probabilities[:, i]
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            pr_auc = auc(recall, precision)
            pr_curves[label] = {
                'precision': precision,
                'recall': recall,
                'auc': pr_auc
            }
        
        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_curves': roc_curves,
            'pr_curves': pr_curves
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_roc_curves(self, roc_curves: Dict, save_path: str = None):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        for label, curve in roc_curves.items():
            plt.plot(
                curve['fpr'], 
                curve['tpr'], 
                label=f'{self.label_map[label]} (AUC = {curve["auc"]:.2f})'
            )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_pr_curves(self, pr_curves: Dict, save_path: str = None):
        """绘制精确率-召回率曲线"""
        plt.figure(figsize=(10, 8))
        for label, curve in pr_curves.items():
            plt.plot(
                curve['recall'], 
                curve['precision'], 
                label=f'{self.label_map[label]} (AUC = {curve["auc"]:.2f})'
            )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_evaluation_report(self, texts: List[str], true_labels: List[int], 
                                 output_dir: str = 'evaluation_results'):
        """生成完整的评估报告"""
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 计算评估指标
        metrics = self.evaluate(texts, true_labels)
        
        # 保存评估指标到CSV
        metrics_df = pd.DataFrame(metrics['classification_report']).T
        metrics_df.to_csv(output_dir / 'metrics.csv')
        
        # 绘制并保存图表
        self.plot_confusion_matrix(metrics['confusion_matrix'], 
                                 output_dir / 'confusion_matrix.png')
        self.plot_roc_curves(metrics['roc_curves'], 
                           output_dir / 'roc_curves.png')
        self.plot_pr_curves(metrics['pr_curves'], 
                           output_dir / 'pr_curves.png')
        
        # 生成HTML报告
        html_report = f"""
        <html>
        <head>
            <title>模型评估报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin: 20px 0; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>模型评估报告</h1>
            
            <div class="section">
                <h2>评估指标</h2>
                {metrics_df.to_html()}
            </div>
            
            <div class="section">
                <h2>混淆矩阵</h2>
                <img src="confusion_matrix.png" alt="混淆矩阵">
            </div>
            
            <div class="section">
                <h2>ROC曲线</h2>
                <img src="roc_curves.png" alt="ROC曲线">
            </div>
            
            <div class="section">
                <h2>精确率-召回率曲线</h2>
                <img src="pr_curves.png" alt="精确率-召回率曲线">
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / 'report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"评估报告已生成在 {output_dir} 目录下") 
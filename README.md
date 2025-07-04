# AI大模型演示项目

这是一个使用PyTorch、NumPy和Pandas实现的简单文本分类模型演示项目。

## 项目结构

```
.
├── requirements.txt    # 项目依赖
├── src/
│   ├── model.py       # 模型定义
│   ├── data_processor.py  # 数据处理
│   ├── train.py       # 训练脚本
│   └── predict.py     # 预测脚本
└── README.md          # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+
- Pandas 1.3+
- scikit-learn 0.24+
- matplotlib 3.4+

## 安装

1. 克隆项目
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：
   - 将数据文件放在 `data/` 目录下
   - 数据文件应为CSV格式，包含 `text` 和 `label` 两列

2. 训练模型：
```bash
python src/train.py
```

3. 进行预测：
```bash
python src/predict.py
```

## 模型说明

本项目实现了一个基于LSTM的文本分类模型，包含以下特点：
- 双向LSTM层
- Dropout正则化
- 词嵌入层
- 全连接层

## 注意事项

- 训练前请确保有足够的数据
- 可以根据需要调整模型参数
- 建议使用GPU进行训练# IR

# Handw-rs: 手写字母识别系统 ✍️

基于Rust实现的手写字母识别系统，采用自定义K均值聚类(KMeans)算法进行字母分类。

## 功能特点 ✨

- **模型训练**: 从CSV数据集或图像目录创建手写字母识别模型
- **字母识别**: 准确识别单个手写字母图像
- **模型持久化**: 保存训练好的模型以便后续使用
- **多数据源支持**: 兼容CSV格式和图像目录两种数据输入
- **完整字母集**: 支持26个英文大写字母(A-Z)的识别

## 快速开始 🚀

### 安装 

```bash
git clone https://github.com/Wang-Yang-source/Handw-rs.git
cargo build --release
```

### 使用流程

**1. 训练模型**

```bash
# 使用CSV数据集训练
cargo run -- --train --use-csv

# 使用图像目录训练
cargo run -- --train
```

**2. 字母识别**

```bash
cargo run -- --test-image /path/to/image.png
```

## 命令行参数 💻

| 参数          | 缩写 | 功能描述           |
|--------------|------|-------------------|
| --train      | -t   | 训练新模型         |
| --train-path | -p   | 指定训练数据路径    |
| --test-image | -i   | 指定识别图像路径    |
| --model-path | -m   | 指定模型路径        |
| --use-csv    | -u   | 使用CSV格式数据     |

## 支持的数据格式 📊

- **CSV格式**: 兼容"A_Z Handwritten Data.csv"格式（第一列为标签0-25，其余列为784像素值）
- **图像目录**: 按字母分类的PNG图像（如：`data/train/A/`, `data/train/B/`等）

## 技术实现 🧮

- 基于K均值聚类算法的字符分类
- 图像特征向量提取与匹配
- JSON格式模型序列化
- 命令行界面与参数解析

## 依赖库 📦

- image: 图像处理
- ndarray: 多维数组操作
- csv: CSV解析
- clap: 命令行参数
- serde_json: 模型序列化
- anyhow: 错误处理
- rand: 随机数生成

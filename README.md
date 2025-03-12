<<<<<<< HEAD
# Handw-rs
=======
# 手写字母识别系统

这是一个使用Rust语言实现的手写字母识别系统，使用自定义K均值聚类(KMeans)算法进行字母分类。

## 功能

- 训练手写字母识别模型
- 识别单个手写字母图像
- 支持A-Z的26个英文字母
- 支持从CSV文件加载训练数据

## 安装

确保您已安装Rust和Cargo。然后克隆此仓库并构建项目：

```bash
git clone https://github.com/yourusername/handwritten_recognition.git
cd handwritten_recognition
cargo build --release
```

## 数据集准备

### 使用CSV数据集

本项目默认使用Kaggle的"A_Z Handwritten
Data.csv"数据集。该数据集包含手写字母A-Z的样本，每个样本由28x28像素的图像表示，存储在CSV文件中。

CSV文件的格式如下：

- 第一列：标签（0-25对应A-Z）
- 其余列：784个像素值（28x28图像的像素值）

### 使用图像目录

您也可以使用按以下结构组织的图像目录：

```
data/
  train/
    A/
      image1.png
      image2.png
      ...
    B/
      image1.png
      image2.png
      ...
    ...
    Z/
      ...
```

每个字母目录下应包含该字母的多个手写样本图像。

## 使用方法

### 训练模型

从CSV文件训练：

```bash
cargo run -- --train --use-csv
```

从图像目录训练：

```bash
cargo run -- --train
```

默认情况下，训练数据将从`data/A_Z Handwritten Data.csv`（CSV模式）或`data/train`（图像目录模式）加载，模型将保存到`model.json`。

您可以指定自定义路径：

```bash
cargo run -- --train --use-csv --train-path /path/to/data.csv --model-path /path/to/save/model.json
```

或使用短选项：

```bash
cargo run -- -t -u -p /path/to/data.csv -m /path/to/save/model.json
```

### 识别图像

```bash
cargo run -- --test-image /path/to/image.png
```

或使用短选项：

```bash
cargo run -- -i /path/to/image.png
```

默认情况下，将从`model.json`加载模型。您可以指定自定义模型路径：

```bash
cargo run -- --test-image /path/to/image.png --model-path /path/to/model.json
```

或使用短选项：

```bash
cargo run -- -i /path/to/image.png -m /path/to/model.json
```

## 命令行参数说明

| 长选项       | 短选项 | 描述              |
| ------------ | ------ | ----------------- |
| --train      | -t     | 训练模型          |
| --train-path | -p     | 训练数据集路径    |
| --test-image | -i     | 测试图像路径      |
| --model-path | -m     | 模型保存/加载路径 |
| --use-csv    | -u     | 使用CSV数据集     |

## 示例

1. 从CSV训练模型：
   ```bash
   cargo run -- --train --use-csv
   ```
   或
   ```bash
   cargo run -- -t -u
   ```

2. 识别图像：
   ```bash
   cargo run -- --test-image test_image.png
   ```
   或
   ```bash
   cargo run -- -i test_image.png
   ```

## 算法说明

本项目使用自定义实现的K均值聚类(KMeans)算法进行字母识别。算法流程如下：

1. 将每个手写字母图像转换为特征向量（28x28=784维）
2. 使用KMeans算法将特征向量聚类为26个簇（对应26个字母）：
   - 随机初始化26个聚类中心
   - 将每个样本分配到最近的聚类中心
   - 重新计算每个聚类的中心点
   - 重复上述步骤直到收敛或达到最大迭代次数
3. 为每个簇分配最常见的字母标签（通过统计每个簇中样本的真实标签）
4. 识别时，将测试图像转换为特征向量，找到最近的簇，返回该簇对应的字母

## 依赖项

- image: 图像处理
- ndarray: 多维数组操作（带serde支持，用于模型序列化）
- rand: 随机数生成（用于KMeans初始化）
- csv: CSV文件处理
- clap: 命令行参数解析
- serde_json: 模型序列化和反序列化
- anyhow: 错误处理

## 许可证

MIT
>>>>>>> 3b7427c (rs-kmeans手写字母识别)

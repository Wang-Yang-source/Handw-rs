use anyhow::{Context, Result};
use clap::Parser;
use csv::ReaderBuilder;
use image::{DynamicImage, GrayImage};
use ndarray::{Array1, Array2, ArrayView1};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

// 导入可视化模块
mod visualize;

/// 手写字母识别系统
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// 训练数据集路径
    #[clap(short = 'p', long, default_value = "data/A_Z_Handwritten_Data.csv")]
    train_path: PathBuf,

    /// 测试图像路径
    #[clap(short = 'i', long)]
    test_image: Option<PathBuf>,

    /// 训练模型
    #[clap(short, long)]
    train: bool,

    /// 保存模型路径
    #[clap(short, long, default_value = "model.json")]
    model_path: PathBuf,

    /// 使用CSV数据集
    #[clap(short, long)]
    use_csv: bool,

    /// 可视化聚类结果
    #[clap(long)]
    visualize_clusters: bool,

    /// 可视化输出目录
    #[clap(long, default_value = "visualization")]
    output_dir: PathBuf,

    /// 测试目录
    #[clap(long)]
    test_dir: Option<PathBuf>,

    /// 创建混淆矩阵
    #[clap(long)]
    confusion_matrix: bool,
}

/// 自定义的K均值聚类模型
#[derive(Debug, Serialize, Deserialize)]
struct KMeansModel {
    centroids: Array2<f32>,
}

impl KMeansModel {
    /// 创建新的KMeans模型
    fn new(k: usize, data: &Array2<f32>) -> Self {
        // 随机初始化聚类中心
        let n_samples = data.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());

        let centroids = Array2::from_shape_fn((k, data.ncols()), |(i, j)| {
            if i < indices.len() {
                data[[indices[i], j]]
            } else {
                // 如果k大于样本数，则重复使用样本
                data[[indices[i % indices.len()], j]]
            }
        });

        Self { centroids }
    }

    /// 训练模型
    fn fit(&mut self, data: &Array2<f32>, max_iterations: usize, tolerance: f32) {
        let mut old_centroids = self.centroids.clone();

        for _ in 0..max_iterations {
            // 分配样本到最近的聚类
            let labels = self.predict(data);

            // 更新聚类中心
            for i in 0..self.centroids.nrows() {
                let cluster_points: Vec<usize> = labels
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &label)| if label == i { Some(idx) } else { None })
                    .collect();

                if !cluster_points.is_empty() {
                    for j in 0..self.centroids.ncols() {
                        let sum: f32 = cluster_points.iter().map(|&idx| data[[idx, j]]).sum();
                        self.centroids[[i, j]] = sum / cluster_points.len() as f32;
                    }
                }
            }

            // 检查收敛性
            let diff = (&self.centroids - &old_centroids).mapv(|x| x.abs()).sum();

            if diff < tolerance {
                break;
            }

            old_centroids = self.centroids.clone();
        }
    }

    /// 预测样本所属的聚类
    fn predict(&self, data: &Array2<f32>) -> Vec<usize> {
        let mut labels = Vec::with_capacity(data.nrows());

        for sample in data.outer_iter() {
            let mut min_dist = f32::INFINITY;
            let mut closest_centroid = 0;

            for (i, centroid) in self.centroids.outer_iter().enumerate() {
                let dist = euclidean_distance(&sample, &centroid);
                if dist < min_dist {
                    min_dist = dist;
                    closest_centroid = i;
                }
            }

            labels.push(closest_centroid);
        }

        labels
    }
}

/// 计算欧几里得距离
fn euclidean_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// 将图像转换为特征向量
fn image_to_features(img: &GrayImage) -> Array1<f32> {
    let (width, height) = img.dimensions();
    let mut features = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            // 将灰度值归一化到 0-1 范围
            features.push(1.0 - (pixel[0] as f32 / 255.0));
        }
    }

    Array1::from(features)
}

/// 预处理图像：调整大小、转换为灰度
fn preprocess_image(img: DynamicImage) -> GrayImage {
    // 调整图像大小为 28x28 像素（与 MNIST 数据集相同）
    let resized = img.resize_exact(28, 28, image::imageops::FilterType::Lanczos3);
    // 转换为灰度图像
    resized.to_luma8()
}

/// 从CSV文件加载训练数据
fn load_training_data_from_csv(file_path: &Path) -> Result<(Array2<f32>, Array1<u8>)> {
    println!("从CSV加载数据: {}", file_path.display());

    // 打开CSV文件
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut features = Vec::new();
    let mut labels = Vec::new();

    // 读取CSV记录
    for result in rdr.records() {
        let record = result?;

        // 第一列是标签（0-25对应A-Z）
        if let Some(label_str) = record.get(0) {
            if let Ok(label) = label_str.parse::<u8>() {
                // 将CSV中的标签（0-25）转换为我们的标签（0-25）
                labels.push(label);

                // 剩余的列是特征（像素值）
                let mut row_features = Vec::with_capacity(record.len() - 1);
                for i in 1..record.len() {
                    if let Some(value_str) = record.get(i) {
                        if let Ok(value) = value_str.parse::<f32>() {
                            // 归一化像素值（0-255）到0-1范围
                            row_features.push(value / 255.0);
                        }
                    }
                }

                features.push(Array1::from(row_features));
            }
        }
    }

    // 将特征向量和标签转换为 ndarray 格式
    let features_array = Array2::from_shape_vec(
        (features.len(), features[0].len()),
        features.into_iter().flatten().collect(),
    )?;

    let labels_array = Array1::from(labels);

    Ok((features_array, labels_array))
}

/// 从目录加载训练数据
fn load_training_data_from_dir(dir_path: &Path) -> Result<(Array2<f32>, Array1<u8>)> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    // 遍历目录中的每个子目录（每个字母一个子目录）
    for entry in std::fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // 子目录名称应该是字母（A-Z）
            let label = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|s| s.chars().next())
                .context("无法获取标签")?;

            // 将字母转换为数字标签（A=0, B=1, ...）
            let label_num = if label.is_ascii_alphabetic() {
                (label.to_ascii_uppercase() as u8) - b'A'
            } else {
                return Err(anyhow::anyhow!("无效的标签: {}", label));
            };

            // 加载该字母的所有图像
            for img_entry in std::fs::read_dir(&path)? {
                let img_entry = img_entry?;
                let img_path = img_entry.path();

                if img_path.extension().and_then(|ext| ext.to_str()) == Some("png")
                    || img_path.extension().and_then(|ext| ext.to_str()) == Some("jpg")
                {
                    let img = image::open(&img_path)?;
                    let processed_img = preprocess_image(img);
                    let feature_vector = image_to_features(&processed_img);

                    features.push(feature_vector);
                    labels.push(label_num);
                }
            }
        }
    }

    // 将特征向量和标签转换为 ndarray 格式
    let features_array = Array2::from_shape_vec(
        (features.len(), features[0].len()),
        features.into_iter().flatten().collect(),
    )?;

    let labels_array = Array1::from(labels);

    Ok((features_array, labels_array))
}

/// 训练模型
fn train_model(
    features: &Array2<f32>,
    labels: &Array1<u8>,
) -> Result<(KMeansModel, HashMap<usize, u8>)> {
    // 创建并训练KMeans模型
    let mut model = KMeansModel::new(26, features);
    model.fit(features, 100, 1e-5);

    // 预测聚类
    let predictions = model.predict(features);

    // 创建聚类到标签的映射
    let mut cluster_to_label_count = HashMap::new();

    // 对于每个聚类，统计每个标签的出现次数
    for (i, &cluster) in predictions.iter().enumerate() {
        let label = labels[i];
        let entry = cluster_to_label_count
            .entry(cluster)
            .or_insert_with(HashMap::new);
        *entry.entry(label).or_insert(0) += 1;
    }

    // 为每个聚类分配最常见的标签
    let mut cluster_label_map = HashMap::new();
    for (cluster, label_counts) in cluster_to_label_count {
        let mut max_count = 0;
        let mut most_common_label = 0;

        for (label, count) in label_counts {
            if count > max_count {
                max_count = count;
                most_common_label = label;
            }
        }

        cluster_label_map.insert(cluster, most_common_label);
    }

    Ok((model, cluster_label_map))
}

/// 保存模型到文件
fn save_model(model: &(KMeansModel, HashMap<usize, u8>), path: &Path) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer(file, model)?;
    Ok(())
}

/// 从文件加载模型
fn load_model(path: &Path) -> Result<(KMeansModel, HashMap<usize, u8>)> {
    let file = File::open(path)?;
    let model = serde_json::from_reader(file)?;
    Ok(model)
}

/// 识别单个图像
fn recognize_image(model: &(KMeansModel, HashMap<usize, u8>), img_path: &Path) -> Result<char> {
    let (kmeans, cluster_map) = model;

    let img = image::open(img_path)?;
    let processed_img = preprocess_image(img);
    let features = image_to_features(&processed_img);

    // 创建一个只有一个样本的数据集
    let sample = Array2::from_shape_vec((1, features.len()), features.to_vec())?;

    // 预测聚类
    let predictions = kmeans.predict(&sample);
    let cluster = predictions[0];

    // 从聚类映射中获取标签
    let label_num = cluster_map.get(&cluster).copied().unwrap_or(0);

    // 将数字标签转换回字母（0=A, 1=B, ...）
    let letter = (label_num + b'A') as char;

    Ok(letter)
}

/// 主函数
fn main() -> Result<()> {
    let args = Args::parse();

    if args.train {
        // 加载训练数据
        let (features, labels) = if args.use_csv {
            load_training_data_from_csv(&args.train_path)?
        } else {
            load_training_data_from_dir(&args.train_path)?
        };

        println!("已加载{}个训练样本", features.nrows());

        // 训练模型
        println!("正在训练模型...");
        let model = train_model(&features, &labels)?;

        // 保存模型
        println!("正在保存模型到 {}...", args.model_path.display());
        save_model(&model, &args.model_path)?;
        println!("模型已保存!");

        // 可视化聚类结果
        if args.visualize_clusters {
            println!("正在可视化聚类结果...");
            visualize::visualize_clusters(&model.0, &model.1, &args.output_dir)?;
        }

        // 创建混淆矩阵
        if args.confusion_matrix {
            println!("正在创建混淆矩阵...");
            let output_path = args.output_dir.join("confusion_matrix.png");
            visualize::create_confusion_matrix(
                &model.0,
                &model.1,
                &features,
                &labels,
                &output_path,
            )?;
        }
    }

    // 如果提供了测试目录，则测试模型
    if let Some(test_dir) = args.test_dir {
        // 加载模型
        println!("正在加载模型...");
        let model = load_model(&args.model_path)?;

        // 测试模型
        println!("正在测试模型...");
        let output_dir = args.output_dir.join("test_results");
        visualize::test_model(&model.0, &model.1, &test_dir, &output_dir)?;
    }

    if let Some(test_image) = args.test_image {
        // 加载模型
        println!("正在加载模型...");
        let model = load_model(&args.model_path)?;

        // 识别图像
        println!("正在识别图像...");
        let letter = recognize_image(&model, &test_image)?;
        println!("识别结果: {}", letter);

        // 可视化测试图像
        let output_path = args.output_dir.join(format!(
            "prediction_{}",
            test_image.file_name().unwrap().to_string_lossy()
        ));
        visualize::visualize_test_image(&model.0, &model.1, &test_image, &output_path)?;
    }

    Ok(())
}

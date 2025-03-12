use anyhow::{Context, Result};
use image::{DynamicImage, GrayImage, ImageBuffer, Rgb};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::{image_to_features, preprocess_image, KMeansModel};

/// 可视化聚类结果
pub fn visualize_clusters(
    model: &KMeansModel,
    cluster_map: &HashMap<usize, u8>,
    output_dir: &Path,
) -> Result<()> {
    // 创建输出目录
    fs::create_dir_all(output_dir)?;

    // 为每个聚类中心创建可视化图像
    for (i, centroid) in model.centroids.outer_iter().enumerate() {
        // 将聚类中心转换为图像
        let img = visualize_feature_vector(
            &centroid
                .to_owned()
                .into_shape_with_order((28, 28))?
                .as_standard_layout()
                .mapv(|x| x * 255.0),
        )?;

        // 获取该聚类对应的字母标签
        let label = cluster_map.get(&i).copied().unwrap_or(0);
        let letter = (label + b'A') as char;

        // 保存图像
        let output_path = output_dir.join(format!("cluster_{}_label_{}.png", i, letter));
        img.save(&output_path)?;

        println!(
            "已保存聚类 {} (标签: {}) 的可视化图像到 {}",
            i,
            letter,
            output_path.display()
        );
    }

    Ok(())
}

/// 将特征向量可视化为图像
fn visualize_feature_vector(features: &Array2<f32>) -> Result<DynamicImage> {
    let height = features.nrows();
    let width = features.ncols();

    // 创建灰度图像
    let mut img = GrayImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let value = features[[y, x]].min(255.0).max(0.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Luma([value]));
        }
    }

    Ok(DynamicImage::ImageLuma8(img))
}

/// 创建混淆矩阵可视化
pub fn create_confusion_matrix(
    model: &KMeansModel,
    cluster_map: &HashMap<usize, u8>,
    test_data: &Array2<f32>,
    test_labels: &Array1<u8>,
    output_path: &Path,
) -> Result<()> {
    // 初始化26x26的混淆矩阵（A-Z）
    let mut confusion_matrix = Array2::<u32>::zeros((26, 26));

    // 预测测试数据
    let predictions = model.predict(test_data);

    // 填充混淆矩阵
    for (i, &cluster) in predictions.iter().enumerate() {
        let true_label = test_labels[i] as usize;
        let predicted_label = *cluster_map.get(&cluster).unwrap_or(&0) as usize;

        confusion_matrix[[true_label, predicted_label]] += 1;
    }

    // 可视化混淆矩阵
    let img = visualize_confusion_matrix(&confusion_matrix)?;
    img.save(output_path)?;

    println!("已保存混淆矩阵到 {}", output_path.display());

    // 计算准确率
    let mut correct = 0;
    let total = test_labels.len();

    for (i, &cluster) in predictions.iter().enumerate() {
        let true_label = test_labels[i];
        let predicted_label = *cluster_map.get(&cluster).unwrap_or(&0);

        if true_label == predicted_label {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / total as f32;
    println!(
        "测试准确率: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct,
        total
    );

    Ok(())
}

/// 将混淆矩阵可视化为图像
fn visualize_confusion_matrix(matrix: &Array2<u32>) -> Result<DynamicImage> {
    let cell_size = 20; // 每个单元格的像素大小
    let width = (matrix.ncols() + 1) * cell_size;
    let height = (matrix.nrows() + 1) * cell_size;

    // 创建RGB图像
    let mut img = ImageBuffer::new(width as u32, height as u32);

    // 填充背景为白色
    for pixel in img.pixels_mut() {
        *pixel = Rgb([255, 255, 255]);
    }

    // 找到矩阵中的最大值，用于归一化颜色
    let max_value = *matrix.iter().max().unwrap_or(&1);

    // 绘制混淆矩阵
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let value = matrix[[i, j]];
            let intensity = (255.0 * (1.0 - (value as f32 / max_value as f32))).round() as u8;

            // 绘制单元格
            for y in 0..cell_size {
                for x in 0..cell_size {
                    let px = (j + 1) * cell_size + x;
                    let py = (i + 1) * cell_size + y;
                    img.put_pixel(px as u32, py as u32, Rgb([intensity, intensity, 255]));
                }
            }

            // 如果值不为0，在单元格中绘制数字
            if value > 0 {
                // 这里简化处理，实际上应该使用字体渲染
                let color = if intensity < 128 { 255 } else { 0 };
                let center_x = (j + 1) * cell_size + cell_size / 2;
                let center_y = (i + 1) * cell_size + cell_size / 2;

                // 在中心点绘制一个小点表示数值
                img.put_pixel(center_x as u32, center_y as u32, Rgb([color, color, color]));
            }
        }
    }

    // 绘制标签
    for i in 0..26 {
        let _letter = (b'A' + i as u8) as char;

        // 绘制行标签（真实标签）
        let y = (i + 1) * cell_size + cell_size / 2;
        let x = cell_size / 2;

        // 简化处理，实际上应该使用字体渲染
        for dx in -2i32..3 {
            for dy in -2i32..3 {
                if dx.abs() + dy.abs() <= 2 {
                    img.put_pixel(
                        (x as i32 + dx) as u32,
                        (y as i32 + dy) as u32,
                        Rgb([0, 0, 0]),
                    );
                }
            }
        }

        // 绘制列标签（预测标签）
        let x = (i + 1) * cell_size + cell_size / 2;
        let y = cell_size / 2;

        for dx in -2i32..3 {
            for dy in -2i32..3 {
                if dx.abs() + dy.abs() <= 2 {
                    img.put_pixel(
                        (x as i32 + dx) as u32,
                        (y as i32 + dy) as u32,
                        Rgb([0, 0, 0]),
                    );
                }
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(img))
}

/// 测试模型在测试集上的性能
pub fn test_model(
    model: &KMeansModel,
    cluster_map: &HashMap<usize, u8>,
    test_dir: &Path,
    output_dir: &Path,
) -> Result<()> {
    // 创建输出目录
    fs::create_dir_all(output_dir)?;

    let mut correct = 0;
    let mut total = 0;
    let mut results = Vec::new();

    // 遍历测试目录中的每个子目录（每个字母一个子目录）
    for entry in fs::read_dir(test_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // 子目录名称应该是字母（A-Z）
            let true_label = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|s| s.chars().next())
                .context("无法获取标签")?;

            // 将字母转换为数字标签（A=0, B=1, ...）
            let true_label_num = if true_label.is_ascii_alphabetic() {
                (true_label.to_ascii_uppercase() as u8) - b'A'
            } else {
                return Err(anyhow::anyhow!("无效的标签: {}", true_label));
            };

            // 测试该字母的所有图像
            for img_entry in fs::read_dir(&path)? {
                let img_entry = img_entry?;
                let img_path = img_entry.path();

                if img_path.extension().and_then(|ext| ext.to_str()) == Some("png")
                    || img_path.extension().and_then(|ext| ext.to_str()) == Some("jpg")
                {
                    let img = image::open(&img_path)?;
                    let processed_img = preprocess_image(img);
                    let features = image_to_features(&processed_img);

                    // 创建一个只有一个样本的数据集
                    let sample = Array2::from_shape_vec((1, features.len()), features.to_vec())?;

                    // 预测聚类
                    let predictions = model.predict(&sample);
                    let cluster = predictions[0];

                    // 从聚类映射中获取标签
                    let predicted_label_num = cluster_map.get(&cluster).copied().unwrap_or(0);
                    let predicted_letter = (predicted_label_num + b'A') as char;

                    // 检查预测是否正确
                    let is_correct = predicted_label_num == true_label_num;
                    if is_correct {
                        correct += 1;
                    }
                    total += 1;

                    // 保存结果
                    results.push((img_path.clone(), true_label, predicted_letter, is_correct));

                    // 如果预测错误，保存图像以供分析
                    if !is_correct {
                        let output_filename = format!(
                            "error_true_{}_pred_{}_{}",
                            true_label,
                            predicted_letter,
                            img_path.file_name().unwrap().to_string_lossy()
                        );
                        let output_path = output_dir.join(output_filename);
                        fs::copy(&img_path, &output_path)?;
                    }
                }
            }
        }
    }

    // 计算并打印准确率
    let accuracy = correct as f32 / total as f32;
    println!(
        "测试准确率: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct,
        total
    );

    // 保存详细结果到CSV文件
    let mut wtr = csv::Writer::from_path(output_dir.join("test_results.csv"))?;
    wtr.write_record(&["图像路径", "真实标签", "预测标签", "是否正确"])?;

    for (path, true_label, pred_label, is_correct) in results {
        wtr.write_record(&[
            path.to_string_lossy().to_string(),
            true_label.to_string(),
            pred_label.to_string(),
            if is_correct { "是" } else { "否" }.to_string(),
        ])?;
    }

    wtr.flush()?;

    Ok(())
}

/// 可视化测试图像及其预测结果
pub fn visualize_test_image(
    model: &KMeansModel,
    cluster_map: &HashMap<usize, u8>,
    img_path: &Path,
    output_path: &Path,
) -> Result<()> {
    let img = image::open(img_path)?;
    let processed_img = preprocess_image(img.clone());
    let features = image_to_features(&processed_img);

    // 创建一个只有一个样本的数据集
    let sample = Array2::from_shape_vec((1, features.len()), features.to_vec())?;

    // 预测聚类
    let predictions = model.predict(&sample);
    let cluster = predictions[0];

    // 从聚类映射中获取标签
    let predicted_label_num = cluster_map.get(&cluster).copied().unwrap_or(0);
    let predicted_letter = (predicted_label_num + b'A') as char;

    // 创建可视化图像
    let output_img = img.to_rgb8();

    // 在图像上添加预测结果
    // 这里简化处理，实际上应该使用字体渲染
    // 注释掉未使用的变量，或者使用 _ 前缀来表明有意不使用
    let _text = format!("预测: {}", predicted_letter);

    // 保存带有预测结果的图像
    output_img.save(output_path)?;

    println!("已保存预测结果图像到 {}", output_path.display());
    println!("预测结果: {}", predicted_letter);

    Ok(())
}

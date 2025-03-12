// src/visualization.rs

use ndarray::{Array2, Array1};
use plotters::prelude::*;

/// 可视化聚类结果
pub fn visualize_clusters(data: &Array2<f32>, labels: &Array1<usize>, centroids: &Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("clusters.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("KMeans Clustering Results", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..1f32, 0f32..1f32)?;

    chart.configure_mesh().draw()?;

    // 绘制数据点
    for (i, point) in data.outer_iter().enumerate() {
        let label = labels[i];
        let color = match label {
            0 => RED,
            1 => GREEN,
            2 => BLUE,
            3 => CYAN,
            4 => MAGENTA,
            5 => YELLOW,
            _ => BLACK,
        };
        chart.draw_series(PointSeries::of_element(
            point.iter().cloned().take(2).collect::<Vec<f32>>(),
            5,
            color,
            &|c, s, st| {
                Circle::new((c[0], c[1]), s, st)
            },
        ))?;
    }

    // 绘制聚类中心
    for centroid in centroids.outer_iter() {
        chart.draw_series(std::iter::once(Circle::new((centroid[0], centroid[1]), 10, RED.filled())))?;
    }

    Ok(())
}

/// 显示字母分布
pub fn display_letter_distribution(labels: &Array1<usize>) -> Result<(), Box<dyn std::error::Error>> {
    let mut distribution = vec![0; 26]; // A-Z

    for &label in labels {
        if label < 26 {
            distribution[label] += 1;
        }
    }

    let root = BitMapBackend::new("letter_distribution.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Letter Distribution", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..26, 0..*distribution.iter().max().unwrap())?;

    chart.configure_mesh().draw()?;

    chart.draw_series(distribution.iter().enumerate().map(|(x, &y)| {
        Rectangle::new([(x, 0), (x + 1, y)], BLUE.filled())
    }))?;

    Ok(())
}
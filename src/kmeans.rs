// This file contains the implementation of the KMeans clustering algorithm, including the KMeansModel struct, methods for fitting the model, predicting clusters, and calculating Euclidean distances.

use ndarray::{Array1, Array2, ArrayView1};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// 自定义的K均值聚类模型
#[derive(Debug)]
pub struct KMeansModel {
    centroids: Array2<f32>,
}

impl KMeansModel {
    /// 创建新的KMeans模型
    pub fn new(k: usize, data: &Array2<f32>) -> Self {
        let n_samples = data.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());

        let centroids = Array2::from_shape_fn((k, data.ncols()), |(i, j)| {
            if i < indices.len() {
                data[[indices[i], j]]
            } else {
                data[[indices[i % indices.len()], j]]
            }
        });

        Self { centroids }
    }

    /// 训练模型
    pub fn fit(&mut self, data: &Array2<f32>, max_iterations: usize, tolerance: f32) {
        let mut old_centroids = self.centroids.clone();

        for _ in 0..max_iterations {
            let labels = self.predict(data);

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

            let diff = (&self.centroids - &old_centroids).mapv(|x| x.abs()).sum();

            if diff < tolerance {
                break;
            }

            old_centroids = self.centroids.clone();
        }
    }

    /// 预测样本所属的聚类
    pub fn predict(&self, data: &Array2<f32>) -> Vec<usize> {
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

use ndarray::{Array2, Array1};
use serde_json::from_reader;
use std::fs::File;
use std::path::Path;
use crate::visualization::{plot_clusters, plot_letter_distribution};

fn main() {
    // Load the KMeans model and the training data
    let model_path = Path::new("model.json");
    let file = File::open(model_path).expect("Unable to open model file");
    let (model, cluster_map): (KMeansModel, HashMap<usize, u8>) = from_reader(file).expect("Unable to read model");

    let data_path = Path::new("data/A_Z_Handwritten_Data.csv");
    let (features, labels) = load_training_data_from_csv(data_path).expect("Unable to load training data");

    // Predict clusters for the training data
    let predictions = model.predict(&features);

    // Visualize the clusters
    plot_clusters(&features, &predictions, &cluster_map);
    plot_letter_distribution(&predictions, &labels);
}
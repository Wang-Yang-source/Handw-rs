// This file contains unit tests for the visualization functions, ensuring that the visual outputs are generated correctly and meet expected criteria.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualization::{plot_clusters, display_letter_distribution};
    use ndarray::Array2;

    #[test]
    fn test_plot_clusters() {
        // Create a mock dataset for testing
        let data = Array2::from_shape_vec((5, 2), vec![
            1.0, 2.0,
            1.5, 1.8,
            5.0, 8.0,
            8.0, 8.0,
            1.0, 0.6,
        ]).unwrap();

        // Call the plot_clusters function
        let result = plot_clusters(&data);

        // Assert that the result is as expected (e.g., check if a plot was created)
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_letter_distribution() {
        // Create a mock distribution for testing
        let distribution = vec![
            ('A', 10),
            ('B', 15),
            ('C', 5),
        ];

        // Call the display_letter_distribution function
        let result = display_letter_distribution(&distribution);

        // Assert that the result is as expected (e.g., check if a plot was created)
        assert!(result.is_ok());
    }
}
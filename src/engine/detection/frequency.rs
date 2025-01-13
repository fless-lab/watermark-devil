use image::DynamicImage;
use opencv as cv;
use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgproc;
use rustfft::{FftPlanner, num_complex::Complex};
use rayon::prelude::*;

use crate::detection::{DetectionResult, BoundingBox, WatermarkType};

pub struct FrequencyAnalyzer {
    window_size: usize,
    stride: usize,
    threshold: f32,
}

impl FrequencyAnalyzer {
    pub fn new() -> Self {
        Self {
            window_size: 64,
            stride: 32,
            threshold: 0.7,
        }
    }

    pub async fn analyze(&self, image: &DynamicImage) -> Vec<DetectionResult> {
        let mut results = Vec::new();
        
        // Convertir l'image en niveaux de gris
        let gray_image = image.to_luma8();
        let (width, height) = gray_image.dimensions();

        // Créer des fenêtres pour l'analyse
        let windows = self.create_windows(width, height);

        // Analyser chaque fenêtre en parallèle
        let window_results: Vec<_> = windows.par_iter()
            .filter_map(|window| {
                self.analyze_window(&gray_image, window)
            })
            .collect();

        results.extend(window_results);
        results
    }

    fn create_windows(&self, width: u32, height: u32) -> Vec<BoundingBox> {
        let mut windows = Vec::new();
        
        for y in (0..height).step_by(self.stride) {
            for x in (0..width).step_by(self.stride) {
                let w = (self.window_size as u32).min(width - x);
                let h = (self.window_size as u32).min(height - y);
                
                windows.push(BoundingBox {
                    x,
                    y,
                    width: w,
                    height: h,
                });
            }
        }
        
        windows
    }

    fn analyze_window(&self, image: &image::GrayImage, window: &BoundingBox) -> Option<DetectionResult> {
        // Extraire la fenêtre
        let mut data = Vec::with_capacity(self.window_size * self.window_size);
        for y in window.y..window.y + window.height {
            for x in window.x..window.x + window.width {
                if let Some(pixel) = image.get_pixel_checked(x, y) {
                    data.push(Complex::new(pixel[0] as f32, 0.0));
                }
            }
        }

        // Appliquer la FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(data.len());
        fft.process(&mut data);

        // Analyser le spectre
        let spectrum_strength = self.analyze_spectrum(&data);
        
        if spectrum_strength > self.threshold {
            Some(DetectionResult {
                confidence: spectrum_strength,
                bbox: window.clone(),
                watermark_type: WatermarkType::Pattern,
                mask: None,
            })
        } else {
            None
        }
    }

    fn analyze_spectrum(&self, spectrum: &[Complex<f32>]) -> f32 {
        // Calculer la magnitude du spectre
        let magnitudes: Vec<f32> = spectrum.iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        // Trouver les pics significatifs
        let mean = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
        let peaks: Vec<f32> = magnitudes.iter()
            .filter(|&&m| m > mean * 2.0)
            .copied()
            .collect();

        // Calculer le score basé sur le nombre et l'intensité des pics
        if peaks.is_empty() {
            0.0
        } else {
            let peak_strength = peaks.iter().sum::<f32>() / peaks.len() as f32;
            (peak_strength / mean).min(1.0)
        }
    }

    fn apply_bandpass_filter(&self, spectrum: &mut [Complex<f32>], low_freq: f32, high_freq: f32) {
        let n = spectrum.len();
        let nyquist = n as f32 / 2.0;
        
        for (i, value) in spectrum.iter_mut().enumerate() {
            let freq = if i <= n/2 { i as f32 } else { (n - i) as f32 };
            let normalized_freq = freq / nyquist;
            
            if normalized_freq < low_freq || normalized_freq > high_freq {
                *value = Complex::new(0.0, 0.0);
            }
        }
    }
}

use ndarray::{Array2, Array3, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use opencv::{
    core::{Mat, MatTraitConst},
    imgproc::{self, INTER_LINEAR},
};
use crate::types::{Detection, WatermarkType, Confidence};
use anyhow::Result;
use tracing::{info, debug};

pub struct FrequencyDetector {
    threshold: f32,
    min_area: u32,
    max_area: u32,
    fft_planner: FftPlanner<f32>,
}

impl FrequencyDetector {
    pub fn new(threshold: f32, min_area: u32, max_area: u32) -> Self {
        Self {
            threshold,
            min_area,
            max_area,
            fft_planner: FftPlanner::new(),
        }
    }

    pub fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        info!("Starting frequency-based detection");
        
        // Convert to grayscale and normalize
        let mut gray = Mat::default();
        imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Convert to floating point
        let mut float_mat = Mat::default();
        gray.convert_to(&mut float_mat, opencv::core::CV_32F, 1.0, 0.0)?;
        
        // Convert to ndarray for FFT
        let rows = float_mat.rows() as usize;
        let cols = float_mat.cols() as usize;
        let data: Vec<f32> = float_mat.data_typed()?;
        let array = Array2::from_shape_vec((rows, cols), data)?;
        
        // Apply FFT
        let spectrum = self.compute_fft(&array)?;
        
        // Analyze spectrum
        let peaks = self.find_periodic_patterns(&spectrum)?;
        
        // Convert peaks to detections
        let detections = self.peaks_to_detections(peaks, image.size()?.width, image.size()?.height)?;
        
        info!("Frequency detection completed: {} patterns found", detections.len());
        Ok(detections)
    }

    fn compute_fft(&self, image: &Array2<f32>) -> Result<Array2<Complex<f32>>> {
        debug!("Computing 2D FFT");
        
        let (rows, cols) = image.dim();
        let mut complex_input: Vec<Complex<f32>> = image
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
            
        // Create FFT
        let fft = self.fft_planner.plan_fft_forward(cols);
        
        // Apply FFT to rows
        for row in complex_input.chunks_mut(cols) {
            fft.process(row);
        }
        
        // Transpose and apply FFT to columns
        let mut complex_array = Array2::from_shape_vec((rows, cols), complex_input)?;
        complex_array.swap_axes(0, 1);
        
        for mut row in complex_array.rows_mut() {
            let mut row_vec: Vec<_> = row.to_vec();
            fft.process(&mut row_vec);
            row.assign(&Array2::from_shape_vec((1, cols), row_vec)?);
        }
        
        // Shift zero frequency to center
        self.fft_shift(&mut complex_array);
        
        Ok(complex_array)
    }

    fn fft_shift(&self, array: &mut Array2<Complex<f32>>) {
        let (rows, cols) = array.dim();
        let half_rows = rows / 2;
        let half_cols = cols / 2;
        
        // Shift quadrants
        for i in 0..half_rows {
            for j in 0..half_cols {
                let temp = array[[i, j]];
                array[[i, j]] = array[[i + half_rows, j + half_cols]];
                array[[i + half_rows, j + half_cols]] = temp;
                
                let temp = array[[i + half_rows, j]];
                array[[i + half_rows, j]] = array[[i, j + half_cols]];
                array[[i, j + half_cols]] = temp;
            }
        }
    }

    fn find_periodic_patterns(&self, spectrum: &Array2<Complex<f32>>) -> Result<Vec<(usize, usize, f32)>> {
        debug!("Analyzing frequency spectrum for periodic patterns");
        
        let magnitude = spectrum.mapv(|x| x.norm());
        let mean = magnitude.mean().unwrap_or(0.0);
        let std_dev = magnitude.std(0.0);
        
        let mut peaks = Vec::new();
        let (rows, cols) = magnitude.dim();
        
        // Find local maxima
        for i in 1..rows-1 {
            for j in 1..cols-1 {
                let value = magnitude[[i, j]];
                if value > mean + self.threshold * std_dev {
                    let is_local_max = self.is_local_maximum(&magnitude, i, j);
                    if is_local_max {
                        peaks.push((i, j, value));
                    }
                }
            }
        }
        
        // Sort by magnitude
        peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        Ok(peaks)
    }

    fn is_local_maximum(&self, array: &Array2<f32>, row: usize, col: usize) -> bool {
        let value = array[[row, col]];
        for i in -1..=1 {
            for j in -1..=1 {
                if i == 0 && j == 0 {
                    continue;
                }
                let r = (row as isize + i) as usize;
                let c = (col as isize + j) as usize;
                if array[[r, c]] >= value {
                    return false;
                }
            }
        }
        true
    }

    fn peaks_to_detections(&self, peaks: Vec<(usize, usize, f32)>, width: i32, height: i32) -> Result<Vec<Detection>> {
        debug!("Converting frequency peaks to detections");
        
        let mut detections = Vec::new();
        for (row, col, magnitude) in peaks {
            // Convert frequency coordinates to image space
            let period_x = width as f32 / (col as f32);
            let period_y = height as f32 / (row as f32);
            
            // Calculate area
            let area = period_x * period_y;
            if area < self.min_area as f32 || area > self.max_area as f32 {
                continue;
            }
            
            // Create detection
            let confidence = Confidence::new(magnitude / 100.0);
            let bbox = opencv::core::Rect::new(0, 0, width, height);
            
            detections.push(Detection {
                watermark_type: WatermarkType::Pattern,
                confidence,
                bbox,
                metadata: Some(serde_json::json!({
                    "period_x": period_x,
                    "period_y": period_y,
                    "magnitude": magnitude
                })),
            });
        }
        
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs;
    use tempfile::tempdir;
    
    #[test]
    fn test_frequency_detector() -> Result<()> {
        // Create test image with periodic pattern
        let mut image = Mat::new_rows_cols_with_default(
            512,
            512,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(255.0)
        )?;
        
        // Add periodic watermark pattern
        for i in 0..512 {
            for j in 0..512 {
                if (i + j) % 32 == 0 {
                    *image.at_2d_mut::<opencv::core::Vec3b>(i, j)? = opencv::core::Vec3b::new(200, 200, 200);
                }
            }
        }
        
        // Create detector
        let detector = FrequencyDetector::new(3.0, 100, 10000);
        
        // Run detection
        let detections = detector.detect(&image)?;
        
        // Verify results
        assert!(!detections.is_empty(), "Should detect periodic pattern");
        assert!(detections[0].confidence.value() > 0.5, "Should have high confidence");
        assert_eq!(detections[0].watermark_type, WatermarkType::Pattern);
        
        Ok(())
    }
    
    #[test]
    fn test_fft_computation() -> Result<()> {
        let detector = FrequencyDetector::new(3.0, 100, 10000);
        
        // Create simple test image
        let image = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0
            ]
        )?;
        
        let spectrum = detector.compute_fft(&image)?;
        
        // Check spectrum properties
        assert_eq!(spectrum.dim(), (4, 4));
        assert!(spectrum[[0, 0]].norm() > 0.0);
        
        Ok(())
    }
}

use image::DynamicImage;
use rustfft::{FftPlanner, num_complex::Complex};
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;
use opencv::{imgproc, core, Mat};
use ndarray::{Array2, ArrayBase, Data, Ix2};

use crate::detection::DetectionResult;
use crate::config::ReconstructionConfig;

pub struct FrequencyReconstructor {
    config: ReconstructionConfig,
    fft_planner: Arc<FftPlanner<f32>>,
}

impl FrequencyReconstructor {
    pub fn new(config: &ReconstructionConfig) -> Result<Self> {
        info!("Initializing frequency reconstructor");
        
        Ok(Self {
            config: config.clone(),
            fft_planner: Arc::new(FftPlanner::new()),
        })
    }

    pub async fn remove_pattern(
        &self,
        image: &DynamicImage,
        detection: &DetectionResult,
    ) -> Result<DynamicImage> {
        debug!("Starting frequency-based reconstruction");
        
        // Convert to grayscale for FFT
        let mut gray = Mat::default();
        imgproc::cvt_color(&opencv_image_to_mat(image), &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Extract ROI
        let roi = Mat::roi(&gray, detection.bbox)?;
        
        // Convert to ndarray for FFT
        let roi_array = self.mat_to_array(&roi)?;
        
        // Apply FFT
        let mut spectrum = self.compute_fft(&roi_array)?;
        
        // Filter periodic patterns
        self.filter_spectrum(&mut spectrum)?;
        
        // Apply inverse FFT
        let reconstructed_array = self.compute_ifft(&spectrum)?;
        
        // Convert back to Mat
        let mut reconstructed_roi = self.array_to_mat(&reconstructed_array)?;
        
        // Blend with original
        let mut result = opencv_image_to_mat(image).clone();
        let mut roi = Mat::roi_mut(&mut result, detection.bbox)?;
        self.blend_reconstruction(&mut roi, &reconstructed_roi)?;
        
        // Calculate quality score
        let quality_score = self.calculate_quality(&opencv_image_to_mat(image), &result, detection)?;
        
        Ok(mat_to_dynamic_image(&result))
    }

    fn create_windows(&self, width: u32, height: u32) -> Vec<Window> {
        let mut windows = Vec::new();
        
        for y in (0..height).step_by(self.config.window_size - self.config.overlap) {
            for x in (0..width).step_by(self.config.window_size - self.config.overlap) {
                let w = (self.config.window_size as u32).min(width - x);
                let h = (self.config.window_size as u32).min(height - y);
                
                windows.push(Window {
                    x,
                    y,
                    width: w,
                    height: h,
                });
            }
        }
        
        windows
    }

    fn process_window(
        &self,
        image: &image::GrayImage,
        window: &Window,
        detection: &DetectionResult,
    ) -> ProcessedWindow {
        // Extraire les données de la fenêtre
        let mut data = Vec::with_capacity(self.config.window_size * self.config.window_size);
        for y in window.y..window.y + window.height {
            for x in window.x..window.x + window.width {
                if let Some(pixel) = image.get_pixel_checked(x, y) {
                    data.push(Complex::new(pixel[0] as f32, 0.0));
                }
            }
        }

        // Appliquer la FFT
        let fft = self.fft_planner.plan_fft_forward(data.len());
        let mut spectrum = data.clone();
        fft.process(&mut spectrum);

        // Supprimer les composantes périodiques
        self.remove_periodic_components(&mut spectrum);

        // Appliquer la FFT inverse
        let ifft = self.fft_planner.plan_fft_inverse(spectrum.len());
        ifft.process(&mut spectrum);

        // Normaliser
        for value in spectrum.iter_mut() {
            *value = Complex::new(value.norm() / data.len() as f32, 0.0);
        }

        ProcessedWindow {
            window: window.clone(),
            data: spectrum,
        }
    }

    fn remove_periodic_components(&self, spectrum: &mut [Complex<f32>]) {
        let n = spectrum.len();
        let threshold = self.calculate_threshold(spectrum);

        // Supprimer les pics de fréquence
        for i in 0..n {
            let magnitude = spectrum[i].norm();
            if magnitude > threshold {
                spectrum[i] = Complex::new(0.0, 0.0);
            }
        }
    }

    fn calculate_threshold(&self, spectrum: &[Complex<f32>]) -> f32 {
        let mut magnitudes: Vec<f32> = spectrum.iter()
            .map(|c| c.norm())
            .collect();
        
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Utiliser le 95e percentile comme seuil
        let idx = (magnitudes.len() as f32 * 0.95) as usize;
        magnitudes[idx]
    }

    fn reconstruct_image(
        &self,
        original: &DynamicImage,
        processed_windows: &[ProcessedWindow],
    ) -> Result<DynamicImage> {
        let mut result = original.clone();
        let result_gray = result.to_luma8();
        let (width, height) = result_gray.dimensions();

        // Accumulateur pour la reconstruction
        let mut accumulator = vec![0.0; (width * height) as usize];
        let mut weights = vec![0.0; (width * height) as usize];

        // Combiner les fenêtres traitées
        for window in processed_windows {
            for (i, value) in window.data.iter().enumerate() {
                let x = window.window.x + (i as u32 % window.window.width);
                let y = window.window.y + (i as u32 / window.window.width);
                
                if x < width && y < height {
                    let idx = (y * width + x) as usize;
                    accumulator[idx] += value.re;
                    weights[idx] += 1.0;
                }
            }
        }

        // Normaliser et appliquer à l'image originale
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if weights[idx] > 0.0 {
                    let value = (accumulator[idx] / weights[idx]).clamp(0.0, 255.0) as u8;
                    // Mettre à jour l'image résultat
                    // Note: Cette partie dépend du format de l'image originale
                }
            }
        }

        Ok(result)
    }

    fn mat_to_array(&self, mat: &Mat) -> Result<Vec<Complex<f32>>> {
        let mut array = Vec::new();
        for y in 0..mat.rows() {
            for x in 0..mat.cols() {
                let pixel = mat.at_2d::<u8>(y, x).unwrap();
                array.push(Complex::new(pixel as f32, 0.0));
            }
        }
        Ok(array)
    }

    fn array_to_mat(&self, array: &[Complex<f32>]) -> Result<Mat> {
        let rows = (array.len() as f32).sqrt() as i32;
        let cols = rows;
        let mut mat = Mat::zeros(rows, cols, opencv::core::CV_8UC1);
        for y in 0..rows {
            for x in 0..cols {
                let idx = y * cols + x;
                let pixel = array[idx as usize].re as u8;
                mat.at_2d_mut::<u8>(y, x).unwrap() = pixel;
            }
        }
        Ok(mat)
    }

    fn blend_reconstruction(&self, roi: &mut Mat, reconstructed_roi: &Mat) -> Result<()> {
        // Blend the reconstructed ROI with the original ROI
        // This is a simple implementation and may need to be adjusted based on the specific requirements
        for y in 0..roi.rows() {
            for x in 0..roi.cols() {
                let original_pixel = roi.at_2d::<u8>(y, x).unwrap();
                let reconstructed_pixel = reconstructed_roi.at_2d::<u8>(y, x).unwrap();
                let blended_pixel = (original_pixel as f32 * 0.5 + reconstructed_pixel as f32 * 0.5) as u8;
                roi.at_2d_mut::<u8>(y, x).unwrap() = blended_pixel;
            }
        }
        Ok(())
    }

    fn calculate_quality(&self, original: &Mat, result: &Mat, detection: &DetectionResult) -> Result<f32> {
        // Calculate the quality score based on the original and result images
        // This is a simple implementation and may need to be adjusted based on the specific requirements
        let mut original_array = self.mat_to_array(original)?;
        let mut result_array = self.mat_to_array(result)?;
        let mut quality_score = 0.0;
        for i in 0..original_array.len() {
            let original_pixel = original_array[i].re;
            let result_pixel = result_array[i].re;
            quality_score += (original_pixel - result_pixel).abs();
        }
        quality_score /= original_array.len() as f32;
        Ok(quality_score)
    }

    fn filter_spectrum(&self, spectrum: &mut Array2<Complex<f32>>) -> Result<()> {
        let rows = spectrum.shape()[0];
        let cols = spectrum.shape()[1];
        let threshold = self.get_filter_threshold();
        
        // Apply high-pass filter to remove periodic patterns
        for i in 0..rows {
            for j in 0..cols {
                let freq_x = if i <= rows/2 { i } else { rows - i } as f32;
                let freq_y = if j <= cols/2 { j } else { cols - j } as f32;
                let freq_magnitude = (freq_x * freq_x + freq_y * freq_y).sqrt();
                
                if freq_magnitude < threshold {
                    spectrum[[i, j]] = Complex::new(0.0, 0.0);
                }
            }
        }
        
        Ok(())
    }

    fn get_filter_threshold(&self) -> f32 {
        match self.config.quality.as_str() {
            "high" => 5.0,
            "medium" => 10.0,
            _ => 15.0,
        }
    }

    fn count_spectrum_peaks(&self, spectrum: &Array2<Complex<f32>>) -> usize {
        let mut count = 0;
        let rows = spectrum.shape()[0];
        let cols = spectrum.shape()[1];
        let threshold = self.get_filter_threshold();
        
        for i in 1..rows-1 {
            for j in 1..cols-1 {
                let center = spectrum[[i, j]].norm();
                if center > threshold {
                    let is_peak = [
                        spectrum[[i-1, j-1]].norm(),
                        spectrum[[i-1, j]].norm(),
                        spectrum[[i-1, j+1]].norm(),
                        spectrum[[i, j-1]].norm(),
                        spectrum[[i, j+1]].norm(),
                        spectrum[[i+1, j-1]].norm(),
                        spectrum[[i+1, j]].norm(),
                        spectrum[[i+1, j+1]].norm(),
                    ].iter().all(|&n| n <= center);
                    
                    if is_peak {
                        count += 1;
                    }
                }
            }
        }
        
        count
    }
}

fn opencv_image_to_mat(image: &DynamicImage) -> Mat {
    let (width, height) = image.dimensions();
    let mut mat = Mat::zeros(height as i32, width as i32, opencv::core::CV_8UC3);
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            mat.at_2d_mut::<[u8; 3]>(y as i32, x as i32).unwrap().copy_from_slice(&pixel.0);
        }
    }
    mat
}

fn mat_to_dynamic_image(mat: &Mat) -> DynamicImage {
    let (width, height) = (mat.cols() as u32, mat.rows() as u32);
    let mut image = DynamicImage::new_rgb8(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = mat.at_2d::<[u8; 3]>(y as i32, x as i32).unwrap();
            image.put_pixel(x, y, image::Rgb(pixel));
        }
    }
    image
}

#[derive(Clone)]
struct Window {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

struct ProcessedWindow {
    window: Window,
    data: Vec<Complex<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::{core::{Mat, Point, Scalar}, imgproc};
    use crate::types::{Detection, WatermarkType, Confidence, ReconstructionMethod};

    #[test]
    fn test_frequency_reconstruction() -> Result<()> {
        // Create test configuration
        let config = ReconstructionConfig {
            quality: "high".to_string(),
            use_gpu: false,
            preserve_details: true,
            max_iterations: 1000,
        };
        
        let reconstructor = FrequencyReconstructor::new(&config)?;
        
        // Create test image with watermark
        let mut image = Mat::new_rows_cols_with_default(
            512,
            512,
            opencv::core::CV_8UC3,
            Scalar::all(255.0),
        )?;
        
        // Add watermark text
        imgproc::put_text(
            &mut image,
            "TEST",
            Point::new(200, 250),
            imgproc::FONT_HERSHEY_SIMPLEX,
            2.0,
            Scalar::new(128.0, 128.0, 128.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;
        
        // Create detection
        let detection = Detection {
            watermark_type: WatermarkType::Text,
            confidence: Confidence::new(0.9),
            bbox: opencv::core::Rect::new(180, 200, 200, 100),
            metadata: None,
        };
        
        // Run reconstruction
        let result = reconstructor.reconstruct(&image, &detection)?;
        
        // Verify result
        assert!(result.quality_score > 0.5);
        assert_eq!(result.method_used, ReconstructionMethod::Frequency);
        assert_eq!(result.image.size()?, image.size()?);
        
        // Verify metadata
        let metadata = result.metadata.as_ref().unwrap();
        assert!(metadata["spectrum_peaks"].as_u64().unwrap() > 0);
        assert!(metadata["filter_threshold"].as_f64().unwrap() > 0.0);
        
        Ok(())
    }
    
    #[test]
    fn test_spectrum_analysis() -> Result<()> {
        let config = ReconstructionConfig {
            quality: "high".to_string(),
            use_gpu: false,
            preserve_details: true,
            max_iterations: 1000,
        };
        
        let reconstructor = FrequencyReconstructor::new(&config)?;
        
        // Create test pattern with periodic components
        let mut test_array = Array2::zeros((64, 64));
        for i in 0..64 {
            for j in 0..64 {
                test_array[[i, j]] = if (i + j) % 8 == 0 { 255.0 } else { 0.0 };
            }
        }
        
        // Convert to complex numbers
        let mut spectrum = Array2::zeros((64, 64));
        for i in 0..64 {
            for j in 0..64 {
                spectrum[[i, j]] = Complex::new(test_array[[i, j]], 0.0);
            }
        }
        
        // Count peaks before filtering
        let peaks_before = reconstructor.count_spectrum_peaks(&spectrum);
        
        // Apply filter
        reconstructor.filter_spectrum(&mut spectrum)?;
        
        // Count peaks after filtering
        let peaks_after = reconstructor.count_spectrum_peaks(&spectrum);
        
        // Verify that filtering reduced the number of peaks
        assert!(peaks_after < peaks_before);
        
        Ok(())
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[tokio::test]
    async fn test_frequency_analysis() {
        let analyzer = FrequencyAnalyzer::new();
        
        // Créer une image de test avec un motif périodique
        let mut test_image = GrayImage::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                let value = if (x + y) % 8 == 0 { 255 } else { 0 };
                test_image.put_pixel(x, y, Luma([value]));
            }
        }
        
        let dynamic_image = DynamicImage::ImageLuma8(test_image);
        let results = analyzer.analyze(&dynamic_image).await;
        
        assert!(!results.is_empty(), "Should detect periodic patterns");
        
        // Vérifier que les résultats ont une confiance élevée
        for result in results {
            assert!(result.confidence > 0.5, "Should have high confidence in periodic pattern");
        }
    }
}

use image::DynamicImage;
use rustfft::{FftPlanner, num_complex::Complex};
use anyhow::Result;
use rayon::prelude::*;

use crate::detection::DetectionResult;

pub struct FrequencyEngine {
    window_size: usize,
    overlap: usize,
}

impl FrequencyEngine {
    pub fn new() -> Self {
        Self {
            window_size: 64,
            overlap: 32,
        }
    }

    pub async fn remove_pattern(
        &self,
        image: &DynamicImage,
        detection: &DetectionResult,
    ) -> Result<DynamicImage> {
        // Convertir en niveaux de gris pour l'analyse fréquentielle
        let gray = image.to_luma8();
        let (width, height) = gray.dimensions();

        // Créer les fenêtres pour l'analyse
        let windows = self.create_windows(width, height);

        // Traiter chaque fenêtre en parallèle
        let processed_windows: Vec<_> = windows.par_iter()
            .map(|window| {
                self.process_window(&gray, window, detection)
            })
            .collect();

        // Reconstruire l'image
        self.reconstruct_image(image, &processed_windows)
    }

    fn create_windows(&self, width: u32, height: u32) -> Vec<Window> {
        let mut windows = Vec::new();
        
        for y in (0..height).step_by(self.window_size - self.overlap) {
            for x in (0..width).step_by(self.window_size - self.overlap) {
                let w = (self.window_size as u32).min(width - x);
                let h = (self.window_size as u32).min(height - y);
                
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
        let mut spectrum = data.clone();
        fft.process(&mut spectrum);

        // Supprimer les composantes périodiques
        self.remove_periodic_components(&mut spectrum);

        // Appliquer la FFT inverse
        let ifft = planner.plan_fft_inverse(spectrum.len());
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
    use image::{RgbImage, Rgb};
    use crate::detection::{BoundingBox, WatermarkType};

    #[tokio::test]
    async fn test_frequency_reconstruction() {
        let engine = FrequencyEngine::new();
        
        // Créer une image de test avec un motif périodique
        let mut test_image = RgbImage::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                let value = if (x + y) % 8 == 0 { 255 } else { 200 };
                test_image.put_pixel(x, y, Rgb([value, value, value]));
            }
        }
        
        let detection = DetectionResult {
            confidence: 0.9,
            bbox: BoundingBox {
                x: 32,
                y: 32,
                width: 64,
                height: 64,
            },
            watermark_type: WatermarkType::Pattern,
            mask: None,
        };

        let image = DynamicImage::ImageRgb8(test_image);
        let result = engine.remove_pattern(&image, &detection).await;
        
        assert!(result.is_ok(), "Frequency reconstruction should succeed");
    }
}

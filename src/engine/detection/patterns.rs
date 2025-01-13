use image::DynamicImage;
use opencv as cv;
use opencv::prelude::*;
use opencv::core::{Mat, Size, Point};
use opencv::imgproc;
use crate::detection::{DetectionResult, BoundingBox, WatermarkType};

pub struct PatternDetector {
    template_size: Size,
    correlation_threshold: f64,
}

impl PatternDetector {
    pub fn new() -> Self {
        Self {
            template_size: Size::new(8, 8),
            correlation_threshold: 0.8,
        }
    }

    pub async fn detect(&self, image: &DynamicImage) -> Vec<DetectionResult> {
        let mut results = Vec::new();
        
        // Convertir l'image en Mat OpenCV
        let img = self.dynamic_image_to_mat(image);
        if img.is_err() {
            return results;
        }
        let img = img.unwrap();

        // 1. Détection de motifs répétitifs
        if let Some(pattern_results) = self.detect_repetitive_patterns(&img) {
            results.extend(pattern_results);
        }

        // 2. Détection de régions semi-transparentes
        if let Some(transparency_results) = self.detect_transparent_regions(&img) {
            results.extend(transparency_results);
        }

        results
    }

    fn detect_repetitive_patterns(&self, img: &Mat) -> Option<Vec<DetectionResult>> {
        let mut results = Vec::new();
        
        // Convertir en niveaux de gris
        let mut gray = Mat::default();
        imgproc::cvt_color(img, &mut gray, imgproc::COLOR_BGR2GRAY, 0).ok()?;

        // Appliquer FFT
        let mut complex_mat = Mat::default();
        let mut planes = vec![Mat::default(), Mat::default()];
        gray.convert_to(&mut planes[0], cv::core::CV_32F, 1.0, 0.0).ok()?;
        cv::core::dft(&planes[0], &mut complex_mat, cv::core::DFT_COMPLEX_OUTPUT, 0).ok()?;

        // Analyser le spectre de fréquence
        let mut magnitude = Mat::default();
        cv::core::magnitude(&complex_mat, &Mat::default(), &mut magnitude).ok()?;

        // Détecter les pics dans le spectre
        let mut peaks = Vec::new();
        self.find_frequency_peaks(&magnitude, &mut peaks);

        // Convertir les pics en détections
        for peak in peaks {
            results.push(DetectionResult {
                confidence: 0.85,
                bbox: BoundingBox {
                    x: peak.x as u32,
                    y: peak.y as u32,
                    width: self.template_size.width as u32,
                    height: self.template_size.height as u32,
                },
                watermark_type: WatermarkType::Pattern,
                mask: None,
            });
        }

        Some(results)
    }

    fn detect_transparent_regions(&self, img: &Mat) -> Option<Vec<DetectionResult>> {
        let mut results = Vec::new();
        
        // Convertir en RGBA
        let mut rgba = Mat::default();
        imgproc::cvt_color(img, &mut rgba, imgproc::COLOR_BGR2BGRA, 0).ok()?;

        // Extraire le canal alpha
        let mut alpha = Mat::default();
        let mut channels = Vec::new();
        cv::core::split(&rgba, &mut channels).ok()?;
        alpha = channels[3].clone();

        // Détecter les régions semi-transparentes
        let mut binary = Mat::default();
        cv::imgproc::threshold(&alpha, &mut binary, 0.0, 255.0, cv::imgproc::THRESH_BINARY_INV).ok()?;

        // Trouver les contours
        let mut contours = Vec::new();
        let mut hierarchy = Mat::default();
        imgproc::find_contours(
            &binary,
            &mut contours,
            &mut hierarchy,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        ).ok()?;

        // Convertir les contours en détections
        for contour in contours {
            let rect = imgproc::bounding_rect(&contour).ok()?;
            results.push(DetectionResult {
                confidence: 0.9,
                bbox: BoundingBox {
                    x: rect.x as u32,
                    y: rect.y as u32,
                    width: rect.width as u32,
                    height: rect.height as u32,
                },
                watermark_type: WatermarkType::Transparent,
                mask: None,
            });
        }

        Some(results)
    }

    fn find_frequency_peaks(&self, magnitude: &Mat, peaks: &mut Vec<Point>) {
        // Implémentation de la détection des pics de fréquence
        // Utilise une fenêtre glissante pour trouver les maxima locaux
        let kernel_size = 3;
        let mut local_max = Mat::default();
        
        cv::imgproc::dilate(
            magnitude,
            &mut local_max,
            &Mat::default(),
            Point::new(-1, -1),
            1,
            cv::core::BORDER_CONSTANT,
            cv::core::Scalar::default(),
        ).unwrap_or(());

        // Trouver les points où magnitude == local_max
        for y in kernel_size..(magnitude.rows() - kernel_size) {
            for x in kernel_size..(magnitude.cols() - kernel_size) {
                let mag_val = magnitude.at_2d::<f32>(y, x).unwrap_or(&0.0);
                let max_val = local_max.at_2d::<f32>(y, x).unwrap_or(&0.0);
                
                if (mag_val - max_val).abs() < 1e-6 && *mag_val > self.correlation_threshold {
                    peaks.push(Point::new(x, y));
                }
            }
        }
    }

    fn dynamic_image_to_mat(&self, image: &DynamicImage) -> cv::Result<Mat> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        unsafe {
            Mat::new_rows_cols_with_data(
                height as i32,
                width as i32,
                cv::core::CV_8UC3,
                rgb.as_ptr() as *mut _,
                cv::core::Mat_AUTO_STEP,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbImage, Rgb};

    #[tokio::test]
    async fn test_pattern_detection() {
        let detector = PatternDetector::new();
        
        // Créer une image de test avec un motif répétitif
        let mut test_image = RgbImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                if (x + y) % 10 == 0 {
                    test_image.put_pixel(x, y, Rgb([255, 255, 255]));
                }
            }
        }
        
        let dynamic_image = DynamicImage::ImageRgb8(test_image);
        let results = detector.detect(&dynamic_image).await;
        
        assert!(!results.is_empty(), "Should detect patterns in test image");
    }
}

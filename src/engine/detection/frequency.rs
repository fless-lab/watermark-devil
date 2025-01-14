use std::sync::Arc;
use opencv::{
    core::{Mat, Point, Scalar, CV_32F, CV_8U},
    imgproc,
    dnn,
    prelude::*,
};
use anyhow::{Result, Context};
use tracing::{debug, instrument};

use crate::{
    types::{Detection, WatermarkType, BoundingBox, Confidence},
    config::DetectionConfig,
    error::EngineError,
};

use super::models::WatermarkDetector;

/// Détecteur basé sur l'analyse des fréquences
#[derive(Clone)]
pub struct FrequencyDetector {
    config: Arc<DetectionConfig>,
}

impl FrequencyDetector {
    pub fn new(config: &DetectionConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config.clone()),
        })
    }
    
    fn compute_dft(&self, image: &Mat) -> Result<Mat> {
        let mut gray = Mat::default();
        imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Conversion en float32 pour la DFT
        let mut float_img = Mat::default();
        gray.convert_to(&mut float_img, CV_32F, 1.0, 0.0)?;
        
        // Optimisation de la taille pour la DFT
        let optimal_rows = opencv::core::get_optimal_dft_size(float_img.rows())?;
        let optimal_cols = opencv::core::get_optimal_dft_size(float_img.cols())?;
        
        let mut padded = Mat::default();
        opencv::core::copy_make_border(
            &float_img,
            &mut padded,
            0,
            optimal_rows - float_img.rows(),
            0,
            optimal_cols - float_img.cols(),
            opencv::core::BORDER_CONSTANT,
            Scalar::all(0.0)
        )?;
        
        // Calcul de la DFT
        let mut planes = vec![padded, Mat::zeros(optimal_rows, optimal_cols, CV_32F)?.to_mat()?];
        let mut complex = Mat::default();
        opencv::core::merge(&planes, &mut complex)?;
        
        let mut dft = Mat::default();
        opencv::core::dft(&complex, &mut dft, opencv::core::DFT_COMPLEX_OUTPUT, 0)?;
        
        Ok(dft)
    }
    
    fn analyze_spectrum(&self, dft: &Mat) -> Result<Vec<Point>> {
        let mut planes = Vec::new();
        opencv::core::split(dft, &mut planes)?;
        
        let mut magnitude = Mat::default();
        opencv::core::magnitude(&planes[0], &planes[1], &mut magnitude)?;
        
        // Log scale
        opencv::core::add(&magnitude, &Scalar::all(1.0), &mut magnitude, &Mat::default(), -1)?;
        opencv::core::log(&magnitude, &mut magnitude)?;
        
        // Normalisation
        opencv::core::normalize(
            &magnitude,
            &mut magnitude,
            0.0,
            1.0,
            opencv::core::NORM_MINMAX,
            -1,
            &Mat::default()
        )?;
        
        // Détection des pics
        let mut peaks = Vec::new();
        let kernel_size = 15;
        let half = kernel_size / 2;
        
        for y in half..magnitude.rows() - half {
            for x in half..magnitude.cols() - half {
                let window = magnitude.roi(opencv::core::Rect::new(
                    x - half,
                    y - half,
                    kernel_size,
                    kernel_size
                ))?;
                
                let center_value = *magnitude.at_2d::<f32>(y, x)?;
                let mut is_peak = true;
                
                for wy in 0..kernel_size {
                    for wx in 0..kernel_size {
                        if wy == half && wx == half {
                            continue;
                        }
                        
                        if *window.at_2d::<f32>(wy, wx)? >= center_value {
                            is_peak = false;
                            break;
                        }
                    }
                    
                    if !is_peak {
                        break;
                    }
                }
                
                if is_peak && center_value > 0.7 {
                    peaks.push(Point::new(x, y));
                }
            }
        }
        
        Ok(peaks)
    }
    
    fn detect_periodic_patterns(&self, peaks: &[Point], image_size: (i32, i32)) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();
        let (width, height) = image_size;
        
        // Analyse des pics par paires pour trouver des motifs périodiques
        for i in 0..peaks.len() {
            for j in i + 1..peaks.len() {
                let p1 = peaks[i];
                let p2 = peaks[j];
                
                // Calcul de la période
                let dx = (p2.x - p1.x).abs();
                let dy = (p2.y - p1.y).abs();
                
                // Si la période est significative
                if dx > 10 || dy > 10 {
                    let confidence = self.estimate_pattern_confidence(dx, dy, width, height);
                    
                    if confidence > self.config.confidence_threshold {
                        // Création d'une détection pour le motif périodique
                        detections.push(Detection {
                            watermark_type: WatermarkType::Pattern,
                            bbox: BoundingBox {
                                x: 0,
                                y: 0,
                                width: width as u32,
                                height: height as u32,
                            },
                            confidence: Confidence::new(confidence),
                            mask: None,
                        });
                        
                        break; // Une seule détection par pic principal
                    }
                }
            }
        }
        
        Ok(detections)
    }
    
    fn estimate_pattern_confidence(&self, dx: i32, dy: i32, width: i32, height: i32) -> f32 {
        // Heuristique simple basée sur la taille relative du motif
        let pattern_size = (dx * dx + dy * dy) as f32;
        let image_size = (width * width + height * height) as f32;
        let relative_size = pattern_size / image_size;
        
        // Les motifs trop petits ou trop grands sont moins probables d'être des filigranes
        if relative_size < 0.001 || relative_size > 0.5 {
            return 0.0;
        }
        
        // Score basé sur la régularité du motif
        let regularity = if dx == 0 || dy == 0 {
            1.0 // Motif parfaitement aligné
        } else {
            0.8 // Motif diagonal
        };
        
        regularity * (1.0 - relative_size * 2.0) // Plus le motif est petit, plus il est probable
    }
}

impl WatermarkDetector for FrequencyDetector {
    #[instrument(skip(self, image))]
    fn detect(&self, image: &Mat) -> Result<Vec<Detection>> {
        debug!("Starting frequency-based detection");
        
        // Calcul de la transformée de Fourier
        let dft = self.compute_dft(image)
            .context("Failed to compute DFT")?;
            
        // Analyse du spectre pour trouver les pics
        let peaks = self.analyze_spectrum(&dft)
            .context("Failed to analyze frequency spectrum")?;
            
        debug!("Found {} frequency peaks", peaks.len());
        
        // Détection des motifs périodiques
        let detections = self.detect_periodic_patterns(
            &peaks,
            (image.cols(), image.rows())
        )?;
        
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs;
    
    #[test]
    fn test_frequency_detector_creation() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = FrequencyDetector::new(&config)?;
        Ok(())
    }
    
    #[test]
    fn test_dft_computation() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = FrequencyDetector::new(&config)?;
        
        let image = imgcodecs::imread(
            "tests/resources/images/pattern_sample.png",
            imgcodecs::IMREAD_COLOR
        )?;
        
        let dft = detector.compute_dft(&image)?;
        assert!(!dft.empty());
        Ok(())
    }
    
    #[test]
    fn test_periodic_pattern_detection() -> Result<()> {
        let config = DetectionConfig::default();
        let detector = FrequencyDetector::new(&config)?;
        
        let image = imgcodecs::imread(
            "tests/resources/images/pattern_sample.png",
            imgcodecs::IMREAD_COLOR
        )?;
        
        let detections = detector.detect(&image)?;
        assert!(!detections.is_empty());
        
        // Vérification que les détections sont de type Pattern
        for detection in detections {
            assert_eq!(detection.watermark_type, WatermarkType::Pattern);
            assert!(detection.confidence.value >= config.confidence_threshold);
        }
        
        Ok(())
    }
}

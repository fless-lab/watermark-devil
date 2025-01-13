use std::sync::Arc;
use opencv::core::{Mat, MatTraitConst};
use anyhow::Result;
use serde::{Serialize, Deserialize};
use tracing::{info, debug};

use crate::types::{Detection, WatermarkType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub avg_confidence: f32,
    pub avg_processing_time: f32,
    pub avg_psnr: f32,
    pub avg_ssim: f32,
    pub avg_fid: Option<f32>,
    pub num_samples: usize,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            avg_confidence: 0.0,
            avg_processing_time: 0.0,
            avg_psnr: 0.0,
            avg_ssim: 0.0,
            avg_fid: None,
            num_samples: 0,
        }
    }
}

pub struct MetricsCalculator {
    reference_images: Arc<Vec<Mat>>,
}

impl MetricsCalculator {
    pub fn new(reference_images: Vec<Mat>) -> Self {
        Self {
            reference_images: Arc::new(reference_images),
        }
    }
    
    pub fn calculate_metrics(
        &self,
        detections: &[Detection],
        results: &[Mat],
        processing_times: &[f32],
    ) -> Result<ModelMetrics> {
        let mut metrics = ModelMetrics::default();
        
        // Skip if no samples
        if detections.is_empty() {
            return Ok(metrics);
        }
        
        // Calculate basic metrics
        let mut total_confidence = 0.0;
        let mut total_processing_time = 0.0;
        let mut total_psnr = 0.0;
        let mut total_ssim = 0.0;
        let mut total_fid = 0.0;
        
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;
        
        for ((detection, result), &time) in detections.iter()
            .zip(results.iter())
            .zip(processing_times.iter())
        {
            // Update confidence and time
            total_confidence += detection.confidence.value;
            total_processing_time += time;
            
            // Calculate image quality metrics
            if let Some(ref_image) = self.reference_images.get(0) {
                total_psnr += self.calculate_psnr(result, ref_image)?;
                total_ssim += self.calculate_ssim(result, ref_image)?;
                if let Some(fid) = self.calculate_fid(result, ref_image)? {
                    total_fid += fid;
                }
            }
            
            // Update detection metrics
            if detection.confidence.value > 0.5 {
                if self.is_true_detection(detection) {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            } else if self.is_missed_detection(detection) {
                false_negatives += 1;
            }
        }
        
        let num_samples = detections.len();
        
        // Calculate final metrics
        metrics.num_samples = num_samples;
        metrics.avg_confidence = total_confidence / num_samples as f32;
        metrics.avg_processing_time = total_processing_time / num_samples as f32;
        metrics.avg_psnr = total_psnr / num_samples as f32;
        metrics.avg_ssim = total_ssim / num_samples as f32;
        
        if true_positives + false_positives > 0 {
            metrics.precision = true_positives as f32 / (true_positives + false_positives) as f32;
        }
        if true_positives + false_negatives > 0 {
            metrics.recall = true_positives as f32 / (true_positives + false_negatives) as f32;
        }
        if metrics.precision + metrics.recall > 0.0 {
            metrics.f1_score = 2.0 * (metrics.precision * metrics.recall) 
                / (metrics.precision + metrics.recall);
        }
        metrics.accuracy = true_positives as f32 
            / (true_positives + false_positives + false_negatives) as f32;
        
        if num_samples > 0 {
            metrics.avg_fid = Some(total_fid / num_samples as f32);
        }
        
        Ok(metrics)
    }
    
    fn calculate_psnr(&self, img1: &Mat, img2: &Mat) -> Result<f32> {
        let mut diff = Mat::default();
        opencv::core::absdiff(img1, img2, &mut diff)?;
        opencv::core::multiply(&diff, &diff, &mut diff, 1.0, -1)?;
        
        let mse = opencv::core::mean(&diff, &Mat::default())?[0];
        if mse == 0.0 {
            return Ok(f32::INFINITY);
        }
        
        let max_value = 255.0;
        Ok(20.0 * (max_value / mse.sqrt()).log10() as f32)
    }
    
    fn calculate_ssim(&self, img1: &Mat, img2: &Mat) -> Result<f32> {
        let c1 = (0.01 * 255.0).powi(2);
        let c2 = (0.03 * 255.0).powi(2);
        
        let mut mu1 = Mat::default();
        let mut mu2 = Mat::default();
        let mut sigma1_sq = Mat::default();
        let mut sigma2_sq = Mat::default();
        let mut sigma12 = Mat::default();
        
        opencv::imgproc::gaussian_blur(
            img1,
            &mut mu1,
            opencv::core::Size::new(11, 11),
            1.5,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;
        
        opencv::imgproc::gaussian_blur(
            img2,
            &mut mu2,
            opencv::core::Size::new(11, 11),
            1.5,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;
        
        opencv::core::multiply(&mu1, &mu1, &mut sigma1_sq, 1.0, -1)?;
        opencv::core::multiply(&mu2, &mu2, &mut sigma2_sq, 1.0, -1)?;
        opencv::core::multiply(&mu1, &mu2, &mut sigma12, 1.0, -1)?;
        
        let mut ssim_map = Mat::default();
        let numerator1 = 2.0 * opencv::core::mean(&sigma12, &Mat::default())?[0] + c2;
        let numerator2 = 2.0 * opencv::core::mean(&mu1, &Mat::default())?[0] 
            * opencv::core::mean(&mu2, &Mat::default())?[0] + c1;
        let denominator1 = opencv::core::mean(&sigma1_sq, &Mat::default())?[0] 
            + opencv::core::mean(&sigma2_sq, &Mat::default())?[0] + c2;
        let denominator2 = opencv::core::mean(&mu1, &Mat::default())?[0].powi(2) 
            + opencv::core::mean(&mu2, &Mat::default())?[0].powi(2) + c1;
        
        let ssim = (numerator1 * numerator2) / (denominator1 * denominator2);
        Ok(ssim as f32)
    }
    
    fn calculate_fid(&self, img1: &Mat, img2: &Mat) -> Result<Option<f32>> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;
            let inception = PyModule::import(py, "torchvision.models")?
                .getattr("inception_v3")?
                .call0()?;
            
            // Convert images to PyTorch tensors
            let tensor1 = mat_to_tensor(py, img1)?;
            let tensor2 = mat_to_tensor(py, img2)?;
            
            // Get inception features
            let features1 = inception.call1((tensor1,))?;
            let features2 = inception.call1((tensor2,))?;
            
            // Calculate FID
            let mu1 = features1.call_method0("mean")?;
            let mu2 = features2.call_method0("mean")?;
            let sigma1 = features1.call_method0("cov")?;
            let sigma2 = features2.call_method0("cov")?;
            
            let diff = mu1.call_method1("sub", (mu2,))?;
            let covmean = sigma1.call_method1("mm", (sigma2,))?
                .call_method0("sqrt")?;
            
            let fid = diff.call_method0("pow")?.call_method0("sum")?
                .call_method1("add", (
                    sigma1.call_method0("trace")?
                        .call_method1("add", (sigma2.call_method0("trace")?,))?
                        .call_method1("sub", (covmean.call_method0("trace")?.call_method0("mul")?,))?
                ,))?;
            
            Ok(Some(fid.extract::<f32>()?))
        })
    }
    
    fn is_true_detection(&self, detection: &Detection) -> bool {
        // Compare with ground truth
        // This is a placeholder - in practice, you would compare with actual ground truth
        detection.confidence.value > 0.8
    }
    
    fn is_missed_detection(&self, detection: &Detection) -> bool {
        // Check if we missed a real watermark
        // This is a placeholder - in practice, you would compare with actual ground truth
        detection.confidence.value < 0.3
    }
}

fn mat_to_tensor<'py>(py: Python<'py>, mat: &Mat) -> Result<&'py PyAny> {
    let torch = PyModule::import(py, "torch")?;
    let numpy = PyModule::import(py, "numpy")?;
    
    // Convert Mat to NumPy array
    let array = numpy.getattr("array")?.call1((mat.data_bytes()?,))?;
    
    // Convert to torch tensor
    let tensor = torch.getattr("from_numpy")?.call1((array,))?;
    
    // Add batch dimension and normalize
    let tensor = tensor.call_method1("unsqueeze", (0,))?;
    let tensor = tensor.call_method1("float", ())?;
    let tensor = tensor.call_method1("div", (255.0,))?;
    
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Scalar;
    
    #[test]
    fn test_metrics_calculation() -> Result<()> {
        // Create test images
        let img1 = Mat::new_rows_cols_with_default(
            100,
            100,
            opencv::core::CV_8UC3,
            Scalar::all(255.0),
        )?;
        
        let img2 = Mat::new_rows_cols_with_default(
            100,
            100,
            opencv::core::CV_8UC3,
            Scalar::all(128.0),
        )?;
        
        let calculator = MetricsCalculator::new(vec![img1.clone()]);
        
        let detections = vec![
            Detection {
                watermark_type: WatermarkType::Logo,
                confidence: crate::types::Confidence::new(0.9),
                bbox: opencv::core::Rect::new(0, 0, 50, 50),
                metadata: None,
            },
        ];
        
        let results = vec![img2];
        let processing_times = vec![0.1];
        
        let metrics = calculator.calculate_metrics(&detections, &results, &processing_times)?;
        
        assert!(metrics.avg_psnr > 0.0);
        assert!(metrics.avg_ssim > 0.0);
        assert_eq!(metrics.num_samples, 1);
        
        Ok(())
    }
}

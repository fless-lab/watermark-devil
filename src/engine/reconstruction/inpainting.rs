use image::DynamicImage;
use opencv as cv;
use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgproc;
use opencv::photo;
use anyhow::Result;

use crate::detection::DetectionResult;

pub struct InpaintingEngine {
    radius: i32,
    method: i32,
}

impl InpaintingEngine {
    pub fn new() -> Self {
        Self {
            radius: 3,
            method: photo::INPAINT_NS,
        }
    }

    pub async fn remove_watermark(
        &self,
        image: &DynamicImage,
        detection: &DetectionResult,
    ) -> Result<DynamicImage> {
        // Convertir l'image en Mat OpenCV
        let img = self.dynamic_image_to_mat(image)?;
        
        // Créer le masque pour l'inpainting
        let mut mask = Mat::new_rows_cols_with_default(
            img.rows(),
            img.cols(),
            cv::core::CV_8UC1,
            Scalar::from(0.0),
        )?;

        // Dessiner le masque basé sur la détection
        let bbox = &detection.bbox;
        let rect = cv::core::Rect::new(
            bbox.x as i32,
            bbox.y as i32,
            bbox.width as i32,
            bbox.height as i32,
        );
        
        if let Some(detection_mask) = &detection.mask {
            // Utiliser le masque précis si disponible
            let mask_mat = self.mask_to_mat(detection_mask)?;
            mask_mat.copy_to(&mut mask)?;
        } else {
            // Sinon, utiliser la bounding box
            let color = Scalar::from(255.0);
            imgproc::rectangle(&mut mask, rect, color, -1, imgproc::LINE_8, 0)?;
        }

        // Appliquer l'inpainting
        let mut result = Mat::default();
        photo::inpaint(&img, &mask, &mut result, self.radius as f64, self.method)?;

        // Post-process for better blending
        self.post_process(&img, &mut result, &mask)?;

        // Convertir le résultat en DynamicImage
        self.mat_to_dynamic_image(&result)
    }

    fn dynamic_image_to_mat(&self, image: &DynamicImage) -> Result<Mat> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        unsafe {
            Ok(Mat::new_rows_cols_with_data(
                height as i32,
                width as i32,
                cv::core::CV_8UC3,
                rgb.as_ptr() as *mut _,
                cv::core::Mat_AUTO_STEP,
            )?)
        }
    }

    fn mat_to_dynamic_image(&self, mat: &Mat) -> Result<DynamicImage> {
        let mut rgb_mat = Mat::default();
        imgproc::cvt_color(mat, &mut rgb_mat, imgproc::COLOR_BGR2RGB, 0)?;
        
        let width = rgb_mat.cols() as u32;
        let height = rgb_mat.rows() as u32;
        
        let mut buffer = vec![0u8; (width * height * 3) as usize];
        rgb_mat.data_bytes()?.copy_to_slice(&mut buffer);
        
        Ok(DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(width, height, buffer)
                .ok_or_else(|| anyhow::anyhow!("Failed to create image from buffer"))?
        ))
    }

    fn mask_to_mat(&self, mask: &image::RgbImage) -> Result<Mat> {
        let (width, height) = mask.dimensions();
        
        unsafe {
            Ok(Mat::new_rows_cols_with_data(
                height as i32,
                width as i32,
                cv::core::CV_8UC1,
                mask.as_ptr() as *mut _,
                cv::core::Mat_AUTO_STEP,
            )?)
        }
    }

    fn post_process(&self, original: &Mat, result: &mut Mat, mask: &Mat) -> Result<()> {
        // Blend edges for smoother transition
        let mut edges = Mat::default();
        imgproc::canny(mask, &mut edges, 50.0, 150.0, 3, false)?;
        
        let mut dilated_edges = Mat::default();
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            Size::new(3, 3),
            Point::new(-1, -1),
        )?;
        imgproc::dilate(&edges, &mut dilated_edges, &kernel, Point::new(-1, -1), 1, 1, 1.0)?;
        
        // Blend original and result at edges
        let mut blend = Mat::default();
        opencv::core::add_weighted(
            original,
            0.5,
            result,
            0.5,
            0.0,
            &mut blend,
            -1,
        )?;
        
        blend.copy_to_masked(result, &dilated_edges)?;
        
        Ok(())
    }

    // Méthodes avancées d'inpainting
    async fn texture_aware_inpainting(
        &self,
        image: &DynamicImage,
        mask: &Mat,
    ) -> Result<DynamicImage> {
        // Implémentation de PatchMatch ou autre algorithme avancé
        // Pour l'instant, utilise l'inpainting standard
        let img = self.dynamic_image_to_mat(image)?;
        let mut result = Mat::default();
        photo::inpaint(&img, mask, &mut result, self.radius as f64, photo::INPAINT_TELEA)?;
        self.mat_to_dynamic_image(&result)
    }

    async fn edge_preserving_inpainting(
        &self,
        image: &DynamicImage,
        mask: &Mat,
    ) -> Result<DynamicImage> {
        let img = self.dynamic_image_to_mat(image)?;
        
        // Détecter les bords
        let mut edges = Mat::default();
        imgproc::canny(&img, &mut edges, 100.0, 200.0, 3, false)?;
        
        // Combiner avec le masque original
        let mut combined_mask = Mat::default();
        cv::core::bitwise_or(&edges, mask, &mut combined_mask, &Mat::default())?;
        
        // Appliquer l'inpainting avec le masque combiné
        let mut result = Mat::default();
        photo::inpaint(&img, &combined_mask, &mut result, self.radius as f64, self.method)?;
        
        self.mat_to_dynamic_image(&result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbImage, Rgb};
    use crate::detection::BoundingBox;

    #[tokio::test]
    async fn test_inpainting() {
        let engine = InpaintingEngine::new();
        
        // Créer une image de test
        let mut test_image = RgbImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                test_image.put_pixel(x, y, Rgb([200, 200, 200]));
            }
        }
        
        // Ajouter un "watermark" simulé
        for y in 40..60 {
            for x in 40..60 {
                test_image.put_pixel(x, y, Rgb([100, 100, 100]));
            }
        }

        let detection = DetectionResult {
            confidence: 0.9,
            bbox: BoundingBox {
                x: 40,
                y: 40,
                width: 20,
                height: 20,
            },
            watermark_type: crate::detection::WatermarkType::Logo,
            mask: None,
        };

        let image = DynamicImage::ImageRgb8(test_image);
        let result = engine.remove_watermark(&image, &detection).await;
        
        assert!(result.is_ok(), "Inpainting should succeed");
    }
}

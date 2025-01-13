use std::sync::Arc;
use opencv::{
    core::{Mat, MatTraitConst, Size},
    imgproc::{self, INTER_LINEAR},
};
use tch::{Device, Tensor, nn};
use anyhow::Result;
use tracing::{info, debug};

use crate::types::{Detection, ReconstructionResult, ReconstructionMethod};
use super::ReconstructionConfig;

pub struct DiffusionReconstructor {
    model: Arc<DiffusionModel>,
    device: Device,
    config: ReconstructionConfig,
}

struct DiffusionModel {
    unet: nn::Sequential,
    noise_scheduler: NoiseScheduler,
    vs: nn::VarStore,
}

#[derive(Debug)]
struct NoiseScheduler {
    num_inference_steps: i64,
    beta_start: f64,
    beta_end: f64,
    betas: Tensor,
    alphas: Tensor,
    alphas_cumprod: Tensor,
}

impl DiffusionReconstructor {
    pub fn new(config: &ReconstructionConfig) -> Result<Self> {
        info!("Initializing diffusion reconstructor");
        
        let device = if config.use_gpu {
            Device::cuda_if_available()
        } else {
            Device::Cpu
        };
        
        let model = Arc::new(DiffusionModel::new(device, config)?);
        
        Ok(Self {
            model,
            device,
            config: config.clone(),
        })
    }

    pub async fn reconstruct(&self, image: &Mat, detection: &Detection) -> Result<ReconstructionResult> {
        debug!("Starting diffusion-based reconstruction for detection: {:?}", detection);
        let start_time = std::time::Instant::now();

        // Prétraitement
        let original_tensor = self.preprocess_image(image)?;
        let condition_mask = self.create_condition_mask(image, detection)?;
        
        // Initialisation du bruit
        let noise = self.generate_initial_noise(original_tensor.size().as_slice())?;
        let mut x_t = noise;

        // Processus de débruitage progressif
        for timestep in (0..self.model.noise_scheduler.num_inference_steps).rev() {
            debug!("Denoising step {}", timestep);
            x_t = self.denoise_step(x_t, &original_tensor, &condition_mask, timestep)?;
        }

        // Combinaison avec l'image originale
        let reconstructed = self.combine_with_original(image, &x_t, &condition_mask)?;
        
        // Évaluation de la qualité
        let quality_score = self.calculate_quality(image, &reconstructed, detection)?;
        
        let processing_time = start_time.elapsed();
        info!(
            "Reconstruction completed in {:?} with quality score: {:.2}",
            processing_time, quality_score
        );

        Ok(ReconstructionResult {
            success: true,
            quality_score,
            processing_time,
            method: ReconstructionMethod::Diffusion,
            output_image: reconstructed,
        })
    }

    fn preprocess_image(&self, image: &Mat) -> Result<Tensor> {
        debug!("Preprocessing image for diffusion");
        
        // Convertir en RGB si nécessaire
        let mut rgb_image = Mat::default();
        if image.channels()? == 1 {
            imgproc::cvt_color(image, &mut rgb_image, imgproc::COLOR_GRAY2RGB, 0)?;
        } else if image.channels()? == 4 {
            imgproc::cvt_color(image, &mut rgb_image, imgproc::COLOR_BGRA2RGB, 0)?;
        } else {
            imgproc::cvt_color(image, &mut rgb_image, imgproc::COLOR_BGR2RGB, 0)?;
        }

        // Normaliser à [-1, 1]
        let normalized = unsafe {
            let mut float_mat = Mat::new_rows_cols(
                rgb_image.rows()?,
                rgb_image.cols()?,
                opencv::core::CV_32FC3,
            )?;
            rgb_image.convert_to(&mut float_mat, opencv::core::CV_32F, 1.0/127.5, -1.0)?;
            float_mat
        };

        // Convertir en Tensor
        let tensor = Tensor::try_from(normalized)?
            .to_device(self.device)
            .permute(&[2, 0, 1])  // HWC -> CHW
            .unsqueeze(0);        // Ajouter dimension batch
        
        Ok(tensor)
    }

    fn create_condition_mask(&self, image: &Mat, detection: &Detection) -> Result<Tensor> {
        debug!("Creating condition mask for detection: {:?}", detection);
        
        // Créer un masque binaire
        let mut mask = Mat::new_rows_cols(
            image.rows()?,
            image.cols()?,
            opencv::core::CV_8UC1,
        )?;
        mask.set_scalar(0.into())?;

        // Dessiner la zone de détection
        let bbox = detection.bbox;
        let rect = opencv::core::Rect::new(
            bbox.x,
            bbox.y,
            bbox.width,
            bbox.height,
        );
        
        // Appliquer un flou gaussien aux bords pour transition douce
        let mut smooth_mask = mask.clone();
        imgproc::gaussian_blur(
            &mask,
            &mut smooth_mask,
            Size::new(5, 5),
            1.5,
            1.5,
            opencv::core::BORDER_DEFAULT,
        )?;

        // Convertir en Tensor
        let mask_tensor = Tensor::try_from(smooth_mask)?
            .to_device(self.device)
            .to_kind(tch::Kind::Float)
            .div(255.0)  // Normaliser à [0, 1]
            .unsqueeze(0)  // Ajouter dimension batch
            .unsqueeze(0); // Ajouter dimension canal

        Ok(mask_tensor)
    }

    fn generate_initial_noise(&self, size: &[i64]) -> Result<Tensor> {
        Ok(Tensor::randn(size, (tch::Kind::Float, self.device)))
    }

    fn denoise_step(
        &self,
        x_t: Tensor,
        original: &Tensor,
        condition_mask: &Tensor,
        timestep: i64,
    ) -> Result<Tensor> {
        // Obtenir le niveau de bruit pour ce pas de temps
        let noise_level = self.model.noise_scheduler.get_noise_level(timestep);
        
        // Prédire le bruit
        let noise_pred = self.model.unet.forward_t(
            &x_t,
            Some(timestep),
            Some(condition_mask),
            false
        )?;
        
        // Débruiter avec le scheduler
        let denoised = self.model.noise_scheduler.step(
            &x_t,
            &noise_pred,
            timestep,
            noise_level,
        )?;
        
        // Appliquer le masque de condition
        let result = denoised.mul(condition_mask) + original.mul(condition_mask.neg() + 1.0);
        
        Ok(result)
    }

    fn combine_with_original(&self, original: &Mat, generated: &Tensor, mask: &Tensor) -> Result<Mat> {
        // Convertir le tensor généré en Mat
        let mut gen_mat = Mat::default();
        generated
            .squeeze()  // Enlever dimension batch
            .permute(&[1, 2, 0])  // CHW -> HWC
            .to_device(Device::Cpu)
            .try_into_mat()?
            .convert_to(&mut gen_mat, opencv::core::CV_8U, 127.5, 127.5)?;
        
        // Convertir le masque en Mat
        let mut mask_mat = Mat::default();
        mask
            .squeeze_all()
            .to_device(Device::Cpu)
            .try_into_mat()?
            .convert_to(&mut mask_mat, opencv::core::CV_8U, 255.0, 0.0)?;
        
        // Combiner avec l'image originale
        let mut result = Mat::default();
        opencv::core::add_weighted(
            &gen_mat,
            1.0,
            original,
            1.0,
            0.0,
            &mut result,
            -1,
        )?;
        
        Ok(result)
    }

    fn calculate_quality(&self, original: &Mat, result: &Mat, detection: &Detection) -> Result<f32> {
        // Calculer SSIM uniquement sur la zone reconstruite
        let bbox = detection.bbox;
        let rect = opencv::core::Rect::new(
            bbox.x,
            bbox.y,
            bbox.width,
            bbox.height,
        );
        
        let orig_roi = Mat::roi(original, rect)?;
        let result_roi = Mat::roi(result, rect)?;
        
        // Convertir en niveaux de gris pour SSIM
        let mut orig_gray = Mat::default();
        let mut result_gray = Mat::default();
        imgproc::cvt_color(&orig_roi, &mut orig_gray, imgproc::COLOR_BGR2GRAY, 0)?;
        imgproc::cvt_color(&result_roi, &mut result_gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Calculer SSIM
        let mut ssim = 0.0;
        unsafe {
            opencv::quality::quality_ssim(&orig_gray, &result_gray, None)?
                .data_typed::<f32>()?
                .iter()
                .for_each(|&x| ssim += x);
        }
        
        Ok(ssim)
    }
}

impl DiffusionModel {
    fn new(device: Device, config: &ReconstructionConfig) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        
        // Create U-Net model
        let unet = Self::create_unet(&vs.root(), 6)?; // 3 channels image + 3 channels mask
        
        // Load weights if available
        if let Some(weights_path) = &config.model_path {
            vs.load(weights_path)?;
        }
        
        let noise_scheduler = NoiseScheduler::new(
            if config.quality == "high" { 50 } else { 20 },
            1e-4,
            0.02,
        )?;
        
        Ok(Self {
            unet,
            noise_scheduler,
            vs,
        })
    }

    fn create_unet(vs: &nn::Path, in_channels: i64) -> Result<nn::Sequential> {
        let seq = nn::seq()
            .add(nn::conv2d(
                vs / "conv_in",
                in_channels,
                128,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
            // Encoder
            .add(Self::down_block(vs / "down1", 128, 256))
            .add(Self::down_block(vs / "down2", 256, 512))
            .add(Self::down_block(vs / "down3", 512, 512))
            // Middle
            .add(Self::attention_block(vs / "middle", 512))
            // Decoder
            .add(Self::up_block(vs / "up3", 512, 512))
            .add(Self::up_block(vs / "up2", 512, 256))
            .add(Self::up_block(vs / "up1", 256, 128))
            // Output
            .add(nn::conv2d(
                vs / "conv_out",
                128,
                3,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    ..Default::default()
                },
            ));
        
        Ok(seq)
    }

    fn down_block(vs: nn::Path, in_channels: i64, out_channels: i64) -> nn::Sequential {
        nn::seq()
            .add(nn::conv2d(
                &vs / "conv1",
                in_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    stride: 2,
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(
                &vs / "conv2",
                out_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
    }

    fn up_block(vs: nn::Path, in_channels: i64, out_channels: i64) -> nn::Sequential {
        nn::seq()
            .add_fn_t(move |x, train| {
                x.upsample_nearest2d(&[2, 2], None, None)
            })
            .add(nn::conv2d(
                &vs / "conv1",
                in_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
            .add(nn::conv2d(
                &vs / "conv2",
                out_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.relu())
    }

    fn attention_block(vs: nn::Path, channels: i64) -> nn::Sequential {
        nn::seq()
            .add(nn::conv2d(
                &vs / "query",
                channels,
                channels,
                1,
                nn::ConvConfig::default(),
            ))
            .add(nn::conv2d(
                &vs / "key",
                channels,
                channels,
                1,
                nn::ConvConfig::default(),
            ))
            .add(nn::conv2d(
                &vs / "value",
                channels,
                channels,
                1,
                nn::ConvConfig::default(),
            ))
            .add_fn(move |x| {
                let (b, c, h, w) = x.size4().unwrap();
                let query = x.slice(1, 0, c/3, 1).view([b, c/3, h*w]);
                let key = x.slice(1, c/3, 2*c/3, 1).view([b, c/3, h*w]);
                let value = x.slice(1, 2*c/3, c, 1).view([b, c/3, h*w]);
                
                let attention = query.bmm(&key.transpose(1, 2)) / (c as f64).sqrt();
                let attention = attention.softmax(-1, tch::Kind::Float);
                
                let out = attention.bmm(&value).view([b, c/3, h, w]);
                out
            })
    }
}

impl NoiseScheduler {
    fn new(num_inference_steps: i64, beta_start: f64, beta_end: f64) -> Result<Self> {
        let device = Device::cuda_if_available();
        
        // Create noise schedule
        let betas = Tensor::linspace(beta_start, beta_end, num_inference_steps, (tch::Kind::Float, device));
        let alphas = Tensor::ones(&[num_inference_steps], (tch::Kind::Float, device)) - &betas;
        let alphas_cumprod = alphas.cumprod(0, tch::Kind::Float);
        
        Ok(Self {
            num_inference_steps,
            beta_start,
            beta_end,
            betas,
            alphas,
            alphas_cumprod,
        })
    }

    fn get_noise_level(&self, timestep: i64) -> f64 {
        let alpha_cumprod = self.alphas_cumprod.get(timestep).double_value(&[]);
        (1.0 - alpha_cumprod).sqrt()
    }

    fn step(
        &self,
        x_t: &Tensor,
        noise_pred: &Tensor,
        timestep: i64,
        noise_level: f64,
    ) -> Result<Tensor> {
        let alpha = self.alphas.get(timestep).double_value(&[]);
        let alpha_cumprod = self.alphas_cumprod.get(timestep).double_value(&[]);
        let beta = self.betas.get(timestep).double_value(&[]);
        
        let pred_original = (x_t - noise_pred.mul(noise_level)) / (1.0 - noise_level.powi(2)).sqrt();
        let pred_sample_direction = noise_pred.mul((beta * (1.0 - alpha_cumprod).sqrt() / (1.0 - alpha)).sqrt());
        
        Ok(pred_original + pred_sample_direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs;
    
    #[test]
    fn test_diffusion_reconstruction() -> Result<()> {
        // Create test image
        let mut image = Mat::new_rows_cols_with_default(
            512,
            512,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(255.0),
        )?;
        
        // Add watermark
        imgproc::put_text(
            &mut image,
            "TEST",
            opencv::core::Point::new(200, 250),
            imgproc::FONT_HERSHEY_SIMPLEX,
            2.0,
            opencv::core::Scalar::new(128.0, 128.0, 128.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;
        
        // Create detection
        let detection = Detection {
            watermark_type: crate::types::WatermarkType::Text,
            confidence: crate::types::Confidence::new(0.9),
            bbox: opencv::core::Rect::new(180, 200, 200, 100),
            metadata: None,
        };
        
        // Create reconstructor
        let config = ReconstructionConfig {
            method: ReconstructionMethod::Diffusion,
            quality: "high".to_string(),
            use_gpu: false,
            preserve_details: true,
            max_iterations: 1000,
        };
        
        let reconstructor = DiffusionReconstructor::new(&config)?;
        
        // Run reconstruction
        let result = reconstructor.reconstruct(&image, &detection)?;
        
        // Verify result
        assert!(result.success);
        assert!(result.quality_score > 0.5);
        assert_eq!(result.method, ReconstructionMethod::Diffusion);
        assert_eq!(result.output_image.size()?, image.size()?);
        
        Ok(())
    }
    
    #[test]
    fn test_noise_scheduler() -> Result<()> {
        let scheduler = NoiseScheduler::new(50, 1e-4, 0.02)?;
        
        // Test noise levels
        let start_level = scheduler.get_noise_level(0);
        let mid_level = scheduler.get_noise_level(25);
        let end_level = scheduler.get_noise_level(49);
        
        assert!(start_level < mid_level);
        assert!(mid_level < end_level);
        assert!(end_level <= 1.0);
        
        Ok(())
    }
}

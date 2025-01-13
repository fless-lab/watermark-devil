//! Tests for engine components.

use super::*;
use std::path::PathBuf;
use image::{DynamicImage, ImageBuffer, Rgb};
use tempfile::tempdir;
use metrics::{counter, gauge};

#[cfg(test)]
mod tests {
    use super::*;

    // Helper functions
    fn create_test_image() -> DynamicImage {
        let buffer = ImageBuffer::from_fn(100, 100, |_, _| {
            Rgb([0, 0, 0])
        });
        DynamicImage::ImageRgb8(buffer)
    }

    #[test]
    fn test_detection_validation() {
        // Test avec une image valide
        let img = create_test_image();
        let result = validate_detection_input(&img);
        assert!(result.is_ok());

        // Test avec une image trop grande
        let large_img = DynamicImage::ImageRgb8(
            ImageBuffer::from_fn(5000, 5000, |_, _| {
                Rgb([0, 0, 0])
            })
        );
        let result = validate_detection_input(&large_img);
        assert!(result.is_err());
    }

    #[test]
    fn test_reconstruction_validation() {
        // Test avec des paramètres valides
        let img = create_test_image();
        let result = validate_reconstruction_input(&img, 0.8);
        assert!(result.is_ok());

        // Test avec qualité invalide
        let result = validate_reconstruction_input(&img, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_optimization_validation() {
        // Test avec une image valide
        let img = create_test_image();
        let result = validate_optimization_input(&img);
        assert!(result.is_ok());

        // Test avec une image invalide
        let empty_img = DynamicImage::ImageRgb8(
            ImageBuffer::new(0, 0)
        );
        let result = validate_optimization_input(&empty_img);
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_support() {
        // Test la détection CUDA
        let has_cuda = check_cuda_support();
        
        // La valeur dépend de l'environnement
        gauge("test.cuda_available", has_cuda as i64);
    }

    #[test]
    fn test_memory_management() {
        // Test l'allocation mémoire
        let result = allocate_gpu_memory(1024);
        assert!(result.is_ok());

        // Test la libération
        let handle = result.unwrap();
        let result = free_gpu_memory(handle);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_processing() {
        // Test le traitement parallèle
        let data = vec![1, 2, 3, 4, 5];
        let result = process_parallel(&data, |x| x * 2);
        
        assert_eq!(result.len(), data.len());
        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_error_handling() {
        // Test la gestion des erreurs
        let result: Result<(), EngineError> = Err(
            EngineError::ValidationError("Test error".to_string())
        );
        
        match result {
            Ok(_) => panic!("Should be error"),
            Err(e) => {
                assert!(matches!(e, EngineError::ValidationError(_)));
                counter("test.errors", 1);
            }
        }
    }

    #[test]
    fn test_file_io() {
        // Créer un répertoire temporaire
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.png");
        
        // Sauvegarder l'image
        let img = create_test_image();
        let result = img.save(&file_path);
        assert!(result.is_ok());
        
        // Charger l'image
        let result = image::open(&file_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metrics() {
        // Test l'enregistrement des métriques
        gauge("test.value", 42);
        counter("test.counter", 1);
        
        // Les métriques sont enregistrées de manière asynchrone
        // donc on ne peut pas les vérifier directement
    }

    #[test]
    fn test_optimization() {
        // Test l'optimisation
        let img = create_test_image();
        let result = optimize_image(&img);
        
        assert!(result.is_ok());
        let optimized = result.unwrap();
        
        // Vérifier les dimensions
        assert!(optimized.width() <= img.width());
        assert!(optimized.height() <= img.height());
    }
}

use std::path::PathBuf;
use opencv::core::{Mat, MatTraitConst};
use opencv::imgcodecs;
use anyhow::Result;

use watermark_evil::engine::detection::{DetectionEngine, Detection};
use watermark_evil::config::DetectionConfig;
use watermark_evil::types::WatermarkType;

// Helper pour charger une image de test
fn load_test_image(name: &str) -> Result<Mat> {
    let test_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("integration")
        .join("resources")
        .join("images");
    
    let image_path = test_dir.join(name);
    Ok(imgcodecs::imread(
        image_path.to_str().unwrap(),
        imgcodecs::IMREAD_COLOR
    )?)
}

#[test]
fn test_detection_engine_initialization() -> Result<()> {
    let config = DetectionConfig::default();
    let engine = DetectionEngine::new(config)?;
    println!("Python integration successful");
    Ok(())
}

#[test]
fn test_logo_detection() -> Result<()> {
    let config = DetectionConfig::default();
    let engine = DetectionEngine::new(config)?;
    
    let image = load_test_image("logo.png")?;
    let detections = engine.detect(&image)?;
    
    assert!(!detections.is_empty(), "Should detect at least one logo");
    
    // Vérifier que nous avons au moins une détection de type Logo
    let logo_detections: Vec<&Detection> = detections.iter()
        .filter(|d| d.watermark_type == WatermarkType::Logo)
        .collect();
    
    assert!(!logo_detections.is_empty(), "Should have logo detections");
    
    // Vérifier les propriétés de la détection
    let detection = logo_detections[0];
    assert!(detection.confidence.value > 0.3, "Should have reasonable confidence");
    assert!(detection.bbox.width > 0 && detection.bbox.height > 0, "Should have valid bbox");
    
    Ok(())
}

#[test]
fn test_text_detection() -> Result<()> {
    let config = DetectionConfig::default();
    let engine = DetectionEngine::new(config)?;
    
    let image = load_test_image("text.png")?;
    let detections = engine.detect(&image)?;
    
    let text_detections: Vec<&Detection> = detections.iter()
        .filter(|d| d.watermark_type == WatermarkType::Text)
        .collect();
    
    assert!(!text_detections.is_empty(), "Should detect text watermark");
    
    // Vérifier les métadonnées (qui devraient contenir le texte)
    if let Some(metadata) = &text_detections[0].metadata {
        assert!(metadata.get("text").is_some(), "Should contain detected text");
    }
    
    Ok(())
}

#[test]
fn test_pattern_detection() -> Result<()> {
    let config = DetectionConfig::default();
    let engine = DetectionEngine::new(config)?;
    
    let image = load_test_image("pattern.png")?;
    let detections = engine.detect(&image)?;
    
    let pattern_detections: Vec<&Detection> = detections.iter()
        .filter(|d| d.watermark_type == WatermarkType::Pattern)
        .collect();
    
    assert!(!pattern_detections.is_empty(), "Should detect pattern watermark");
    
    // Vérifier la périodicité des détections
    if pattern_detections.len() >= 2 {
        let d1 = &pattern_detections[0].bbox;
        let d2 = &pattern_detections[1].bbox;
        let distance = ((d1.x - d2.x).pow(2) + (d1.y - d2.y).pow(2)) as f32;
        assert!(distance > 0.0, "Patterns should be spatially separated");
    }
    
    Ok(())
}

#[test]
fn test_transparency_detection() -> Result<()> {
    let config = DetectionConfig::default();
    let engine = DetectionEngine::new(config)?;
    
    let image = load_test_image("transparent.png")?;
    let detections = engine.detect(&image)?;
    
    let transparent_detections: Vec<&Detection> = detections.iter()
        .filter(|d| d.watermark_type == WatermarkType::Transparent)
        .collect();
    
    assert!(!transparent_detections.is_empty(), "Should detect transparent watermark");
    
    // Vérifier la présence du masque
    assert!(transparent_detections[0].mask.is_some(), "Should have transparency mask");
    
    Ok(())
}

#[test]
fn test_multiple_watermarks() -> Result<()> {
    let config = DetectionConfig::default();
    let engine = DetectionEngine::new(config)?;
    
    // Charger une image avec plusieurs types de filigranes
    let image = load_test_image("combined.png")?;
    let detections = engine.detect(&image)?;
    
    // Vérifier que nous détectons plusieurs types
    let watermark_types: std::collections::HashSet<_> = detections.iter()
        .map(|d| d.watermark_type)
        .collect();
    
    assert!(watermark_types.len() > 1, "Should detect multiple watermark types");
    
    Ok(())
}

#[test]
fn test_batch_detection() -> Result<()> {
    let config = DetectionConfig::default();
    let engine = DetectionEngine::new(config)?;
    
    // Charger plusieurs images
    let images = vec![
        load_test_image("logo.png")?,
        load_test_image("text.png")?,
        load_test_image("pattern.png")?,
    ];
    
    let batch_results = engine.detect_batch(&images)?;
    
    assert_eq!(batch_results.len(), images.len(), "Should have results for all images");
    assert!(batch_results.iter().all(|r| !r.is_empty()), "All images should have detections");
    
    Ok(())
}

#[test]
fn test_detection_confidence() -> Result<()> {
    let mut config = DetectionConfig::default();
    config.confidence_threshold = 0.8; // Seuil élevé
    
    let engine = DetectionEngine::new(config)?;
    let image = load_test_image("logo.png")?;
    let detections = engine.detect(&image)?;
    
    // Vérifier que toutes les détections respectent le seuil
    assert!(detections.iter().all(|d| d.confidence.value >= 0.8),
            "All detections should meet confidence threshold");
    
    Ok(())
}

#[test]
fn test_error_handling() -> Result<()> {
    let config = DetectionConfig::default();
    let engine = DetectionEngine::new(config)?;
    
    // Tester avec une image invalide
    let invalid_image = Mat::default();
    let result = engine.detect(&invalid_image);
    
    assert!(result.is_err(), "Should handle invalid input gracefully");
    
    Ok(())
}

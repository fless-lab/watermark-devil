use watermark_evil::{
    ml::models::{
        logo_detector::LogoDetector,
        text_detector::TextDetector,
        pattern_detector::PatternDetector,
    }
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize detectors
    let logo_detector = LogoDetector::new();
    let text_detector = TextDetector::new();
    let pattern_detector = PatternDetector::new();

    // Path to image
    let image_path = PathBuf::from("path/to/your/image.jpg");

    // Detect watermarks
    println!("Detecting watermarks in {:?}", image_path);

    // Logo detection
    if let Ok(logos) = logo_detector.detect(&image_path) {
        println!("Found {} logo watermarks:", logos.len());
        for (i, logo) in logos.iter().enumerate() {
            println!("  Logo {}: confidence={:.2}, bbox={:?}", 
                    i + 1, logo.confidence, logo.bbox);
        }
    }

    // Text detection
    if let Ok(texts) = text_detector.detect(&image_path) {
        println!("Found {} text watermarks:", texts.len());
        for (i, text) in texts.iter().enumerate() {
            println!("  Text {}: content='{}', confidence={:.2}, bbox={:?}",
                    i + 1, text.content, text.confidence, text.bbox);
        }
    }

    // Pattern detection
    if let Ok(patterns) = pattern_detector.detect(&image_path) {
        println!("Found {} pattern watermarks:", patterns.len());
        for (i, pattern) in patterns.iter().enumerate() {
            println!("  Pattern {}: type={}, confidence={:.2}, bbox={:?}",
                    i + 1, pattern.pattern_type, pattern.confidence, pattern.bbox);
        }
    }

    Ok(())
}

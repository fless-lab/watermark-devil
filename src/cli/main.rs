use clap::{Parser, ValueEnum};
use serde_json::json;
use std::path::PathBuf;
use std::fs;
use log::{info, error};
use anyhow::{Result, Context};

use crate::ml::models::{
    logo_detector::LogoDetector,
    text_detector::TextDetector,
    pattern_detector::PatternDetector,
    transparency_detector::TransparencyDetector,
};

#[derive(Debug, Clone, ValueEnum)]
enum DetectorType {
    All,
    Logo,
    Text,
    Pattern,
    Transparency,
}

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the image file to analyze
    #[arg(value_name = "IMAGE_PATH")]
    image_path: PathBuf,

    /// Which detectors to use
    #[arg(short, long, value_enum, num_args = 1.., default_value = "all")]
    detectors: Vec<DetectorType>,

    /// Output format
    #[arg(short, long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Output file path (optional)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, serde::Serialize)]
struct Detection {
    detector_type: String,
    confidence: f32,
    bbox: Option<[f32; 4]>,
    text: Option<String>,
    area: Option<u32>,
}

#[derive(Debug, serde::Serialize)]
struct DetectionResults {
    image_path: String,
    detections: Vec<Detection>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    errors: Vec<String>,
}

fn detect_watermarks(image_path: &PathBuf, detectors: &[DetectorType]) -> Result<DetectionResults> {
    let image = image::open(image_path)
        .with_context(|| format!("Failed to open image: {}", image_path.display()))?;
    
    let mut results = DetectionResults {
        image_path: image_path.display().to_string(),
        detections: Vec::new(),
        errors: Vec::new(),
    };

    let use_all = detectors.contains(&DetectorType::All);
    
    // Logo detection
    if use_all || detectors.contains(&DetectorType::Logo) {
        match LogoDetector::new() {
            Ok(detector) => {
                let processed = detector.preprocess(&image);
                if let Ok(detections) = detector.detect(&processed) {
                    for (bbox, conf) in detections {
                        results.detections.push(Detection {
                            detector_type: "logo".to_string(),
                            confidence: conf,
                            bbox: Some(bbox),
                            text: None,
                            area: None,
                        });
                    }
                }
            },
            Err(e) => results.errors.push(format!("Logo detector error: {}", e)),
        }
    }

    // Text detection
    if use_all || detectors.contains(&DetectorType::Text) {
        match TextDetector::new() {
            Ok(detector) => {
                let processed = detector.preprocess(&image);
                if let Ok(detections) = detector.detect(&processed) {
                    for (bbox, conf, text) in detections {
                        results.detections.push(Detection {
                            detector_type: "text".to_string(),
                            confidence: conf,
                            bbox: Some(bbox),
                            text: Some(text),
                            area: None,
                        });
                    }
                }
            },
            Err(e) => results.errors.push(format!("Text detector error: {}", e)),
        }
    }

    // Pattern detection
    if use_all || detectors.contains(&DetectorType::Pattern) {
        match PatternDetector::new() {
            Ok(detector) => {
                let processed = detector.preprocess(&image);
                if let Ok(detections) = detector.detect(&processed) {
                    for (area, conf) in detections {
                        results.detections.push(Detection {
                            detector_type: "pattern".to_string(),
                            confidence: conf,
                            bbox: None,
                            text: None,
                            area: Some(area),
                        });
                    }
                }
            },
            Err(e) => results.errors.push(format!("Pattern detector error: {}", e)),
        }
    }

    // Transparency detection
    if use_all || detectors.contains(&DetectorType::Transparency) {
        match TransparencyDetector::new() {
            Ok(detector) => {
                let processed = detector.preprocess(&image);
                if let Ok(detections) = detector.detect(&processed) {
                    for (area, conf) in detections {
                        results.detections.push(Detection {
                            detector_type: "transparency".to_string(),
                            confidence: conf,
                            bbox: None,
                            text: None,
                            area: Some(area),
                        });
                    }
                }
            },
            Err(e) => results.errors.push(format!("Transparency detector error: {}", e)),
        }
    }

    Ok(results)
}

fn format_results(results: &DetectionResults, format: &OutputFormat) -> String {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(&results).unwrap_or_else(|e| {
            format!("Error formatting JSON: {}", e)
        }),
        
        OutputFormat::Text => {
            let mut output = vec![format!("Results for {}:", results.image_path)];
            
            if results.detections.is_empty() {
                output.push("No watermarks detected.".to_string());
            } else {
                for det in &results.detections {
                    match det.detector_type.as_str() {
                        "logo" => {
                            output.push(format!("Logo detected (confidence: {:.2})", det.confidence));
                            if let Some(bbox) = det.bbox {
                                output.push(format!("  Location: {:?}", bbox));
                            }
                        },
                        "text" => {
                            output.push(format!("Text detected: '{}'", 
                                det.text.as_ref().unwrap_or(&String::from("unknown"))));
                            output.push(format!("  Confidence: {:.2}", det.confidence));
                            if let Some(bbox) = det.bbox {
                                output.push(format!("  Location: {:?}", bbox));
                            }
                        },
                        "pattern" => {
                            output.push("Repeating pattern detected".to_string());
                            output.push(format!("  Confidence: {:.2}", det.confidence));
                            if let Some(area) = det.area {
                                output.push(format!("  Area: {} pixels", area));
                            }
                        },
                        "transparency" => {
                            output.push("Semi-transparent watermark detected".to_string());
                            output.push(format!("  Confidence: {:.2}", det.confidence));
                            if let Some(area) = det.area {
                                output.push(format!("  Area: {} pixels", area));
                            }
                        },
                        _ => {}
                    }
                }
            }

            if !results.errors.is_empty() {
                output.push("\nErrors encountered:".to_string());
                for error in &results.errors {
                    output.push(format!("  {}", error));
                }
            }

            output.join("\n")
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    info!("Processing image: {}", args.image_path.display());

    match detect_watermarks(&args.image_path, &args.detectors) {
        Ok(results) => {
            let output = format_results(&results, &args.format);
            
            if let Some(output_path) = args.output {
                fs::write(&output_path, output)
                    .with_context(|| format!("Failed to write to output file: {}", output_path.display()))?;
                info!("Results written to: {}", output_path.display());
            } else {
                println!("{}", output);
            }
        },
        Err(e) => {
            error!("Error processing image: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

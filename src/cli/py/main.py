"""
Command-line interface for watermark detection
"""
import argparse
import sys
from pathlib import Path
import cv2
import json
from typing import List, Dict, Any
import logging

from src.ml.models.logo_detector import LogoDetector
from src.ml.models.text_detector import TextDetector
from src.ml.models.pattern_detector import PatternDetector
from src.ml.models.transparency_detector import TransparencyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_watermarks(image_path: str, detectors: List[str], output_format: str = 'text') -> Dict[str, Any]:
    """
    Detect watermarks in an image using specified detectors
    
    Args:
        image_path: Path to the image file
        detectors: List of detectors to use ('logo', 'text', 'pattern', 'transparency', or 'all')
        output_format: Output format ('text' or 'json')
        
    Returns:
        Dictionary containing detection results
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    results = {
        'image_path': image_path,
        'detections': []
    }
    
    # Initialize requested detectors
    if 'all' in detectors:
        detectors = ['logo', 'text', 'pattern', 'transparency']
    
    for detector_type in detectors:
        try:
            if detector_type == 'logo':
                detector = LogoDetector()
                processed = detector.preprocess(image)
                detector_results = detector.detect(processed)
                for box, conf in detector_results:
                    results['detections'].append({
                        'type': 'logo',
                        'confidence': float(conf),
                        'bbox': box
                    })
                
            elif detector_type == 'text':
                detector = TextDetector()
                processed = detector.preprocess(image)
                detector_results = detector.detect(processed)
                for box, conf, text in detector_results:
                    results['detections'].append({
                        'type': 'text',
                        'text': text,
                        'confidence': float(conf),
                        'bbox': box
                    })
                
            elif detector_type == 'pattern':
                detector = PatternDetector()
                processed = detector.preprocess(image)
                detector_results = detector.detect(processed)
                for area, conf in detector_results:
                    results['detections'].append({
                        'type': 'pattern',
                        'confidence': float(conf),
                        'area': area
                    })
                
            elif detector_type == 'transparency':
                detector = TransparencyDetector()
                processed = detector.preprocess(image)
                detector_results = detector.detect(processed)
                for area, conf in detector_results:
                    results['detections'].append({
                        'type': 'transparency',
                        'confidence': float(conf),
                        'area': area
                    })
            
            else:
                logger.warning(f"Unknown detector type: {detector_type}")
                
        except Exception as e:
            logger.error(f"Error with {detector_type} detector: {str(e)}")
            if output_format == 'json':
                results['errors'] = results.get('errors', []) + [
                    {'detector': detector_type, 'error': str(e)}
                ]
    
    return results

def format_results(results: Dict[str, Any], output_format: str = 'text') -> str:
    """Format detection results for output"""
    if output_format == 'json':
        return json.dumps(results, indent=2)
    
    # Text format
    output = [f"Results for {results['image_path']}:"]
    
    if not results['detections']:
        output.append("No watermarks detected.")
        return "\n".join(output)
    
    for det in results['detections']:
        if det['type'] == 'logo':
            output.append(f"Logo detected (confidence: {det['confidence']:.2f})")
            output.append(f"  Location: {det['bbox']}")
            
        elif det['type'] == 'text':
            output.append(f"Text detected: '{det['text']}'")
            output.append(f"  Confidence: {det['confidence']:.2f}")
            output.append(f"  Location: {det['bbox']}")
            
        elif det['type'] == 'pattern':
            output.append(f"Repeating pattern detected")
            output.append(f"  Confidence: {det['confidence']:.2f}")
            output.append(f"  Area: {det['area']} pixels")
            
        elif det['type'] == 'transparency':
            output.append(f"Semi-transparent watermark detected")
            output.append(f"  Confidence: {det['confidence']:.2f}")
            output.append(f"  Area: {det['area']} pixels")
    
    if 'errors' in results:
        output.append("\nErrors encountered:")
        for error in results['errors']:
            output.append(f"  {error['detector']}: {error['error']}")
    
    return "\n".join(output)

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Detect watermarks in images using various detection methods."
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to analyze"
    )
    
    parser.add_argument(
        "--detectors",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "logo", "text", "pattern", "transparency"],
        help="Which detectors to use (default: all)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate image path
        image_path = Path(args.image_path)
        if not image_path.exists():
            logger.error(f"Image file not found: {args.image_path}")
            sys.exit(1)
        
        # Run detection
        results = detect_watermarks(
            str(image_path),
            args.detectors,
            args.format
        )
        
        # Format output
        output = format_results(results, args.format)
        
        # Write or print results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output)
            logger.info(f"Results written to {args.output}")
        else:
            print(output)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

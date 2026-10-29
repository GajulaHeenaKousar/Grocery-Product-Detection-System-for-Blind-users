"""
Test with Image - Test the grocery detection system on a single image file
Usage: python test_with_image.py <path_to_image>
       python test_with_image.py                   (uses a sample from dataset)
"""

import cv2
import sys
import os
import glob
import logging

from config import Config
from barcode_detector import BarcodeDetector
from object_detector import ObjectDetector
from product_info import ProductInfo
from audio_output import AudioOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_image(image_path: str):
    """Test detection on a single image"""
    
    print("=" * 60)
    print("Grocery Product Detection - Image Test")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n✗ Image not found: {image_path}")
        return
    
    print(f"\nTesting image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Could not read image: {image_path}")
        return
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize config
    config = Config()
    
    # Initialize components
    print("\n--- Initializing Components ---")
    
    barcode_detector = BarcodeDetector()
    print("✓ Barcode detector ready")
    
    dataset_yaml_path = os.path.join(config.DATASET_DIR, "data.yaml")
    object_detector = ObjectDetector(
        config.MODEL_PATH,
        config.OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
        dataset_yaml_path=dataset_yaml_path
    )
    print("✓ Object detector ready")
    
    product_info = ProductInfo(config)
    print("✓ Product info module ready")
    
    audio_output = AudioOutput(
        rate=config.AUDIO_RATE,
        voice_index=config.AUDIO_VOICE_INDEX
    )
    print("✓ Audio output ready")
    
    # Step 1: Try barcode detection
    print("\n--- Step 1: Barcode Detection ---")
    barcodes = barcode_detector.detect_barcodes(image)
    
    if barcodes:
        print(f"✓ Found {len(barcodes)} barcode(s)!")
        for i, bc in enumerate(barcodes):
            print(f"  Barcode {i+1}: {bc['data']} (type: {bc['type']})")
        
        best_barcode = barcode_detector.get_best_barcode(barcodes)
        if best_barcode:
            barcode_data = best_barcode['data']
            print(f"\nLooking up barcode: {barcode_data}")
            
            info = product_info.get_product_from_barcode(barcode_data)
            if info and info.get('name') and info['name'] not in ['Unknown Product', '']:
                description = product_info.format_product_description(info)
                print(f"\n✓ PRODUCT FOUND: {description}")
                print(f"  Source: {info.get('source', 'unknown')}")
                
                if config.AUDIO_ENABLED:
                    audio_output.speak(description)
            else:
                print("  Barcode not found in Open Food Facts database")
                # Try LLM fallback
                if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
                    print("  Trying LLM fallback...")
                    info = product_info.get_product_from_llm(
                        barcode_data,
                        f"Barcode scanned: {barcode_data}",
                        detection_type="barcode"
                    )
                    if info:
                        description = product_info.format_product_description(info)
                        print(f"\n✓ PRODUCT (LLM): {description}")
    else:
        print("  No barcodes detected in image")
    
    # Step 2: Object detection
    print("\n--- Step 2: Object Detection ---")
    detections = object_detector.detect_objects(image)
    
    if detections:
        print(f"✓ Found {len(detections)} object(s)!")
        for i, det in enumerate(detections):
            confidence_pct = det['confidence'] * 100
            in_dataset = object_detector.is_in_dataset(det['class_name'])
            dataset_status = "IN dataset" if in_dataset else "NOT in dataset"
            print(f"  {i+1}. {det['class_name']} - {confidence_pct:.1f}% confidence ({dataset_status})")
        
        best = object_detector.get_best_detection(detections)
        if best:
            confidence = best['confidence']
            class_name = best['class_name']
            is_in_dataset = object_detector.is_in_dataset(class_name)
            
            should_use_llm = confidence < config.LLM_FALLBACK_THRESHOLD or not is_in_dataset
            
            if should_use_llm and (config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY):
                print(f"\n  Using LLM for better product info (confidence: {confidence*100:.1f}%)...")
                info = product_info.get_product_from_llm(
                    class_name,
                    f"Detected as '{class_name}' with {confidence*100:.1f}% confidence.",
                    detection_type="object"
                )
                if info:
                    description = product_info.format_product_description(info)
                    print(f"\n✓ PRODUCT (LLM): {description}")
            else:
                description = f"Detected: {class_name} ({confidence*100:.1f}% confidence)"
                print(f"\n✓ PRODUCT: {description}")
            
            if config.AUDIO_ENABLED:
                audio_output.speak(f"Detected {class_name}")
        
        # Draw detections and show image
        output_image = object_detector.draw_detections(image, detections)
        
        # Save annotated image
        output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
        cv2.imwrite(output_path, output_image)
        print(f"\n✓ Annotated image saved to: {output_path}")
        
        # Show image (press any key to close)
        cv2.imshow("Detection Result", output_image)
        print("\nPress any key on the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("  No objects detected in image")
        print("  Tip: Try with a clearer image of a grocery product")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


def find_sample_images():
    """Find sample images from the dataset for testing"""
    sample_dirs = [
        "datasets/test/images",
        "datasets/valid/images",
        "datasets/train/images",
    ]
    
    for dir_path in sample_dirs:
        if os.path.exists(dir_path):
            images = glob.glob(os.path.join(dir_path, "*.jpg")) + \
                     glob.glob(os.path.join(dir_path, "*.png"))
            if images:
                return images[:5]  # Return first 5 sample images
    
    return []


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with provided image
        test_image(sys.argv[1])
    else:
        # Try to use a sample from the dataset
        samples = find_sample_images()
        if samples:
            print("No image path provided. Found sample images from dataset:")
            for i, img in enumerate(samples):
                print(f"  {i+1}. {img}")
            print(f"\nTesting with first sample: {samples[0]}")
            print("(You can also run: python test_with_image.py <your_image_path>)\n")
            test_image(samples[0])
        else:
            print("Usage: python test_with_image.py <path_to_image>")
            print("\nExample:")
            print("  python test_with_image.py my_product_photo.jpg")
            print("  python test_with_image.py C:\\Users\\karti\\Pictures\\grocery.jpg")

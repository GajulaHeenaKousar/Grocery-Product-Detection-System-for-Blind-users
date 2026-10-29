"""
Main Application
Grocery Product Detection System for Blind Users
"""

import cv2
import numpy as np
import logging
import time
import os
from typing import Optional

from config import Config
from barcode_detector import BarcodeDetector
from object_detector import ObjectDetector
from product_info import ProductInfo
from audio_output import AudioOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroceryDetectionSystem:
    """Main system for grocery product detection"""

    def __init__(self):
        """Initialize the detection system"""
        self.config = Config()

        # Initialize components
        self.barcode_detector = BarcodeDetector()

        # Initialize object detector with dataset YAML path for validation
        dataset_yaml_path = os.path.join(self.config.DATASET_DIR, "data.yaml")
        self.object_detector = ObjectDetector(
            self.config.MODEL_PATH,
            self.config.OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
            dataset_yaml_path=dataset_yaml_path
        )

        self.product_info = ProductInfo(self.config)
        self.audio_output = AudioOutput(
            rate=self.config.AUDIO_RATE,
            voice_index=self.config.AUDIO_VOICE_INDEX
        )

        # State management
        self.last_detection_time = 0
        self.detection_cooldown = 3  # seconds between detections
        self.last_product = None

        logger.info("Grocery Detection System initialized")

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Process a single frame and return product description

        Args:
            frame: Camera frame

        Returns:
            Product description string or None
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_detection_time < self.detection_cooldown:
            return None

        # Step 1: Try barcode detection first
        barcodes = self.barcode_detector.detect_barcodes(frame)

        if barcodes:
            best_barcode = self.barcode_detector.get_best_barcode(barcodes)
            if best_barcode:
                barcode_data = best_barcode['data']
                logger.info(f"Barcode detected: {barcode_data}")

                # Get product info from Open Food Facts API
                product_info = self.product_info.get_product_from_barcode(
                    barcode_data)

                # If API call failed or returned no/incomplete info, use LLM
                if not product_info or not product_info.get('name') or product_info.get('name') in ['Unknown Product', '']:
                    logger.info(
                        "Barcode API query failed or returned incomplete info, using LLM fallback...")
                    product_info = self.product_info.get_product_from_llm(
                        barcode_data,
                        f"Barcode scanned: {barcode_data}. Product not found in Open Food Facts database.",
                        detection_type="barcode"
                    )

                if product_info:
                    description = self.product_info.format_product_description(
                        product_info)
                    self.last_detection_time = current_time
                    self.last_product = description
                    logger.info(f"Product description: {description}")
                    return description
                else:
                    logger.warning(
                        f"Could not get product info for barcode: {barcode_data}")
                    return None

        # Step 2: If no barcode, try object detection
        detections = self.object_detector.detect_objects(frame)

        if detections:
            best_detection = self.object_detector.get_best_detection(
                detections)

            if best_detection:
                confidence = best_detection['confidence']
                class_name = best_detection['class_name']

                confidence_percent = confidence * 100
                logger.info(
                    f"Object detected: {class_name} (confidence: {confidence_percent:.1f}%)")

                # Check if product is in dataset
                is_in_dataset = self.object_detector.is_in_dataset(class_name)

                # Use LLM fallback if:
                # 1. Confidence is below 60% (LLM_FALLBACK_THRESHOLD)
                # 2. Product is detected but not in training dataset
                should_use_llm = confidence < self.config.LLM_FALLBACK_THRESHOLD or not is_in_dataset

                if should_use_llm:
                    if confidence < self.config.LLM_FALLBACK_THRESHOLD:
                        logger.info(
                            f"Low confidence ({confidence_percent:.1f}% < 60%), using LLM fallback...")
                        context = f"Detected as '{class_name}' with {confidence_percent:.1f}% confidence. This is below the 60% threshold."
                    else:
                        logger.info(
                            f"Product '{class_name}' is not in training dataset, using LLM fallback...")
                        context = f"Detected as '{class_name}' with {confidence_percent:.1f}% confidence. This product is not in the training dataset."

                    product_info = self.product_info.get_product_from_llm(
                        class_name,
                        context,
                        detection_type="object"
                    )

                    if product_info:
                        description = self.product_info.format_product_description(
                            product_info)
                        self.last_detection_time = current_time
                        self.last_product = description
                        logger.info(
                            f"Product description (LLM): {description}")
                        return description

                # Use detection result directly (high confidence and in dataset)
                product_info = {
                    'name': class_name,
                    'brand': '',
                    'quantity': '',
                    'description': f"Detected {class_name}",
                    'source': 'object_detection'
                }

                description = self.product_info.format_product_description(
                    product_info)
                self.last_detection_time = current_time
                self.last_product = description
                logger.info(f"Product description (direct): {description}")
                return description

        return None

    def run(self):
        """Run the main detection loop"""
        # Initialize camera
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)

        if not cap.isOpened():
            logger.error("Error: Could not open camera")
            return

        logger.info(
            "Camera initialized. Press 'q' to quit, 's' to speak last detection")
        logger.info("Point the camera at a product and wait for detection...")

        # Initial instruction
        if self.config.AUDIO_ENABLED:
            self.audio_output.speak(
                "Grocery detection system ready. Point the camera at a product.")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    logger.error("Error reading frame from camera")
                    break

                # Process frame
                description = self.process_frame(frame)

                # Draw visual feedback (for debugging, not visible to blind users)
                display_frame = frame.copy()

                # Show last product on screen
                if self.last_product:
                    cv2.putText(
                        display_frame,
                        f"Last: {self.last_product}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                # Show instructions
                cv2.putText(
                    display_frame,
                    "Press 'q' to quit, 's' to speak last detection",
                    (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                # Display frame
                cv2.imshow('Grocery Detection System', display_frame)

                # Speak description if detected
                if description and self.config.AUDIO_ENABLED:
                    self.audio_output.speak(description)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s') and self.last_product:
                    # Repeat last detection
                    if self.config.AUDIO_ENABLED:
                        self.audio_output.speak(self.last_product)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("System shut down")


def main():
    """Main entry point"""
    print("="*60)
    print("Grocery Product Detection System for Blind Users")
    print("="*60)
    print("\nInstructions:")
    print("1. Point the camera at a product")
    print("2. The system will detect barcodes first, then objects")
    print("3. Audio feedback will be provided automatically")
    print("4. Press 'q' to quit, 's' to repeat last detection")
    print("\nStarting system...\n")

    system = GroceryDetectionSystem()
    system.run()


if __name__ == "__main__":
    main()

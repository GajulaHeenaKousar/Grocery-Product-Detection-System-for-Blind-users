"""
main1.py - Grocery Product Detection (Camera Mode)
Uses the pre-trained model (no retraining needed).
Connect your phone camera to laptop and point at products.

Usage: python main1.py
       python main1.py --camera 1       (use camera index 1)
       python main1.py --camera 0       (use camera index 0 - default)
"""

import cv2
import numpy as np
import logging
import time
import os
import sys
from typing import Optional

from config import Config
from barcode_detector import BarcodeDetector
from object_detector import ObjectDetector
from product_info import ProductInfo
from audio_output import AudioOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroceryDetector:
    """Live camera-based grocery product detection using pre-trained model"""

    def __init__(self, camera_index=None):
        self.config = Config()

        # Override camera index if provided
        if camera_index is not None:
            self.config.CAMERA_INDEX = camera_index

        # Verify the trained model exists
        if not os.path.exists(self.config.MODEL_PATH):
            print(f"\n[ERROR] Trained model not found at: {self.config.MODEL_PATH}")
            print("This project uses a pre-trained model. Make sure 'models/grocery_detector.pt' exists.")
            sys.exit(1)

        print("=" * 60)
        print("  Grocery Product Detection System for Blind Users")
        print("  Using pre-trained model (no retraining needed)")
        print("=" * 60)

        # Initialize components
        print("\nInitializing components...")

        self.barcode_detector = BarcodeDetector()
        print("  [OK] Barcode detector ready")

        dataset_yaml_path = os.path.join(self.config.DATASET_DIR, "data.yaml")
        self.object_detector = ObjectDetector(
            self.config.MODEL_PATH,
            self.config.OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
            dataset_yaml_path=dataset_yaml_path
        )
        print(f"  [OK] Object detector ready ({len(self.object_detector.class_names)} product classes)")

        self.product_info = ProductInfo(self.config)
        llm_status = "Gemini" if self.config.GEMINI_API_KEY else "No LLM key"
        print(f"  [OK] Product info module ready (LLM: {llm_status})")

        self.audio_output = AudioOutput(
            rate=self.config.AUDIO_RATE,
            voice_index=self.config.AUDIO_VOICE_INDEX
        )
        print("  [OK] Audio output ready")

        # Show what products can be detected
        print(f"\nTrained to detect {len(self.object_detector.class_names)} products:")
        for i, name in self.object_detector.class_names.items():
            print(f"  {i+1:2d}. {name}")

        # State
        self.last_detection_time = 0
        self.detection_cooldown = 3  # seconds between announcements
        self.last_product = None

        print("\n" + "=" * 60)

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """Process a single camera frame and return product description"""
        current_time = time.time()

        # Cooldown to avoid repeating the same product
        if current_time - self.last_detection_time < self.detection_cooldown:
            return None

        # --- Step 1: Try barcode detection first ---
        barcodes = self.barcode_detector.detect_barcodes(frame)
        if barcodes:
            best_barcode = self.barcode_detector.get_best_barcode(barcodes)
            if best_barcode:
                barcode_data = best_barcode['data']
                logger.info(f"Barcode detected: {barcode_data}")

                # Look up barcode in Open Food Facts
                product = self.product_info.get_product_from_barcode(barcode_data)

                # If not found, try LLM
                if not product or not product.get('name') or product.get('name') in ['Unknown Product', '']:
                    logger.info("Barcode not in database, trying LLM...")
                    product = self.product_info.get_product_from_llm(
                        barcode_data,
                        f"Barcode scanned: {barcode_data}. Not found in Open Food Facts.",
                        detection_type="barcode"
                    )

                if product:
                    description = self.product_info.format_product_description(product)
                    self.last_detection_time = current_time
                    self.last_product = description
                    return description

        # --- Step 2: Object detection with YOLO ---
        detections = self.object_detector.detect_objects(frame)
        if detections:
            best = self.object_detector.get_best_detection(detections)
            if best:
                confidence = best['confidence']
                class_name = best['class_name']
                confidence_pct = confidence * 100

                logger.info(f"Detected: {class_name} ({confidence_pct:.1f}%)")

                is_in_dataset = self.object_detector.is_in_dataset(class_name)
                needs_llm = confidence < self.config.LLM_FALLBACK_THRESHOLD or not is_in_dataset

                if needs_llm:
                    reason = "low confidence" if confidence < self.config.LLM_FALLBACK_THRESHOLD else "not in dataset"
                    logger.info(f"Using LLM fallback ({reason})...")

                    product = self.product_info.get_product_from_llm(
                        class_name,
                        f"Detected as '{class_name}' with {confidence_pct:.1f}% confidence.",
                        detection_type="object"
                    )
                    if product:
                        description = self.product_info.format_product_description(product)
                        self.last_detection_time = current_time
                        self.last_product = description
                        return description

                # High confidence + in dataset → use directly
                description = f"{class_name}"
                self.last_detection_time = current_time
                self.last_product = description
                return description

        return None

    def run(self):
        """Run live camera detection loop"""
        camera_idx = self.config.CAMERA_INDEX
        print(f"\nOpening camera (index {camera_idx})...")
        print("If using phone camera via DroidCam/IVCam, try index 1 or 2")

        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)

        if not cap.isOpened():
            print(f"\n[ERROR] Could not open camera at index {camera_idx}")
            print("Try running: python main1.py --camera 1")
            print("Or try:      python main1.py --camera 2")
            return

        print("\n[READY] Camera is live!")
        print("  - Point the camera at a grocery product")
        print("  - Press 'q' to quit")
        print("  - Press 's' to repeat last detection")
        print("  - Press 'c' to switch camera (cycles 0→1→2→0)")
        print()

        if self.config.AUDIO_ENABLED:
            self.audio_output.speak("Grocery detection system ready. Point the camera at a product.")

        current_camera = camera_idx

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Cannot read from camera")
                    break

                # Process frame for detection
                description = self.process_frame(frame)

                # Draw UI overlay on the display frame
                display = frame.copy()

                # Status bar at top
                cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)

                if self.last_product:
                    # Truncate long product names for display
                    display_text = self.last_product[:60] + "..." if len(self.last_product) > 60 else self.last_product
                    cv2.putText(display, f"Product: {display_text}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "Scanning... Point at a product", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Help text at bottom
                cv2.rectangle(display, (0, display.shape[0] - 30), (display.shape[1], display.shape[0]), (0, 0, 0), -1)
                cv2.putText(display, "q=Quit | s=Repeat | c=Switch Camera", (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                cv2.imshow('Grocery Detection', display)

                # Speak the detection
                if description and self.config.AUDIO_ENABLED:
                    print(f"\n  >> DETECTED: {description}")
                    self.audio_output.speak(description)

                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s') and self.last_product:
                    print(f"\n  >> REPEAT: {self.last_product}")
                    if self.config.AUDIO_ENABLED:
                        self.audio_output.speak(self.last_product)
                elif key == ord('c'):
                    # Cycle through cameras 0, 1, 2
                    cap.release()
                    current_camera = (current_camera + 1) % 3
                    print(f"\n  Switching to camera {current_camera}...")
                    cap = cv2.VideoCapture(current_camera)
                    if not cap.isOpened():
                        print(f"  Camera {current_camera} not available, trying next...")
                        current_camera = (current_camera + 1) % 3
                        cap = cv2.VideoCapture(current_camera)

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nSystem shut down. Goodbye!")


def main():
    # Parse --camera argument
    camera_index = None
    if "--camera" in sys.argv:
        try:
            idx = sys.argv.index("--camera")
            camera_index = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Usage: python main1.py --camera <index>")
            print("  Example: python main1.py --camera 1")
            sys.exit(1)

    detector = GroceryDetector(camera_index=camera_index)
    detector.run()


if __name__ == "__main__":
    main()

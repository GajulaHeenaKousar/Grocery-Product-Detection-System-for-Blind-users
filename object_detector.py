"""
Object Detection Module
Uses YOLO for detecting grocery products when barcode is not available
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Dict, Optional, Tuple, Set
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """Detects grocery products using YOLO object detection"""

    def __init__(self, model_path: str, confidence_threshold: float = 0.4, dataset_yaml_path: str = None):
        """
        Initialize object detector

        Args:
            model_path: Path to YOLO model file (.pt)
            confidence_threshold: Minimum confidence for detections
            dataset_yaml_path: Path to dataset YAML file to load valid class names
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = {}
        self.dataset_classes = set()  # Valid classes from dataset

        self._load_model()
        self._load_dataset_classes(dataset_yaml_path)

    def _load_model(self):
        """Load YOLO model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.warning(
                    f"Model not found at {self.model_path}. Using default YOLOv8n model.")
                # Use a pre-trained YOLO model as fallback
                self.model = YOLO('yolov8n.pt')
                logger.info(
                    "Using default YOLOv8n model. Train a custom model for better grocery detection.")

            # Get class names if available
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names

            logger.info(f"Loaded model with {len(self.class_names)} classes")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to default model
            self.model = YOLO('yolov8n.pt')

    def _load_dataset_classes(self, dataset_yaml_path: str = None):
        """Load valid dataset classes from YAML file"""
        try:
            if dataset_yaml_path is None:
                # Try to find dataset YAML in common locations
                possible_paths = [
                    "datasets/data.yaml",
                    "datasets/grocery-dataset-q9fj2/data.yaml",
                    os.path.join(os.path.dirname(self.model_path),
                                 "..", "datasets", "data.yaml")
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        dataset_yaml_path = path
                        break

            if dataset_yaml_path and os.path.exists(dataset_yaml_path):
                with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                if 'names' in data:
                    # Convert list to set for faster lookup
                    if isinstance(data['names'], list):
                        self.dataset_classes = set(data['names'])
                    elif isinstance(data['names'], dict):
                        self.dataset_classes = set(data['names'].values())

                    logger.info(
                        f"Loaded {len(self.dataset_classes)} dataset classes from {dataset_yaml_path}")
                else:
                    logger.warning(
                        f"No 'names' field found in {dataset_yaml_path}")
            else:
                logger.warning(
                    "Dataset YAML file not found. Cannot validate if products are in dataset.")

        except Exception as e:
            logger.warning(f"Error loading dataset classes: {str(e)}")
            logger.warning(
                "Will not be able to detect out-of-dataset products")

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in an image

        Args:
            image: Input image as numpy array

        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        try:
            if self.model is None:
                return []

            # Run inference
            results = self.model(
                image, conf=self.confidence_threshold, verbose=False)

            detections = []

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Get class name
                    class_name = self.class_names.get(
                        class_id, f"Class_{class_id}") if self.class_names else f"Class_{class_id}"

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': {
                            'x': int(x1),
                            'y': int(y1),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        }
                    })

            return detections

        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []

    def get_best_detection(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Get the detection with highest confidence

        Args:
            detections: List of detections

        Returns:
            Best detection or None
        """
        if not detections:
            return None

        # Sort by confidence and return the best one
        sorted_detections = sorted(
            detections, key=lambda x: x['confidence'], reverse=True)
        return sorted_detections[0]

    def is_in_dataset(self, class_name: str) -> bool:
        """
        Check if a detected class is in the training dataset

        Args:
            class_name: Name of the detected class

        Returns:
            True if class is in dataset, False otherwise
        """
        if not self.dataset_classes:
            # If dataset classes not loaded, assume it's in dataset
            return True

        # Check if class name is in dataset (case-insensitive)
        class_name_lower = class_name.lower()
        for dataset_class in self.dataset_classes:
            if class_name_lower in dataset_class.lower() or dataset_class.lower() in class_name_lower:
                return True

        return False

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image

        Args:
            image: Input image
            detections: List of detections

        Returns:
            Image with drawn detections
        """
        output = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Draw rectangle
            cv2.rectangle(
                output,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                (255, 0, 0),
                2
            )

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                output,
                label,
                (bbox['x'], bbox['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        return output

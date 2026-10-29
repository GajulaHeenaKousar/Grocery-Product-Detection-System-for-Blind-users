"""
Barcode Detection Module
Uses OpenCV and pyzbar to detect and decode barcodes from camera input
"""

import cv2
import numpy as np
from pyzbar import pyzbar
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarcodeDetector:
    """Detects and decodes barcodes from images"""
    
    def __init__(self):
        self.supported_formats = ['EAN_13', 'EAN_8', 'UPC_A', 'UPC_E', 'CODE_128', 'CODE_39']
    
    def detect_barcodes(self, image: np.ndarray) -> List[dict]:
        """
        Detect barcodes in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected barcodes with their data and positions
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect barcodes
            barcodes = pyzbar.decode(gray)
            
            results = []
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                
                # Get barcode position
                points = barcode.polygon
                if len(points) == 4:
                    # Rectangle coordinates
                    x_coords = [p.x for p in points]
                    y_coords = [p.y for p in points]
                    bbox = {
                        'x': min(x_coords),
                        'y': min(y_coords),
                        'width': max(x_coords) - min(x_coords),
                        'height': max(y_coords) - min(y_coords)
                    }
                else:
                    bbox = None
                
                results.append({
                    'data': barcode_data,
                    'type': barcode_type,
                    'bbox': bbox,
                    'confidence': 1.0  # pyzbar doesn't provide confidence, assume high
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting barcodes: {str(e)}")
            return []
    
    def draw_barcodes(self, image: np.ndarray, barcodes: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes around detected barcodes
        
        Args:
            image: Input image
            barcodes: List of detected barcodes
            
        Returns:
            Image with drawn bounding boxes
        """
        output = image.copy()
        
        for barcode in barcodes:
            if barcode['bbox']:
                bbox = barcode['bbox']
                cv2.rectangle(
                    output,
                    (bbox['x'], bbox['y']),
                    (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                    (0, 255, 0),
                    2
                )
                # Add text
                cv2.putText(
                    output,
                    f"{barcode['type']}: {barcode['data']}",
                    (bbox['x'], bbox['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        return output
    
    def get_best_barcode(self, barcodes: List[dict]) -> Optional[dict]:
        """
        Get the most reliable barcode from detected barcodes
        
        Args:
            barcodes: List of detected barcodes
            
        Returns:
            Best barcode or None
        """
        if not barcodes:
            return None
        
        # Prefer EAN/UPC codes for grocery products
        preferred_types = ['EAN_13', 'EAN_8', 'UPC_A', 'UPC_E']
        
        for barcode in barcodes:
            if barcode['type'] in preferred_types:
                return barcode
        
        # Return first barcode if no preferred type found
        return barcodes[0]


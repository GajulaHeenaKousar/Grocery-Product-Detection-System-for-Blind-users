"""
Model Training Script
Trains YOLO model on the grocery dataset

Optimized for RTX 3050 (6GB VRAM) with 2-day deadline:
- 15 epochs: Good balance - leaves time for testing and bug fixes
- Batch size 8: Fits in 6GB VRAM comfortably
- Early stopping: Saves time if model converges early
- GPU acceleration: YOLO automatically uses GPU if CUDA is available

Expected training time: 20-40 minutes with RTX 3050
"""

import os
import sys
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model():
    """Train YOLO model on grocery dataset"""

    # Check if dataset exists
    dataset_path = "datasets/data.yaml"

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        logger.error(
            "Please run 'python setup_dataset.py' first to download the dataset")
        return False

    # Create models directory
    os.makedirs("models", exist_ok=True)

    try:
        # Initialize YOLO model (using YOLOv8n for faster training, can use YOLOv8s, YOLOv8m, etc.)
        logger.info("Initializing YOLOv8 model...")
        model = YOLO('yolov8n.pt')  # Start with pre-trained weights

        # Check if GPU is available
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("No GPU detected - training will be slower on CPU")

        # Train the model
        logger.info("Starting training...")
        logger.info(
            "Training with 15 epochs (optimized for RTX 3050 + 2-day deadline)")
        logger.info("Expected time: 20-40 minutes with GPU, 1-2 hours on CPU")
        logger.info("Early stopping will save time if model converges early")

        # Training parameters optimized for RTX 3050 (6GB VRAM) and 2-day deadline
        # - 15 epochs: Perfect balance - fast enough but still good accuracy
        # - Batch size 8: Fits comfortably in 6GB VRAM (reduce to 4 if OOM errors)
        # - Early stopping (patience=7): Stops if no improvement for 7 epochs
        # - YOLO automatically uses GPU if CUDA is available
        results = model.train(
            data=dataset_path,
            epochs=15,   # Optimized for 2-day deadline - leaves time for testing
            imgsz=640,   # Image size
            # Optimized for 6GB VRAM (reduce to 4 if out of memory)
            batch=8,
            name='grocery_detector',
            project='models',
            save=True,
            # Early stopping: stops if no improvement for 7 epochs (saves time)
            patience=7,
            device=0 if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        )

        # Save the trained model
        model_path = "models/grocery_detector.pt"
        model.save(model_path)

        logger.info(f"\nTraining completed!")
        logger.info(f"Model saved to: {model_path}")
        logger.info("\nYou can now use this model for object detection.")

        return True

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Ensure dataset is properly downloaded")
        logger.error("2. Check that data.yaml file exists and is valid")
        logger.error(
            "3. If you get 'out of memory' error, reduce batch size to 4")
        logger.error(
            "4. Ensure PyTorch with CUDA is installed for GPU acceleration")
        logger.error("5. Check GPU drivers are up to date")
        return False


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)

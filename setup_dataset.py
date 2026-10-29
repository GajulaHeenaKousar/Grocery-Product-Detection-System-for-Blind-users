"""
Dataset Setup Script
Downloads and sets up the grocery dataset from Roboflow
"""

import os
import sys
from roboflow import Roboflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dataset():
    """Download and set up the grocery dataset from Roboflow"""
    
    # Get API key
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if not api_key:
        print("\n" + "="*60)
        print("Roboflow API Key Required")
        print("="*60)
        print("\nTo get your API key:")
        print("1. Sign up for a free account at https://roboflow.com")
        print("2. Go to your account settings")
        print("3. Copy your API key")
        print("\nYou can set it as an environment variable:")
        print("  Windows: set ROBOFLOW_API_KEY=your_key_here")
        print("  Linux/Mac: export ROBOFLOW_API_KEY=your_key_here")
        print("\nOr enter it now:")
        api_key = input("Enter your Roboflow API key: ").strip()
        
        if not api_key:
            logger.error("API key is required to download the dataset")
            return False
    
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        
        # Get the project
        workspace = "new-workspace-wfzw3"
        project_name = "grocery-dataset-q9fj2"
        
        logger.info(f"Accessing project: {project_name}")
        project = rf.workspace(workspace).project(project_name)
        
        # Get the latest version
        logger.info("Getting dataset version...")
        dataset = project.version(1)  # Use version 1, update if needed
        
        # Create datasets directory
        dataset_dir = "datasets"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Download dataset in YOLOv8 format
        logger.info("Downloading dataset in YOLOv8 format...")
        logger.info("This may take a few minutes depending on dataset size...")
        
        dataset.download("yolov8", location=dataset_dir)
        
        logger.info(f"Dataset downloaded successfully to {dataset_dir}/")
        logger.info("\nDataset structure:")
        logger.info(f"  {dataset_dir}/{project_name}/")
        logger.info("    ├── train/")
        logger.info("    ├── valid/")
        logger.info("    └── data.yaml")
        
        print("\n" + "="*60)
        print("Dataset Setup Complete!")
        print("="*60)
        print(f"\nDataset location: {dataset_dir}/{project_name}/")
        print("\nYou can now train the model using:")
        print("  python train_model.py")
        print("\nOr use the pre-trained model if available.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up dataset: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check your API key is correct")
        logger.error("2. Ensure you have internet connection")
        logger.error("3. Verify the dataset is publicly available on Roboflow")
        return False


if __name__ == "__main__":
    success = setup_dataset()
    sys.exit(0 if success else 1)


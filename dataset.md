# Grocery Dataset Setup Guide

This guide explains how to download and set up the grocery dataset from Roboflow for the Grocery Product Detection System.

## Dataset Source
- **Roboflow Universe**: https://universe.roboflow.com/new-workspace-wfzw3/grocery-dataset-q9fj2
- **Dataset Name**: grocery-dataset-q9fj2

## Download Options

### Option 1: Using Roboflow Python SDK (Recommended)

The dataset can be downloaded programmatically using the Roboflow Python SDK. This is the easiest method and is already integrated into the project.

**Steps:**
1. Sign up for a free Roboflow account at https://roboflow.com
2. Get your API key from your Roboflow account settings
3. The dataset will be automatically downloaded when you run the setup script

**Format to Download:**
- **Format**: YOLOv8 (recommended for YOLO models)
- **Version**: Latest version (usually version 1)
- **Export Format**: YOLOv8 PyTorch format

### Option 2: Manual Download from Roboflow Website

1. Visit the dataset page: https://universe.roboflow.com/new-workspace-wfzw3/grocery-dataset-q9fj2
2. Click on "Download" button
3. Select format: **YOLOv8** (for YOLO models) or **COCO JSON** (for other frameworks)
4. Download the dataset zip file
5. Extract it to the `datasets/` directory in the project root

## Dataset Structure

After downloading, the dataset should have the following structure:

```
datasets/
└── grocery-dataset-q9fj2/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml (or dataset.yaml)
```

## Classification Information

The dataset includes various grocery items. The exact classes depend on the dataset version, but typically includes:
- Fruits (apples, bananas, oranges, etc.)
- Vegetables (tomatoes, carrots, onions, etc.)
- Packaged goods (cereal boxes, canned goods, etc.)
- Dairy products
- Beverages

The `data.yaml` file contains:
- Number of classes
- Class names
- Paths to train/validation images and labels

## Setup Instructions

### Automatic Setup (Using Python Script)

Run the setup script:
```bash
python setup_dataset.py
```

This script will:
1. Prompt for your Roboflow API key
2. Download the dataset in YOLOv8 format
3. Organize it in the correct directory structure
4. Verify the dataset is ready for training

### Manual Setup

1. Download the dataset manually from Roboflow
2. Extract to `datasets/grocery-dataset-q9fj2/`
3. Ensure the structure matches the expected format above
4. Update the path in `config.py` if needed

## Training the Model

After the dataset is set up, you can train the YOLO model:

```bash
python train_model.py
```

The trained model will be saved in `models/grocery_detector.pt`

## Notes

- **No Need to Download Separately**: If you use the automatic setup script, the dataset is downloaded and configured automatically
- **Format**: YOLOv8 format is recommended as it's compatible with Ultralytics YOLO
- **Size**: The dataset size varies, but typically ranges from 100MB to several GB depending on the number of images
- **Updates**: If the dataset is updated on Roboflow, you can re-download it using the same script

## Troubleshooting

1. **API Key Issues**: Make sure your Roboflow API key is valid and has access to the dataset
2. **Path Issues**: Ensure the dataset is extracted to the correct directory
3. **Format Issues**: Verify you downloaded the YOLOv8 format, not COCO or other formats
4. **Missing Files**: Check that `data.yaml` exists in the dataset root directory


# Grocery Product Detection System for Blind Users

A computer vision system that helps blind users shop independently by detecting grocery products through camera input and providing audio feedback.

## Features

- **Barcode Detection**: Automatically detects and reads barcodes from packaged goods using OpenCV and pyzbar
- **Object Detection**: Recognizes products visually using YOLO when barcodes are not available
- **Product Information**: Retrieves detailed product information from Open Food Facts API
- **LLM Fallback**: Uses free LLM APIs when product details are limited or confidence is low
- **Audio Output**: Provides real-time voice feedback describing detected products
- **Real-time Processing**: Continuous detection from webcam feed

## System Requirements

- Python 3.8 or higher
- Webcam/Camera
- Windows/Linux/Mac OS
- Internet connection (for API calls)

## Installation

### 1. Clone or Download the Project

```bash
cd project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Windows users**: You may need to install additional dependencies for pyzbar:
- Install Visual C++ Redistributable
- Or use: `pip install pyzbar[scripts]`

### 3. Set Up the Dataset (Optional - for training custom model)

If you want to train a custom model on the grocery dataset:

```bash
# Set your Roboflow API key (get it from https://roboflow.com)
set ROBOFLOW_API_KEY=your_api_key_here  # Windows
# export ROBOFLOW_API_KEY=your_api_key_here  # Linux/Mac

# Download the dataset
python setup_dataset.py

# Train the model (this may take several hours)
python train_model.py
```

**Note**: You can skip training if you want to use a pre-trained model. The system will use a default YOLO model, though it may be less accurate for grocery-specific items.

### 4. Configure API Keys (Optional)

For LLM fallback features, you can optionally set API keys:

```bash
# For OpenAI (if using)
set OPENAI_API_KEY=your_key_here

# For Anthropic Claude (if using)
set ANTHROPIC_API_KEY=your_key_here
```

**Note**: The system works without LLM keys, but will have limited fallback capabilities.

## Usage

### Basic Usage

1. **Start the system**:
   ```bash
   python main.py
   ```

2. **Point the camera at a product**:
   - The system will first try to detect a barcode
   - If no barcode is found, it will use object detection
   - Audio feedback will be provided automatically

3. **Controls**:
   - `q`: Quit the application
   - `s`: Repeat the last detected product description

### How It Works

1. **Barcode Detection** (Primary Method):
   - Scans for barcodes in the camera feed
   - Decodes barcode (EAN, UPC, etc.)
   - Fetches product information from Open Food Facts API
   - If limited info, uses LLM to get more details

2. **Object Detection** (Fallback Method):
   - Uses YOLO model to detect products visually
   - If confidence is below 40%, uses LLM for better description
   - Provides product name and description

3. **Audio Output**:
   - Converts product information to speech
   - Provides real-time feedback
   - Example: "Coca Cola can, 250ml soft drink" or "Apple, fresh fruit"

## Project Structure

```
project/
├── main.py                 # Main application entry point
├── config.py               # Configuration settings
├── barcode_detector.py     # Barcode detection module
├── object_detector.py      # Object detection module
├── product_info.py         # Product information retrieval
├── audio_output.py         # Text-to-speech module
├── setup_dataset.py        # Dataset download script
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── dataset.md              # Dataset setup guide
├── README.md              # This file
├── models/                 # Trained models (created after training)
└── datasets/              # Dataset files (created after download)
```

## Configuration

Edit `config.py` to customize:

- Camera settings (index, resolution)
- Detection thresholds
- Audio settings (rate, voice)
- API keys and providers
- Model paths

## Troubleshooting

### Camera Not Working
- Check if camera is connected and not used by another application
- Try changing `CAMERA_INDEX` in `config.py` (try 0, 1, 2, etc.)

### Barcode Not Detected
- Ensure good lighting
- Hold the product steady
- Make sure barcode is clearly visible and not damaged
- Try different angles

### Object Detection Not Working
- Train a custom model using `train_model.py` for better accuracy
- Ensure model file exists at `models/grocery_detector.pt`
- Check that dataset is properly downloaded

### Audio Not Working
- On Linux, you may need: `sudo apt-get install espeak`
- On Mac, TTS should work out of the box
- On Windows, ensure you have SAPI5 voices installed

### API Errors
- Check internet connection
- Verify API keys are set correctly (if using LLM features)
- Open Food Facts API is free and doesn't require a key

## Dataset Information

The system uses the grocery dataset from Roboflow:
- **Source**: https://universe.roboflow.com/new-workspace-wfzw3/grocery-dataset-q9fj2
- **Format**: YOLOv8
- **See**: `dataset.md` for detailed setup instructions

## Future Improvements

- Support for multiple product detection in one frame
- Product price detection
- Nutritional information audio output
- Offline mode with local LLM
- Mobile app version
- Integration with shopping list apps

## License

This project is open source and available for educational and personal use.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the dataset setup guide (`dataset.md`)
3. Ensure all dependencies are installed correctly

## Acknowledgments

- Open Food Facts API for product information
- Roboflow for the grocery dataset
- Ultralytics for YOLO implementation
- OpenCV and pyzbar communities


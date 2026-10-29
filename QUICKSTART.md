# Quick Start Guide

Get up and running with the Grocery Product Detection System in minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Windows users**: If you encounter issues with `pyzbar`, you may need to install:
- Visual C++ Redistributable
- Or use: `pip install pyzbar[scripts]`

## Step 2: Run the Application

```bash
python main.py
```

That's it! The system will:
- Open your webcam
- Detect products (barcodes first, then objects)
- Provide audio feedback

## Optional: Set Up Custom Model

For better object detection accuracy on grocery items:

### 2a. Download Dataset

```bash
# Get your free API key from https://roboflow.com
set ROBOFLOW_API_KEY=your_key_here  # Windows
# export ROBOFLOW_API_KEY=your_key_here  # Linux/Mac

python setup_dataset.py
```

### 2b. Train Model

```bash
python train_model.py
```

**Note**: Training can take several hours depending on your hardware. You can use the system without training - it will use a default YOLO model.

## Optional: Configure LLM (for better product descriptions)

If you want enhanced product descriptions when details are missing:

```bash
# For OpenAI
set OPENAI_API_KEY=your_key_here

# OR for Anthropic
set ANTHROPIC_API_KEY=your_key_here
```

Then edit `config.py` to set `LLM_PROVIDER = "openai"` or `"anthropic"`.

## Usage Tips

1. **Point the camera** at a product and hold steady
2. **Wait 2-3 seconds** for detection
3. **Good lighting** helps with barcode detection
4. **Press 's'** to repeat the last detection
5. **Press 'q'** to quit

## Troubleshooting

### Camera not working?
- Check if another app is using the camera
- Try changing `CAMERA_INDEX` in `config.py` (try 0, 1, 2)

### No audio?
- Windows: Should work automatically
- Linux: Install `espeak`: `sudo apt-get install espeak`
- Mac: Should work automatically

### Barcode not detected?
- Ensure good lighting
- Hold product steady
- Make sure barcode is clearly visible

## Next Steps

- Read `README.md` for detailed documentation
- Check `dataset.md` for dataset setup details
- Customize `config.py` for your preferences


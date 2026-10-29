# Next Steps - After Training Your Model

## ✅ Model Training Complete!

Your model has been trained successfully. Now let's test and run the system.

## Step 1: Run System Test (Recommended)

First, test if everything is set up correctly:

```bash
python test_system.py
```

This will check:

- ✓ All required packages are installed
- ✓ Model file exists and can be loaded
- ✓ Camera is accessible
- ✓ Audio system works
- ✓ API connection is working

## Step 2: Verify Model File

Check if your model file exists:

- Main model: `models/grocery_detector.pt`
- Best model (if available): `models/grocery_detector2/weights/best.pt`

**If you want to use the best model from training:**

On Windows:

```bash
copy models\grocery_detector2\weights\best.pt models\grocery_detector.pt
```

On Linux/Mac:

```bash
cp models/grocery_detector2/weights/best.pt models/grocery_detector.pt
```

## Step 3: Run the Main Application

```bash
python main.py
```

**What happens:**

1. Camera window opens
2. Audio says: "Grocery detection system ready. Point the camera at a product."
3. Point camera at products and wait 2-3 seconds
4. System detects products and provides audio feedback

## Step 4: Test with Products

### Test Barcode Detection (Packaged Products)

- Get products with visible barcodes (canned food, boxes, etc.)
- Point camera at barcode
- Wait 2-3 seconds
- System will fetch product info from Open Food Facts API
- Audio will announce product details

### Test Object Detection (Fruits/Vegetables)

- Get products from your training dataset (apple, banana, etc.)
- Point camera at product
- Wait 2-3 seconds
- System will detect using your trained YOLO model
- Audio will announce detected product

## Step 5: Keyboard Controls

- **'q'**: Quit the application
- **'s'**: Repeat the last detected product description

## Troubleshooting

### Model Not Loading

- Verify model exists: `models/grocery_detector.pt`
- Check file size (should be several MB)
- Try using the `best.pt` model if available

### Camera Not Working

- Check if another app is using the camera
- Change `CAMERA_INDEX` in `config.py` to 1 or 2
- On Windows: Check camera permissions

### No Audio

- Windows: Should work automatically
- Linux: Install espeak: `sudo apt-get install espeak`
- Check `AUDIO_ENABLED = True` in `config.py`

### Low Detection Accuracy

- Ensure good lighting
- Hold products steady for 2-3 seconds
- Make sure products are similar to training data
- Try reducing `OBJECT_DETECTION_CONFIDENCE_THRESHOLD` in `config.py` to 0.3

### Barcode Not Detected

- Ensure good lighting
- Hold product steady
- Make sure barcode is clearly visible
- Try different angles

## Demo Preparation Tips

1. **Prepare test products:**

   - 2-3 products with barcodes
   - 2-3 products without barcodes (from your dataset)

2. **Test in good lighting:**

   - Natural light is best
   - Avoid shadows on products

3. **Practice the demo:**

   - Test all features before presentation
   - Have backup products ready
   - Know the keyboard shortcuts

4. **Documentation:**
   - Screenshots of training results
   - Video recording of system in action
   - Explanation of how it works

## Performance Tips

- Good lighting improves accuracy
- Hold products steady for 2-3 seconds
- Clean camera lens
- Position product in center of frame
- For barcodes: ensure they're flat and well-lit

## Quick Command Reference

```bash
# Test system
python test_system.py

# Run main application
python main.py

# View training results
# Check: models/grocery_detector2/results.png
# Check: models/grocery_detector2/confusion_matrix.png
```

## What's Next?

1. ✅ Test the system with `python test_system.py`
2. ✅ Run the main application with `python main.py`
3. ✅ Test with real products
4. ✅ Prepare your demo
5. ✅ Document your results

Good luck with your assignment! 🚀

# Testing Guide - Grocery Product Detection System

## Step 1: Verify Model File

Your trained model should be at: `models/grocery_detector.pt`

If you see a `models/grocery_detector2/weights/best.pt` file, you can optionally use that (it's usually the best model from training).

## Step 2: Quick Test - Run the Main Application

```bash
python main.py
```

**What to expect:**
- Camera window will open
- System will say: "Grocery detection system ready. Point the camera at a product."
- Point camera at products and wait 2-3 seconds for detection
- Audio feedback will announce detected products

## Step 3: Test Barcode Detection

**Test with packaged products that have barcodes:**
1. Get a product with a visible barcode (canned food, box, etc.)
2. Point camera at the barcode
3. Hold steady for 2-3 seconds
4. System should detect barcode and fetch product info from Open Food Facts API
5. Audio will announce: "Brand Name, Product Name, Quantity, Description"

**Expected output example:**
- "Coca Cola, Coca Cola Can, 330ml, Soft drink"
- "Kellogg's, Corn Flakes, 500g, Breakfast cereal"

## Step 4: Test Object Detection

**Test with products without barcodes (fruits, vegetables, etc.):**
1. Get a product from your dataset (apple, banana, etc.)
2. Point camera at the product
3. Hold steady for 2-3 seconds
4. System should detect the object using your trained YOLO model
5. Audio will announce the detected product

**Expected output example:**
- "Apple, fresh fruit"
- "Banana, yellow fruit"

## Step 5: Test Low Confidence Fallback

If detection confidence is below 40%, the system will use LLM to get better descriptions.

## Step 6: Keyboard Controls

- **'q'**: Quit the application
- **'s'**: Repeat the last detected product description

## Troubleshooting

### Camera Not Opening
- Check if another app is using the camera
- Change `CAMERA_INDEX` in `config.py` to 1 or 2
- On Windows: Check camera permissions in Settings

### No Audio Output
- **Windows**: Should work automatically
- **Linux**: Install espeak: `sudo apt-get install espeak`
- **Mac**: Should work automatically
- Check `AUDIO_ENABLED = True` in `config.py`

### Model Not Loading
- Verify model file exists: `models/grocery_detector.pt`
- Check the path in `config.py`
- If you have `best.pt` in `grocery_detector2/weights/`, you can copy it:
  ```bash
  copy models\grocery_detector2\weights\best.pt models\grocery_detector.pt
  ```

### Barcode Not Detected
- Ensure good lighting
- Hold product steady
- Make sure barcode is clearly visible
- Try different angles

### Object Detection Not Working
- Verify model was trained successfully
- Check that product is similar to items in training dataset
- Ensure good lighting and clear view of product
- Try reducing `OBJECT_DETECTION_CONFIDENCE_THRESHOLD` in `config.py` to 0.3

### No Internet (API calls failing)
- Barcode detection requires internet for Open Food Facts API
- Object detection works offline
- LLM fallback requires internet (if enabled)

## Testing Checklist

- [ ] Camera opens and shows video feed
- [ ] Audio output works (hear initial message)
- [ ] Barcode detection works with packaged products
- [ ] Object detection works with fruits/vegetables
- [ ] Product descriptions are accurate
- [ ] Keyboard controls work ('q' to quit, 's' to repeat)
- [ ] System doesn't crash or freeze

## Demo Preparation

For your assignment demo:

1. **Prepare test products:**
   - 2-3 products with barcodes (canned food, boxes)
   - 2-3 products without barcodes (fruits, vegetables from your dataset)

2. **Test sequence:**
   - Start the application
   - Show barcode detection first
   - Then show object detection
   - Demonstrate audio feedback

3. **Have backup plan:**
   - Screenshots of training results
   - Video recording of the system in action
   - Explanation of how it works

## Performance Tips

- Good lighting improves detection accuracy
- Hold products steady for 2-3 seconds
- Clean camera lens
- Position product in center of frame
- For barcodes: ensure they're flat and well-lit


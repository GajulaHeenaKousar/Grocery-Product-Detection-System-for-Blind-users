# Quick Start Guide - After Training

## 🎉 Your Model is Trained! (16 hours - great job!)

## Step 1: Test Your System (2 minutes)

Run the test script to verify everything works:

```bash
python test_system.py
```

This checks:

- ✓ All packages installed
- ✓ Model file exists
- ✓ Camera works
- ✓ Audio works
- ✓ API connection works

## Step 2: Run the Application (Main Step!)

```bash
python main.py
```

**What happens:**

1. Camera window opens
2. You'll hear: "Grocery detection system ready. Point the camera at a product."
3. Point camera at products
4. Wait 2-3 seconds for detection
5. System announces detected products via audio

## Step 3: Test with Products

### Test 1: Barcode Detection

- Get a product with a barcode (canned food, box, etc.)
- Point camera at barcode
- System fetches product info from Open Food Facts
- Audio announces: "Brand, Product Name, Quantity, Description"

### Test 2: Object Detection

- Get a fruit/vegetable from your dataset
- Point camera at it
- System detects using your trained YOLO model
- Audio announces the product

## Keyboard Controls

- **'q'**: Quit
- **'s'**: Repeat last detection

## Troubleshooting

### Camera Not Working?

- Change `CAMERA_INDEX` in `config.py` to 1 or 2
- Check if another app is using camera

### No Audio?

- Windows: Should work automatically
- Linux: `sudo apt-get install espeak`

### Model Not Found?

- Check: `models/grocery_detector.pt` exists
- If you have `best.pt`, copy it:
  ```bash
  copy models\grocery_detector2\weights\best.pt models\grocery_detector.pt
  ```

### Low Accuracy?

- Ensure good lighting
- Hold products steady
- Reduce threshold in `config.py`: `OBJECT_DETECTION_CONFIDENCE_THRESHOLD = 0.3`

## Demo Tips

1. **Prepare products:**

   - 2-3 items with barcodes
   - 2-3 fruits/vegetables

2. **Good lighting:**

   - Natural light works best
   - Avoid shadows

3. **Practice:**
   - Test before presentation
   - Know the keyboard shortcuts

## Files to Check

- Training results: `models/grocery_detector2/results.png`
- Confusion matrix: `models/grocery_detector2/confusion_matrix.png`
- Model file: `models/grocery_detector.pt`

## Next Steps

1. ✅ Run `python test_system.py`
2. ✅ Run `python main.py`
3. ✅ Test with real products
4. ✅ Prepare your demo
5. ✅ Good luck! 🚀

---

**Need help?** Check `TESTING_GUIDE.md` for detailed testing instructions.

"""
Quick System Test Script
Tests all components before running the main application
"""

import os
import sys


def test_imports():
    """Test if all required modules can be imported"""
    print("="*60)
    print("Testing Imports...")
    print("="*60)

    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        from pyzbar import pyzbar
        print("✓ pyzbar imported successfully")
    except ImportError as e:
        print(f"✗ pyzbar import failed: {e}")
        return False

    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False

    try:
        import pyttsx3
        print("✓ pyttsx3 (TTS) imported successfully")
    except ImportError as e:
        print(f"✗ pyttsx3 import failed: {e}")
        return False

    try:
        import requests
        print("✓ requests imported successfully")
    except ImportError as e:
        print(f"✗ requests import failed: {e}")
        return False

    return True


def test_model_file():
    """Test if model file exists"""
    print("\n" + "="*60)
    print("Testing Model File...")
    print("="*60)

    model_path = "models/grocery_detector.pt"
    best_model_path = "models/grocery_detector2/weights/best.pt"

    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✓ Model file found: {model_path} ({size:.2f} MB)")
        return True
    elif os.path.exists(best_model_path):
        print(f"⚠ Best model found at: {best_model_path}")
        print("  Consider copying it to models/grocery_detector.pt")
        print("  Command: copy models\\grocery_detector2\\weights\\best.pt models\\grocery_detector.pt")
        return True
    else:
        print(f"✗ Model file not found at {model_path}")
        print("  Please train the model first: python train_model.py")
        return False


def test_camera():
    """Test if camera is accessible"""
    print("\n" + "="*60)
    print("Testing Camera...")
    print("="*60)

    try:
        import cv2
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera is accessible and working")
                cap.release()
                return True
            else:
                print("✗ Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("✗ Cannot open camera")
            print("  Try changing CAMERA_INDEX in config.py to 1 or 2")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False


def test_audio():
    """Test if audio output works"""
    print("\n" + "="*60)
    print("Testing Audio Output...")
    print("="*60)

    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("✓ Audio engine initialized")

        # Test speaking (optional - uncomment to test)
        # engine.say("Audio test successful")
        # engine.runAndWait()

        return True
    except Exception as e:
        print(f"⚠ Audio test warning: {e}")
        print("  Audio might not work, but system can still run")
        return True  # Don't fail the test, audio is optional


def test_model_loading():
    """Test if model can be loaded"""
    print("\n" + "="*60)
    print("Testing Model Loading...")
    print("="*60)

    try:
        from ultralytics import YOLO
        import torch

        model_path = "models/grocery_detector.pt"

        if not os.path.exists(model_path):
            print("⚠ Model file not found, skipping model load test")
            return True

        print("Loading model... (this may take a moment)")
        model = YOLO(model_path)
        print("✓ Model loaded successfully")

        # Check if GPU is available
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ GPU not available - will use CPU (slower)")

        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_api_connection():
    """Test if Open Food Facts API is accessible"""
    print("\n" + "="*60)
    print("Testing API Connection...")
    print("="*60)

    try:
        import requests
        # Test with a known barcode (Coca Cola)
        test_barcode = "5449000000996"
        url = f"https://world.openfoodfacts.org/api/v0/product/{test_barcode}.json"

        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✓ Open Food Facts API is accessible")
            return True
        else:
            print(f"⚠ API returned status code: {response.status_code}")
            return True  # Don't fail, might be temporary
    except Exception as e:
        print(f"⚠ API connection test failed: {e}")
        print("  This might be a network issue. Barcode detection will not work without internet.")
        return True  # Don't fail the test


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GROCERY DETECTION SYSTEM - PRE-FLIGHT CHECK")
    print("="*60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Model File", test_model_file),
        ("Model Loading", test_model_loading),
        ("Camera", test_camera),
        ("Audio", test_audio),
        ("API Connection", test_api_connection),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! You can run the main application:")
        print("  python main.py")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above before running the main application.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

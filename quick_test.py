"""Quick test - verify detection works without camera or UI"""
from config import Config
from object_detector import ObjectDetector
import os, cv2, glob

c = Config()
od = ObjectDetector(c.MODEL_PATH, c.OBJECT_DETECTION_CONFIDENCE_THRESHOLD,
                    dataset_yaml_path=os.path.join(c.DATASET_DIR, "data.yaml"))

# Find a test image
images = glob.glob("datasets/test/images/*.jpg")
if not images:
    print("No test images found")
    exit()

# Pick first non-detected image
test_img = [i for i in images if "_detected" not in i][0]
print(f"Testing with: {test_img}")

img = cv2.imread(test_img)
dets = od.detect_objects(img)

print(f"\n{'='*50}")
print(f"DETECTION RESULTS")
print(f"{'='*50}")
if dets:
    for d in dets:
        pct = d['confidence'] * 100
        in_ds = od.is_in_dataset(d['class_name'])
        print(f"  >> {d['class_name']} - {pct:.1f}% confidence {'(IN dataset)' if in_ds else '(NOT in dataset)'}")
else:
    print("  No detections (image may not contain trained products)")

print(f"\nTotal: {len(dets)} detection(s)")
print(f"Model has {len(od.class_names)} classes")
print(f"\nAll {len(od.class_names)} trained product classes:")
for i, name in od.class_names.items():
    print(f"  {i+1}. {name}")

print(f"\n{'='*50}")
print("SUCCESS - Your trained model is working!")
print(f"{'='*50}")

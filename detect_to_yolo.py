from ultralytics import YOLO
import os

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # you can use yolov8s.pt for slightly better accuracy

# Input folder (options: single image, folder, or path to images)
# Option 1: Use single test image
# source_folder = "bus1.jpg"
# Option 2: Use training images (uncomment to use)
source_folder = "datasets/recyclables/images/train"
# Option 3: Use validation images (uncomment to use)
# source_folder = "datasets/recyclables/images/val"

# Run detection
results = model.predict(
    source=source_folder,
    save=True,               # save images with bounding boxes
    save_txt=True,           # save YOLO-format .txt labels
    project="runs/detect",   # output folder
    name="label_test"        # subfolder name
)

print("✅ Detection complete! Labels saved inside runs/detect/label_test/labels/")

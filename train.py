from pathlib import Path
from ultralytics import YOLO


pretrained_model_path = Path("data/model/yolo11n.pt")

# Load a pretrained YOLO model (recommended for training)
model = YOLO(pretrained_model_path)

model.train(data="data.yaml", epochs=500, batch=2, imgsz=1280)
# freeze backbone: epoch300.pt, epoch500.pt
# epochs=800,
# epochs=500, imgsz=1600 

import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

model_path = Path("checkpoints/640-epoch800-best.pt")
model = YOLO(model_path)
output_results = {"annotations": []}


for img_path in tqdm(Path("data/test_images").glob("*.jpg")):
    infer_info: list[Results] = model(img_path, show_boxes=False, verbose=False)
    results = infer_info[0]
    filename = f"test_images\\{img_path.name}"
    conf = results.boxes.conf
    for i in range(len(results.boxes.conf)):
        xyxy = results.boxes.xyxy[i]
        data = {
            "filename": f"test_images\\{img_path.name}",
            "conf": results.boxes.conf[i].item(),
            "box": {
                "xmin": xyxy[0].item(),
                "ymin": xyxy[1].item(),
                "xmax": xyxy[2].item(),
                "ymax": xyxy[3].item(),
            },
            "label": results.names.get(int(results.boxes.cls[i].item())),
        }
        output_results["annotations"].append(data)

with open("data/submit.json", "w", encoding="utf8") as f:
    json.dump(output_results, f, indent=4, ensure_ascii=False)

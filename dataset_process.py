import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

INPUT_PATH = Path("data/train")  # 所给数据集

OUTPUT_PATH = Path("data/yolods")  # 转换到此文件夹作为 yolo 训练数据集
for i in ["train", "val"]:
    OUTPUT_PATH.joinpath(f"images/{i}").mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.joinpath(f"labels/{i}").mkdir(parents=True, exist_ok=True)

original_labels: list[dict] = json.loads(INPUT_PATH.joinpath("train.json").read_text())[
    "annotations"
]
t = len(original_labels) // 10


def process(label_data: list[dict], train: bool = True):
    """train: is train dataset?"""
    if train:
        to = "train"
    else:
        to = "val"

    for data in tqdm(label_data, desc=f"{to} dataset"):
        img_path = INPUT_PATH.joinpath(data["filename"].replace("\\", "/"))
        if not img_path.exists():  # 图片不存在
            print(f"{img_path} does not exists")
            continue

        for i in [
            "occluded",
            "truncated",
            "difficult",
        ]:  # 难以识别
            if data.get(i, None):
                continue

        if data["ignore"] == 1 or not data["inbox"]:  # 背面，或无形状/颜色标注
            continue

        box_data = data["inbox"][0]  # 仅一个灯亮
        shape = box_data["shape"]  # 0~5 7~9 -1

        color: str = box_data["color"]  # red green yellow
        bbox: dict[str, float] = data["bndbox"]  # xmin, ymin, xmax, ymax
        w, h = Image.open(img_path).size

        output_img_path = OUTPUT_PATH / "images" / to / img_path.name
        if not output_img_path.exists():
            output_img_path.hardlink_to(img_path)  # linknode -> original

        label_path = OUTPUT_PATH / "labels" / to / f"{img_path.stem}.txt"
        x_center = (bbox["xmin"] + bbox["xmax"]) / 2 / w
        y_center = (bbox["ymin"] + bbox["ymax"]) / 2 / h
        width = (bbox["xmax"] - bbox["xmin"]) / w
        height = (bbox["ymax"] - bbox["ymin"]) / h
        class_id = {"red": 0, "green": 1, "yellow": 2}[color.lower()]
        with label_path.open("+a", encoding="utf8") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


print("total size:", len(original_labels))
process(original_labels[t:])
process(original_labels[:t], train=False)

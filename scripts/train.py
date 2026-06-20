from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset/original",
    epochs=30,
    imgsz=224,
    batch=16,
    project="results",
    name="original"
)

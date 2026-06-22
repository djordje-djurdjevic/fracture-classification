from ultralytics import YOLO

MODEL_PATH = "results/original/original_2/weights/best.pt"

def export_to_onnx():
    model = YOLO(MODEL_PATH)

    model.export(
        format="onnx",
        imgsz=224,
        simplify=True,
        opset=12,
        dynamic=False
    )

    print("ONNX export completed.")


if __name__ == "__main__":
    export_to_onnx()
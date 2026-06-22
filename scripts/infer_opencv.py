import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from preprocess import *

MODEL_PATH = "results/processed/clahe/weights/best.onnx"
TEST_DIR = "dataset/original/test"

CLASS_NAMES = [
    "fractured",
    "not fractured"
]

IMG_SIZE = 224

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def preprocess(image_path, steps = None):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    img = apply_pipeline(img, steps)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    img = np.stack([img, img, img], axis=-1) #3 channels 
    
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img #return tensor

def main():
    steps = [clahe]

    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    y_true = []
    y_pred = []

    for image_path in Path(TEST_DIR, "fractured").iterdir():

        blob = preprocess(image_path, steps)

        net.setInput(blob)

        output = net.forward()

        probs = softmax(output[0])

        pred = np.argmax(probs)

        y_true.append(0)
        y_pred.append(pred)


    for image_path in Path(TEST_DIR, "not fractured").iterdir():

        blob = preprocess(image_path, steps)

        net.setInput(blob)

        output = net.forward()

        probs = softmax(output[0])

        pred = np.argmax(probs)

        y_true.append(1)
        y_pred.append(pred)


    print("Accuracy:")
    print(accuracy_score(y_true, y_pred))

    print()

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print()

    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    ))

if __name__ == "__main__":
    main()
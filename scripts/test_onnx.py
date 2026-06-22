import cv2
import numpy as np
from preprocess import *

MODEL_PATH = "results/original/original_2/weights/best.onnx"

CLASS_NAMES = [
    "fractured",
    "not fractured"
]

IMG_SIZE = 224


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def preprocess(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


    # steps = [gaussian_blur, clahe]
    #apply_pipeline

def main():

    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    image_path = "dataset/original/test/not fractured/5.png"
    blob = preprocess(image_path)
    
    net.setInput(blob)
    output = net.forward()
    probs = softmax(output[0])
    pred_idx = np.argmax(probs)

    print()
    print("Prediction:", CLASS_NAMES[pred_idx])
    print("Confidence:", probs[pred_idx])

    print()
    print("All probabilities:")

    for i, p in enumerate(probs):
        print(f"{CLASS_NAMES[i]}: {p:.4f}")


if __name__ == "__main__":
    main()
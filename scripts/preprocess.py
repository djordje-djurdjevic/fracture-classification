import cv2
from pathlib import Path
import numpy as np

SOURCE_DIR = "dataset/original"
TARGET_DIR = "dataset/processed"

def apply_to_dataset(source_dir: str, target_dir: str, steps):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    processed_count = 0

    for image_path in source_dir.rglob("*"): #all subfolders recursively

        if image_path.suffix.lower() not in valid_extensions:
            continue

        relative_path = image_path.relative_to(source_dir)
        output_path = target_dir / relative_path #dataset/processed/train/fractured/img1.png

        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        processed = apply_pipeline(image, steps)

        cv2.imwrite(str(output_path), processed)
        processed_count += 1

    print(f"Processed {processed_count} images.")




def clahe(img, clip_limit=2.0, grid=(8, 8)):
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return c.apply(img)


def gaussian_blur(img, ksize=(3, 3)):
    return cv2.GaussianBlur(img, ksize, 0)


def hist_equalization(img):
    return cv2.equalizeHist(img)


def sharpen(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(img, -1, kernel)


def apply_pipeline(img, steps):
    for step in steps:
        img = step(img)
    return img



# === MAIN ===

def main():
    # CLAHE
    # steps = [clahe]

    # CLAHE and sharpening
    steps = [clahe, sharpen]

    # blur and CLAHE
    # steps = [gaussian_blur, clahe]

    # full pipeline
    # steps = [gaussian_blur, clahe, sharpen]
    apply_to_dataset(SOURCE_DIR, TARGET_DIR, steps)
    

if __name__ == "__main__":
    main()

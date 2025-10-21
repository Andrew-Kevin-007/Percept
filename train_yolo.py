import os
from ultralytics import YOLO


def main():
    data_yaml = os.path.join("dataset", "dataset.yaml")
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        imgsz=640,
        epochs=20,
        batch=16,
        device="cpu",
        project="runs",
        name="detect/train",
        verbose=True,
    )

    # Validate best weights
    best = os.path.join("runs", "detect", "train", "weights", "best.pt")
    if os.path.isfile(best):
        model_best = YOLO(best)
        model_best.val(data=data_yaml, imgsz=640, device="cpu")
        # Predict on validation images
        val_images = os.path.join("dataset", "images", "val")
        model_best.predict(source=val_images, imgsz=640, device="cpu", save=True, project="runs", name="detect/predict_val", conf=0.25)


if __name__ == "__main__":
    main()



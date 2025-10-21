import os
import time
import json
from datetime import datetime
from typing import List, Dict

from ultralytics import YOLO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class ImageHandler(FileSystemEventHandler):
    def __init__(self, model: YOLO, out_dir: str, log_path: str, conf: float = 0.25) -> None:
        super().__init__()
        self.model = model
        self.out_dir = out_dir
        self.log_path = log_path
        self.conf = conf
        ensure_dir(self.out_dir)

    def on_created(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext not in SUPPORTED_EXT:
            return
        self._process_image(event.src_path)

    def _process_image(self, image_path: str) -> None:
        # Wait briefly to ensure file write completion
        time.sleep(0.1)
        try:
            res_list = self.model.predict(
                source=image_path,
                imgsz=640,
                conf=self.conf,
                device="cpu",
                save=True,
                project=self.out_dir,
                name="annotated",
                exist_ok=True,
            )
            if not res_list:
                return
            res = res_list[0]
            preds = []
            has_defect = False
            names = res.names
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls.item())
                    cls_name = names.get(cls_id, str(cls_id))
                    conf = float(b.conf.item()) if b.conf is not None else 0.0
                    xyxy = [float(x) for x in b.xyxy[0].tolist()]
                    preds.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox_xyxy": xyxy,
                    })
                    if cls_name == "defect":
                        has_defect = True

            entry: Dict = {
                "timestamp": now_iso(),
                "image": os.path.basename(image_path),
                "has_defect": has_defect,
                "num_detections": len(preds),
                "detections": preds,
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            entry = {
                "timestamp": now_iso(),
                "image": os.path.basename(image_path),
                "error": str(e),
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time folder watcher for YOLO inference")
    parser.add_argument("--input_dir", type=str, default="incoming", help="Folder to watch for new images")
    parser.add_argument("--out_dir", type=str, default="runs/realtime", help="Output directory for annotations")
    parser.add_argument("--log", type=str, default="runs/realtime/inference_log.jsonl", help="Path to JSONL log file")
    parser.add_argument("--weights", type=str, default="runs/detect/train3/weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    ensure_dir(args.input_dir)
    ensure_dir(os.path.dirname(args.log))

    weights_path = args.weights if os.path.isfile(args.weights) else "yolov8n.pt"
    model = YOLO(weights_path)

    event_handler = ImageHandler(model=model, out_dir=args.out_dir, log_path=args.log, conf=args.conf)
    observer = Observer()
    observer.schedule(event_handler, path=args.input_dir, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()



from ultralytics import YOLO
import cv2
import torch
from .config import Config

class ObjectDetector:
    def __init__(self, model_path=Config.YOLO_MODEL):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Loading YOLOv8 model on {self.device}...")
        self.model = YOLO(model_path)
        self.classes = Config.CLASSES

    def detect(self, frame):
        """
        Performs inference on a single frame.
        """
        results = self.model.predict(
            frame, 
            conf=Config.CONF_THRESHOLD, 
            iou=Config.IOU_THRESHOLD, 
            classes=self.classes, 
            verbose=False,
            device=self.device
        )
        return results[0]

    def plot_boxes(self, frame, results):
        """
        Draws bounding boxes using Ultralytics built-in plotter for speed.
        """
        return results.plot()
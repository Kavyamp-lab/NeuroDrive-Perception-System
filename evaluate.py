from ultralytics import YOLO
import argparse

def evaluate_model(data_yaml):
    """
    Runs YOLOv8 Validation on a dataset.
    
    args:
        data_yaml (str): Path to dataset.yaml (standard YOLO format)
    """
    model = YOLO('models/yolov8n.pt')
    
    print("[INFO] Starting Evaluation on Dataset...")
    metrics = model.val(data=data_yaml, split='val')
    
    print("\n" + "="*40)
    print("       EVALUATION RESULTS")
    print("="*40)
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Path to data.yaml')
    args = parser.parse_args()
    
    evaluate_model(args.data)
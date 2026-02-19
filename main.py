import cv2
import time
import argparse
from src.detector import ObjectDetector
from src.lanes import LaneDetector

def main(video_path, output_path, show_display):
    # Initialize Modules
    detector = ObjectDetector()
    lane_model = LaneDetector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Video Writer Setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_total = time.time()

    print("[INFO] Starting Inference Loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # 1. Lane Detection (returns frame with lane overlay)
        lane_frame = lane_model.process(frame.copy())

        # 2. Object Detection (draws boxes on top of lane frame)
        det_results = detector.detect(frame)
        final_frame = detector.plot_boxes(lane_frame, det_results)

        # 3. FPS Calculation
        end_time = time.time()
        fps_curr = 1 / (end_time - start_time)
        
        cv2.putText(final_frame, f"FPS: {fps_curr:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if output_path:
            out.write(final_frame)

        if show_display:
            cv2.imshow('NeuroDrive Perception', final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_count} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to input video')
    parser.add_argument('--out', type=str, default='outputs/result.mp4', help='Path to output video')
    parser.add_argument('--view', action='store_true', help='Display video while processing')
    
    args = parser.parse_args()
    main(args.source, args.out, args.view)
import numpy as np

class Config:
    # ROI for Lane Detection (Trapezoid for Perspective Transform)
    # Adjusted for standard 1280x720 driving footage
    SOURCE_POINTS = np.float32([
        [580, 460],  # Top Left
        [205, 720],  # Bottom Left
        [1110, 720], # Bottom Right
        [700, 460]   # Top Right
    ])
    
    DEST_POINTS = np.float32([
        [320, 0],    # Top Left
        [320, 720],  # Bottom Left
        [960, 720],  # Bottom Right
        [960, 0]     # Top Right
    ])

    # Lane Detection Parameters
    N_WINDOWS = 9
    MARGIN = 100
    MINPIX = 50
    YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
    XM_PER_PIX = 3.7 / 700 # meters per pixel in x dimension

    # Object Detection
    YOLO_MODEL = 'yolov8n.pt'  # Nano model for FPS, use 'yolov8x.pt' for accuracy
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    CLASSES = [2, 3, 5, 7]  # COCO IDs: Car, Motorcycle, Bus, Truck
# NeuroDrive Perception System

A research-grade Autonomous Driving Perception System developed in Python. This project fuses Deep Learning (YOLOv8) for Object Detection with Advanced Computer Vision (Perspective Transforms & Polynomial Fitting) for Lane Detection.

## Features
- **Object Detection**: Real-time detection of Cars, Trucks, Buses, and Motorcycles using YOLOv8.
- **Lane Detection**: Robust curved lane detection using HLS thresholding, sliding windows, and 2nd-degree polynomial fitting.
- **Metrics**: Automated scripts for calculating mAP, Precision, and Recall.
- **Modular Design**: Separated configuration, logic, and visualization for research experimentation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NeuroDrive-Perception.git
   cd NeuroDrive-Perception

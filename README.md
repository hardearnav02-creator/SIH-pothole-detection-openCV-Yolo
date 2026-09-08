# Operation Vanguard: Urban Tracking AI

This repository contains the core computer vision engine and backend API for real-time urban hazard tracking.

## Core Components
* **`app.py`**: The YOLOv8 computer vision script. It captures the live webcam feed, detects road hazards, and logs simulated GPS coordinates to a local `detections.csv` file.
* **`server.py`**: A lightweight Flask API bridge. It reads the local CSV log and broadcasts it as a live JSON endpoint for frontend web maps to consume.

## Prerequisites
Ensure you have the required Python libraries installed:
```powershell
pip install flask flask-cors pandas ultralytics opencv-python

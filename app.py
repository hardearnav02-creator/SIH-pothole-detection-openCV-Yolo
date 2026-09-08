import cv2
import time
from datetime import datetime
import csv
import os
from ultralytics import YOLO

# 1. Load trained models
model_traffic = YOLO("yolov8n.pt")
model_hazard  = YOLO("road_model.pt")

# Target class IDs for COCO traffic model: person, bicycle, car, motorcycle, bus, truck
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7]

csv_filename = "detections.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Hazard_Type", "Confidence", "Latitude", "Longitude"])

# Simulated Route Starting Point (Pune - Shivaji Nagar area)
current_lat = 18.5314
current_lon = 73.8446

source = "test_video.mp4" if os.path.exists("test_video.mp4") else 0
cap = cv2.VideoCapture("test_video.mp4")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0
FRAME_SKIP = 2
last_hazard_time = 0
HAZARD_COOLDOWN = 1.5

print("Optimized Edge-AI Pipeline running (Full Screen). Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    current_time = time.time()

    # Full frame inference for both models
    results_traffic = model_traffic(frame, conf=0.55, classes=TRAFFIC_CLASSES, verbose=False)[0]
    results_hazard  = model_hazard(frame, conf=0.45, verbose=False)[0]

    # Collect valid hazards across the entire frame
    new_hazards = []
    for box in results_hazard.boxes:
        cls_id = int(box.cls[0])
        label = model_hazard.names[cls_id]
        conf = float(box.conf[0])
        new_hazards.append((label, conf))

    # Log to CSV with debouncing and simulated GPS
    if new_hazards and (current_time - last_hazard_time > HAZARD_COOLDOWN):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        current_lat += 0.0002
        current_lon += 0.0001
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for label, conf in new_hazards:
                writer.writerow([timestamp, label, round(conf, 2), current_lat, current_lon])
                print(f"[INCIDENT LOGGED] {label.upper()} | Loc: {current_lat:.4f}, {current_lon:.4f}")
        last_hazard_time = current_time

    # Render visualizations on the full frame
    annotated_frame = results_traffic.plot()

    # Draw hazard detections directly without coordinate offsets
    for box in results_hazard.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = f"{model_hazard.names[cls_id]} {conf:.2f}"
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Urban Sensing Intelligence Unit", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
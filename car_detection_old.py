# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
#
# # Load YOLOv8 model
# model = YOLO('yolov8n.pt')  # You can choose different YOLOv8 models like yolov8s.pt, yolov8m.pt, etc.
#
# # Load video
# cap = cv2.VideoCapture('Input/highway.mp4')
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Preprocess and detect objects
#     results = model(frame)
#
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             cls = box.cls
#             conf = box.conf
#             if cls in [2, 5, 3] and conf > 0.5:  # Class IDs for car, truck, motorcycle in COCO dataset
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 if cls in [2, 5, 3] and conf > 0.5:  # Class IDs for car, truck, motorcycle in COCO dataset
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     if cls == 2:
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color for cars
#                         cv2.putText(frame, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     elif cls == 5:
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color for trucks
#                         cv2.putText(frame, 'Truck', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#                     elif cls == 3:
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for motorcycles
#                         cv2.putText(frame, 'Motorcycle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#
#     # Convert BGR to RGB for Matplotlib
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Display output using Matplotlib
#     plt.imshow(frame_rgb)
#     plt.title("Sunsky Software")
#     plt.axis('off')
#     plt.pause(0.01)  # Adjust pause time to control playback speed
#
#     # Clear the current plot to show the next frame
#     plt.clf()
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose different YOLOv8 models like yolov8s.pt, yolov8m.pt, etc.

# Path to input video and output directory
input_video = 'Input/highway.mp4'
output_dir = 'Output'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(input_video)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object for output video
output_filename = os.path.join(output_dir, 'output_highway.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for mp4
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Function to detect cars in each frame, mark rectangle, and save cropped regions
def detect_and_crop(frame):
    # Perform object detection
    results = model(frame.copy())  # Make a copy of the frame to avoid modifying the original

    # Initialize a list to store cropped car regions
    car_crops = []

    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls
            conf = box.conf
            if cls in [2, 5, 3] and conf > 0.5:  # Class IDs for car, truck, motorcycle in COCO dataset
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == 2:  # Car
                    # Crop the region containing the car
                    car_crop = frame[y1:y2, x1:x2]

                    # Resize cropped region to 400x400 pixels
                    car_crop_resized = cv2.resize(car_crop, (400, 400))

                    car_crops.append(car_crop_resized)

                    # Draw rectangle on original frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color for cars
                    cv2.putText(frame, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save all cropped car regions to separate image files
    for idx, car_crop in enumerate(car_crops):
        crop_output_path = os.path.join(output_dir, f'frame_{idx}_car.jpg')
        cv2.imwrite(crop_output_path, car_crop)

    return frame

# Process each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect cars in the frame, mark rectangle, and save cropped regions
    processed_frame = detect_and_crop(frame)

    # Write the processed frame to the output video
    out.write(processed_frame)

    # Display or save the processed frame (optional)
    cv2.imshow('Processed Frame', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

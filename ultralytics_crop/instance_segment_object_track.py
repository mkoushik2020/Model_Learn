from collections import defaultdict
import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(lambda: [])

# Load the YOLO model for segmentation
model = YOLO("yolov8n-seg.pt")

# Ensure the input video path is correct
video_path = os.path.abspath("../Input/highway.mp4")

# Open the input video file
cap = cv2.VideoCapture(video_path)

# Check if the video capture was successfully initialized
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Retrieve video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Print video properties for debugging
print(f"Width: {w}, Height: {h}, FPS: {fps}")

# Ensure the frame rate is valid
if fps < 1:
    fps = 30  # Default to 30 FPS if an invalid frame rate is detected
    print("Invalid frame rate detected. Defaulting to 30 FPS.")

# Set up the video writer with the correct properties
output_path = os.path.abspath("instance-segmentation-object-tracking.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    # Perform object tracking and segmentation
    results = model.track(im0, persist=True)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=str(track_id))

    # Write the annotated frame to the output video file
    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video writer and capture objects, and close all windows
out.release()
cap.release()
cv2.destroyAllWindows()

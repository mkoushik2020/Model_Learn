import cv2
import time
from ultralytics import YOLO, solutions

# Load YOLO model
model = YOLO("yolov8n.pt")

# Function to initialize video capture
def init_video_capture():
    cap = cv2.VideoCapture("rtsp://admin:sunsky@192.168.10.191/stream1")
    return cap

# Initialize video capture
cap = init_video_capture()
assert cap.isOpened(), "Error reading video file"

# Get original video properties
original_w, original_h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Set desired output size (reduce frame size)
output_w, output_h = 1500,900 # You can adjust these values as needed

# Define original region points
original_region_points = [(25, 300), (1080, 404), (1080, 360), (25, 360)]

# Calculate original region width and height
region_width = max(x for x, y in original_region_points) - min(x for x, y in original_region_points)
region_height = max(y for x, y in original_region_points) - min(y for x, y in original_region_points)

# Determine the center of the output frame
center_x, center_y = output_w // 2, output_h // 2

# Define a downward offset (adjust this value as needed)
downward_offset = 60

# Calculate the new region points centered in the middle of the output frame with downward offset
region_points = [
    (center_x - region_width // 2, center_y + region_height // 2 + downward_offset),
    (center_x + region_width // 2, center_y + region_height // 2 + downward_offset),
    (center_x + region_width // 2, center_y - region_height // 2 + downward_offset),
    (center_x - region_width // 2, center_y - region_height // 2 + downward_offset)
]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_w, output_h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while True:
    if not cap.isOpened():
        print("Reconnecting to video stream...")
        time.sleep(5)  # Wait before retrying
        cap = init_video_capture()
        if not cap.isOpened():
            print("Failed to reconnect to video stream.")
            continue
        print("Reconnected to video stream.")

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        cap.release()
        continue

    # Resize the frame
    im0_resized = cv2.resize(im0, (output_w, output_h))

    # Track objects on the resized frame
    tracks = model.track(im0_resized, persist=True, show=False)

    # Count objects and draw results on the frame
    im0_resized = counter.start_counting(im0_resized, tracks)

    # Write the frame to the output video
    video_writer.write(im0_resized)

# Clean up
cap.release()
video_writer.release()
cv2.destroyAllWindows()

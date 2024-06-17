# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
#
# # Load YOLOv8 model
# model = YOLO('best.pt')  # You can choose different YOLOv8 models like yolov8s.pt, yolov8m.pt, etc.
#
# # Load video
# cap = cv2.VideoCapture('Input/img_5098.jpg')
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
#             if cls in [0, 1, 2, 5] and conf > 0.5:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 if cls in [0, 1, 2, 5] and conf > 0.5:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     if cls == 0:
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green color for cars
#                         cv2.putText(frame, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                     elif cls == 2:
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue color for trucks
#                         cv2.putText(frame, 'Truck', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#                     elif cls == 1:
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red color for motorcycles
#                         cv2.putText(frame, 'Motorcycle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#                     elif cls == 5:
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1) # Green color for MotorVan
#                         cv2.putText(frame, 'MotorVan', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load YOLOv8 model
model = YOLO('best.pt')  # You can choose different YOLOv8 models like yolov8s.pt, yolov8m.pt, etc.

def process_frame(frame):
    # Preprocess and detect objects
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls
            conf = box.conf
            if cls in [0, 1, 2, 5] and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green color for cars
                    cv2.putText(frame, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif cls == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red color for motorcycles
                    cv2.putText(frame, 'Motorcycle', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                elif cls == 2:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue color for trucks
                    cv2.putText(frame, 'Truck', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                elif cls == 5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green color for MotorVan
                    cv2.putText(frame, 'MotorVan', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        # Convert BGR to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display output using Matplotlib
        plt.imshow(frame_rgb)
        plt.title("Sunsky Software")
        plt.axis('off')
        plt.pause(0.01)  # Adjust pause time to control playback speed

        # Clear the current plot to show the next frame
        plt.clf()

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    frame = cv2.imread(image_path)
    frame = process_frame(frame)

    # Convert BGR to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display output using Matplotlib
    plt.imshow(frame_rgb)
    plt.title("Sunsky Software")
    plt.axis('off')
    plt.show()  # Use plt.show() to display the image and pause

# Ask the user for the input file path
input_path = input("Enter the path to the video or image file: ").strip()

# Check if the input file exists
if not os.path.isfile(input_path):
    print("File not found!")
else:
    # Determine if the input is a video or image based on the file extension
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov']:
        process_video(input_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
        process_image(input_path)
    else:
        print("Invalid file type. Please enter a valid video or image file.")

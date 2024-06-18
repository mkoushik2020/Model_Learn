# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import os
#
# # Load YOLOv8 model
# model = YOLO('best_new.pt')  # You can choose different YOLOv8 models like yolov8s.pt, yolov8m.pt, etc.
#
# def process_frame(frame):
#     # Preprocess and detect objects
#     results = model(frame)
#
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             cls = box.cls
#             conf = box.conf
#             if cls in [0, 1, 2 , 3 , 4 , 7 , 12 , 14 , 15 , 16] and conf > 0.5:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 if cls == 0:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, '10 wheeler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 elif cls == 1:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, 'MotorVan', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 elif cls == 2:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, 'MotorVan', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 elif cls == 3:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 0, 255), 2)
#                     cv2.putText(frame, '4 Wheeler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#                 elif cls == 4:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, '6 Wheeler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 elif cls == 7:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, '6 Wheeler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 elif cls == 12:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, '4 Wheeler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 elif cls == 14:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, '6 Wheeler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 elif cls == 15:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, '4 Wheeler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 elif cls == 16:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, '4 Wheeler', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     return frame
#
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = process_frame(frame)
#
#         # Convert BGR to RGB for Matplotlib
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Display output using Matplotlib
#         plt.imshow(frame_rgb)
#         plt.title("Sunsky Software Technologies Pvt. Ltd.")
#         plt.axis('off')
#         plt.pause(0.01)  # Adjust pause time to control playback speed
#
#         # Clear the current plot to show the next frame
#         plt.clf()
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# def process_image(image_path):
#     frame = cv2.imread(image_path)
#     frame = process_frame(frame)
#
#     # Convert BGR to RGB for Matplotlib
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Display output using Matplotlib
#     plt.imshow(frame_rgb)
#     plt.title("Sunsky Software")
#     plt.axis('off')
#     plt.show()  # Use plt.show() to display the image and pause
#
# # Ask the user for the input file path
# input_path = input("Enter the path to the video or image file: ").strip()
#
# # Check if the input file exists
# if not os.path.isfile(input_path):
#     print("File not found!")
# else:
#     # Determine if the input is a video or image based on the file extension
#     ext = os.path.splitext(input_path)[1].lower()
#     if ext in ['.mp4', '.avi', '.mov']:
#         process_video(input_path)
#     elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
#         process_image(input_path)
#     else:
#         print("Invalid file type. Please enter a valid video or image file.")
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load YOLOv8 model
model = YOLO('best_new.pt')  # You can choose different YOLOv8 models like yolov8s.pt, yolov8m.pt, etc.

# Define the class names based on your model's training
class_names = {0: '10 Wheeler', 1: 'MotorVan', 2: 'MotorVan', 3: '4 Wheeler',
               4: '6 Wheeler', 7: '6 Wheeler', 12: '4 Wheeler', 14: '6 Wheeler',
               15: '4 Wheeler', 16: '4 Wheeler'}


def process_frame(frame):
    # Preprocess and detect objects
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if cls in class_names and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = class_names[cls]
                color = (0, 255, 0) if 'Wheeler' in label else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    plt.ion()  # Turn on interactive mode for real-time display

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        # Convert BGR to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display output using Matplotlib
        plt.imshow(frame_rgb)
        plt.title("Sunsky Software Technologies Pvt. Ltd.")
        plt.axis('off')
        plt.pause(0.01)  # Adjust pause time to control playback speed

        # Clear the current plot to show the next frame
        plt.clf()

    cap.release()
    plt.ioff()  # Turn off interactive mode
    plt.show()
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

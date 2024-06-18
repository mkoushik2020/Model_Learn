from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Load the class names (COCO dataset in this example)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Load the image
image_path = 'Input/test9.webp'
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Annotate the image
# Assuming results is a list and we take the first element
result = results[0]

# Draw bounding boxes on the image
annotated_image = image.copy()

for box in result.boxes:
    # Extract the bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # Draw the rectangle on the image
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Convert class and confidence from tensor to float/int
    cls = int(box.cls)
    conf = float(box.conf)
    # Get the class name from the COCO_CLASSES list
    class_name = COCO_CLASSES[cls]
    # Add label text
    label = f'{class_name}: {conf:.2f}'
    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the annotated image in a window
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

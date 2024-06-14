from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Train the model
try:
    results = model.train(data="dataset.yaml", epochs=50, imgsz=640)
except Exception as e:
    print(f"Error: {e}")

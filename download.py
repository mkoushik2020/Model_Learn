from roboflow import Roboflow
rf = Roboflow(api_key="KNrDaxrNXMWcyRJTLf3H")
project = rf.workspace("thiraphat").project("classification-of-cars")
version = project.version(3)
dataset = version.download("yolov8")
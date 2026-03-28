from roboflow import Roboflow

rf = Roboflow(api_key="bgM0BpO41XLruUg28dKj")

project = rf.workspace("eksperiment").project("brain-tumor-mri-ycidy")

version = project.version(2)

dataset = version.download("yolov8")

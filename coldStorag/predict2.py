from ultralytics import YOLO


image_path = 'IMG-20231205-WA0015.jpg'

model = YOLO("best_roi.pt")
model.predict(source=image_path, show=True, conf=0.5)

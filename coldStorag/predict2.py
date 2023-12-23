from ultralytics import YOLO


image_path = 'IMG-20231205-WA0015.jpg'

model = YOLO("best_Potato.pt")
model.predict(source=0, show=True, conf=0.5)

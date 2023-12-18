from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2

# Load YOLOv5 model with custom weights
model = YOLO('../Yolo-Weights/yolov8l.pt')

image_path = 'Images/4.jpg'  # Replace with the path to your image
results = model(image_path)
# Check if any objects were detected
if results:
    # Iterate over detected objects and draw bounding boxes
    for det in results[0]:
        label, conf, x_min, y_min, x_max, y_max = det[:6]
        label = int(label)
        conf = float(conf)

        # Draw bounding box
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline='red', width=3)
        del draw

        # Display label and confidence
        label_text = f"{model.names[int(label)]}: {conf:.2f}"
        cv2.putText(image, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the modified image
    image.show()

# Wait for a key press and close the OpenCV window
cv2.waitKey(0)
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

# Step 1: Image Detection with the First Model
image = cv2.imread("IMG-20231205-WA0015.jpg")

model_roi = YOLO("best.pt")
rois_info = model_roi.predict(source=image, show=True, conf=0.5)

# Step 2: Divide Image into Regions
for i, roi_info in enumerate(rois_info.xyxy):
    # Extract relevant information about the bounding box
    x_min, y_min, x_max, y_max = map(int, roi_info[0:4])

    # Extract region from the original image
    region_image = image[y_min:y_max, x_min:x_max]

    # Step 3: Call the Second Model for Each Region
    result_model = YOLO("best_Numbers.pt")
    result = result_model.predict(source=region_image, show=True, conf=0.5)

    # Step 4: Print the Results
    print(f"Result for Region {i+1}: {result}")

# Optionally, you can display the original image with bounding boxes around the detected regions
for roi_info in rois_info.xyxy:
    x_min, y_min, x_max, y_max = map(int, roi_info[0:4])
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imshow("Image with ROIs", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

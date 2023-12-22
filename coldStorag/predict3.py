from ultralytics import YOLO
import cv2


image_path = 'idel.jpg'
original_image = cv2.imread(image_path)

model_roi = YOLO("best_roi.pt")
model_numbers = YOLO("best_Numbers.pt")

#indes
# [Meter, oil_temp, current, machine_status]
meterParts = [0, 1, 2, 3];
# Make predictions
predictions = model_roi.predict(source=image_path, show=True, conf=0.5)

if predictions:
    first_prediction = predictions[0]
    # Access the bounding boxes and print their coordinates
    detections_ = []
    for detections in first_prediction.boxes.data.tolist():
        print('Boxes object for bbox outputs', detections)
        x1, y1, x2, y2 , score, class_id = detections
        if int(class_id) in meterParts:
            detections_.append([x1, y1, x2, y2 , score])
            roi = original_image[int(y1):int(y2), int(x1):int(x2)]
            cv2.imshow("cropped image", roi)
            predictions_numbers = model_numbers.predict(source=roi, show=True, conf=0.5)
            first_prediction_numbers = predictions_numbers[0]
            for detections_numbers in first_prediction_numbers.boxes.data.tolist():
                print('Numbers::', detections_numbers)
                x1, y1, x2, y2, score, class_id = detections_numbers
            cv2.waitKey(0)

else:
    print("No predictions.")


cv2.destroyAllWindows()
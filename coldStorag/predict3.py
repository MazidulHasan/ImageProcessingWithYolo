from ultralytics import YOLO
import cv2


image_path = 'IMG-20231205-WA0016.jpg'
original_image = cv2.imread(image_path)

model_roi = YOLO("best_roi.pt")
model_numbers = YOLO("best_Numbers3.pt")

#indes
# [Meter, oil_temp, current, machine_status]
meterParts = [1, 2];
# Make predictions
predictions = model_roi.predict(source=image_path, show=False, conf=0.5)

if predictions:
    first_prediction = predictions[0]
    # Access the bounding boxes and print their coordinates
    detections_ = []
    for detections in first_prediction.boxes.data.tolist():
        print('Boxes object for bbox outputs', detections)
        x1, y1, x2, y2, score, class_id = detections
        if int(class_id) in meterParts:
            detections_.append([x1, y1, x2, y2, score])
            roi = original_image[int(y1):int(y2), int(x1):int(x2)]
            # cv2.imshow("cropped image", roi)
            filename = "D:/yoloImageRecognitionBasic/coldStorag/roiImages/roi.jpg"

            # Set the zoom-out factor (e.g., 0.8 for 80% of the original size)
            zoom_out_factor = 0.5
            # Calculate the new dimensions
            new_height = int(roi.shape[0] * zoom_out_factor)
            new_width = int(roi.shape[1] * zoom_out_factor)
            # Resize the image
            zoomed_out_image = cv2.resize(roi, (new_width, new_height))

            cv2.imwrite(filename, zoomed_out_image)
            predictions_numbers = model_numbers.predict(source='D:/yoloImageRecognitionBasic/coldStorag/roiImages/roi.jpg', show=True, conf=0.2)
            first_prediction_numbers = predictions_numbers[0]
            for detections_numbers in first_prediction_numbers.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detections_numbers
                number_image = roi[int(y1):int(y2), int(x1):int(x2)]
                print('Numbers::', int(class_id))
            cv2.waitKey(0)

else:
    print("No predictions.")


cv2.destroyAllWindows()
import cv2
import numpy as np

BOX_WIDTH = 10
BOX_HEIGHT = 20


def process_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5))
    img_canny = cv2.Canny(img_gray, 50, 50)
    return img_canny


def get_contour(img):
    contours, hierarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours:
        return max(contours, key=cv2.contourArea)


def get_line_tip(cnt1, cnt2):
    x1, y1, w1, h1 = cv2.boundingRect(cnt1)

    if h1 > BOX_HEIGHT / 2:
        if np.any(cnt2):
            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            if x1 < x2:
                return x1, y1
        return x1 + w1, y1


def get_rect(x, y):
    half_width = BOX_WIDTH // 2
    lift_height = BOX_HEIGHT // 6
    return (x - half_width, y - lift_height), (x + half_width, y + BOX_HEIGHT - lift_height)


cap = cv2.VideoCapture("screen_record.mkv")
success, img_past = cap.read()

cnt_past = np.array([])
line_tip_past = 0, 0

while True:
    success, img_live = cap.read()

    if not success:
        break

    img_live_processed = process_img(img_live)
    img_past_processed = process_img(img_past)

    img_diff = cv2.bitwise_xor(img_live_processed, img_past_processed)
    cnt = get_contour(img_diff)

    line_tip = get_line_tip(cnt, cnt_past)

    if line_tip:
        cnt_past = cnt
        line_tip_past = line_tip
    else:
        line_tip = line_tip_past

    rect = get_rect(*line_tip)
    img_past = img_live.copy()
    cv2.rectangle(img_live, *rect, (0, 0, 255), 2)

    cv2.imshow("Cursor", img_live)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
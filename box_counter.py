import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap= cv2.VideoCapture("kutular.mp4")

model= YOLO("best.pt")
mask= cv2.imread("conveyor_mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits1= [60,400,160,400]
limits2= [380,400,500,400]
limits3= [710,400,830,400]

totalCount1 = []
totalCount2 = []
totalCount3 = []

while True:

    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100

            if conf > 0.3:
                #cvzone.putTextRect(img, "box", (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=1, rt=2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 4)
    cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 4)
    cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 0, 255), 4)

    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(results)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=1, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits1[0] < cx < limits1[2] and limits1[1] - 15 < cy < limits1[1] + 15:
            if totalCount1.count(id) == 0:
                totalCount1.append(id)
                cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 2)

        elif limits2[0] < cx < limits2[2] and limits1[1] - 15 < cy < limits1[1] + 15:
            if totalCount2.count(id) == 0:
                totalCount2.append(id)
                cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 2)

        elif limits3[0] < cx < limits3[2] and limits1[1] - 15 < cy < limits1[1] + 15:
            if totalCount3.count(id) == 0:
                totalCount3.append(id)
                cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 255, 0), 2)

    cv2.putText(img, str(len(totalCount1)), (30, 410), cv2.FONT_HERSHEY_PLAIN, 2, (100, 50, 255), 4)
    cv2.putText(img, str(len(totalCount2)), (350, 410), cv2.FONT_HERSHEY_PLAIN, 2, (100, 50, 255), 4)
    cv2.putText(img, str(len(totalCount3)), (680, 410), cv2.FONT_HERSHEY_PLAIN, 2, (100, 50, 255), 4)

    total= len(totalCount1) + len(totalCount2) + len(totalCount3)

    cvzone.putTextRect(img, f'total: {total} ', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (100, 50, 255), 4, offset=3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

url = "http://localhost:4747/video"

cap = cv2.VideoCapture(url)
detector = FaceMeshDetector(maxFaces=1)

import numpy as np

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, faces = detector.findFaceMesh(img)

    if faces:
        face_landmarks = faces[0]

        face_landmarks_np = np.array(face_landmarks)

        mask = np.zeros_like(img[:, :, 0])

        points = cv2.convexHull(face_landmarks_np)
        cv2.fillConvexPoly(mask, points, (255, 255, 255))

        mask_inv = cv2.bitwise_not(mask)

        img = cv2.bitwise_and(img, img, mask=mask_inv)

        for point in face_landmarks_np:
            x, y = point
            cv2.circle(img, (x, y), 2, (255, 255, 255), thickness=cv2.FILLED)
    else:
        img = cv2.GaussianBlur(img, (55, 55), 100)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

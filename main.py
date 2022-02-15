import cv2
import numpy as np
import yaml
from scanner.docs import getImgDoc, getContours, preprocessing

if __name__ == "__main__":

    # Load configuration
    with open("config.yml", "r") as cfgFile:
        cfg = yaml.load(cfgFile, Loader=yaml.Loader)
    IMAGE_WIDTH, IMAGE_HEIGHT = cfg['IMAGE_SIZE'].values()
    FRAME_WIDTH, FRAME_HEIGHT = cfg['FRAME_SIZE'].values()
    BRIGHTNESS = cfg['BRIGHTNESS']

    # Setting camera
    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    cap.set(10, BRIGHTNESS)

    while True:
        # Capture an image from camera
        success, img = cap.read()
        if success:
            cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            imgDoc = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))
            imgContour = img.copy()
            imgThres = preprocessing(img)
            contourApprox = getContours(imgThres)

            if contourApprox is not None:
                cv2.drawContours(imgContour, contourApprox, -1, (255, 0, 0), 3)
                imgDoc = getImgDoc(img, contourApprox, (IMAGE_WIDTH, IMAGE_HEIGHT))

            cv2.imshow("Document", imgDoc)
            cv2.imshow("Camera", imgContour)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to capture video")
            break

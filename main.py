import cv2
import numpy as np
import yaml
import os
from scanner.image import getImgDoc, getContours, preprocessing, stackImages, imgHiRes, save_image

if __name__ == "__main__":

    # Load configuration
    with open("config.yml", "r") as cfgFile:
        cfg = yaml.load(cfgFile, Loader=yaml.Loader)
    IMAGE_WIDTH, IMAGE_HEIGHT = cfg['IMAGE_SIZE'].values()
    FRAME_WIDTH, FRAME_HEIGHT = cfg['FRAME_SIZE'].values()
    SAVE_DIR_PATH = os.getcwd() + cfg['SAVE_DIR_PATH']
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
            # Initialize image
            imgDoc = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))
            imgContour = img.copy()
            # Preprocess
            imgThres = preprocessing(img)
            # Draw approximate contour
            contourApprox = getContours(imgThres)
            if contourApprox is not None:
                cv2.drawContours(imgContour, contourApprox, -1, (255, 0, 0), 3)
                imgDoc = getImgDoc(img, contourApprox, (IMAGE_WIDTH, IMAGE_HEIGHT))
                imageArray = ([imgContour, imgDoc])
            else:
                imageArray = ([imgContour, img])
            # Show stacked images
            nameArray = (['Camera', 'Detected document'])
            stackedImages = stackImages(0.5, imageArray, nameArray=nameArray)
            cv2.imshow("Camera", stackedImages)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
            elif k & 0xFF == ord('a'):
                print("Capturing document")
                imgHR = imgHiRes(imgDoc)
                cv2.imshow("Document", cv2.cvtColor(imgHR.numpy(), cv2.COLOR_RGB2BGR))
                is_save = input("Save image? yes (y)/no (n)\n")
                if is_save == 'y':
                    filename = input("What is your document name?")
                    save_image(imgHR, filename, dir_path=SAVE_DIR_PATH)
                cv2.destroyWindow("Document")

        else:
            print("Failed to capture video")
            break

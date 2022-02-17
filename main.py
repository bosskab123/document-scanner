import cv2
import numpy as np
import yaml
import os
import datetime
import threading
from scanner import Scanner
from scanner.image import stackImages, imgHiRes, save_image


if __name__ == "__main__":

    # Load configuration
    with open("config.yml", "r") as cfgFile:
        cfg = yaml.load(cfgFile, Loader=yaml.Loader)
    IMAGE_WIDTH, IMAGE_HEIGHT = cfg['IMAGE_SIZE'].values()
    FRAME_WIDTH, FRAME_HEIGHT = cfg['FRAME_SIZE'].values()
    SAVE_DIR_PATH = os.getcwd() + cfg['SAVE_DIR_PATH']
    BRIGHTNESS = cfg['BRIGHTNESS']

    docScanner = Scanner(
        cam_id=0,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        brightness=BRIGHTNESS
    )

    while True:
        # Capture an image from camera
        success, _ = docScanner.read()
        if success:
            docScanner.analyze()
            imgContour, imgDoc = docScanner.get(
                img_type=['doc_contour', 'doc'],
                size=(IMAGE_WIDTH, IMAGE_HEIGHT)
            )
            imgDoc = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.uint8) if imgDoc is None else imgDoc
            imgContour = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.uint8) if imgDoc is None else imgContour
            imageArray = ([imgContour, imgDoc])

            # Show stacked images
            nameArray = (['Camera', 'Detected document'])
            stackedImages = stackImages(0.7, imageArray, nameArray=nameArray)
            cv2.imshow("Camera", stackedImages)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break
            elif k & 0xFF == ord('a'):
                Scanner.save(imgDoc, SAVE_DIR_PATH)

import cv2
import numpy as np
from scanner.utils import reorder


def getContours(img):
    '''

    :param img: image
    :return biggest: 4 coordinates of the corners of the biggest rectangular object in the img
    '''
    biggest = None
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest


def getImgDoc(img, contour_approx, size):
    '''

    :param img: image
    :param contour_approx: 4 coordinates of the corners of the biggest rectangular object in the img
    :param size: return size of the image output
    :return imgOutput: image of a detected document w.r.t. the contour_approx
    '''
    width, height = size
    contourApprox = reorder(contour_approx)
    pts1 = np.float32(contourApprox)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarped = cv2.warpPerspective(img, matrix, (width, height))
    imgCropped = imgWarped[10:width - 10, 10:height - 10]
    imgOutput = cv2.resize(imgCropped, (width, height))

    return imgOutput


def preprocessing(img):
    '''

    :param img: image
    :return: preprocessed image
    '''
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 75, 75)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

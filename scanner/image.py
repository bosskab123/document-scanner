import cv2
import numpy as np
from scanner.utils import reorder
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib.pyplot as plt


def stackImages(scale, imgArray, nameArray=None):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if nameArray is not None:
        text_param = {
            'org': (20, 30),
            'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
            'fontScale': 0.6,
            'color': (255, 255, 255),
            'thickness': None,
            'lineType': cv2.LINE_AA
        }

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
                if nameArray is not None:
                    imgArray[x][y] = cv2.putText(imgArray[x][y], nameArray[x][y], **text_param)

        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)

    else:
        for x in range(rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            if nameArray is not None:
                imgArray[x] = cv2.putText(imgArray[x], nameArray[x], **text_param)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


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


def getImgDoc(img, contour_approx, size=(360, 480)):
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


def preprocess_image(img):
    # 3 channels is required
    if len(img.shape) == 3:
        if img.shape[-1] == 4:
            img = img[...,:-1]
    else:
        img = tf.expand_dims(img, -1)

    img_size = (tf.convert_to_tensor(img.shape[:-1]) // 4) *4
    img = tf.image.crop_to_bounding_box(img, 0, 0, img_size[0], img_size[1])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0)


def save_image(img, filename, dir_path=None):
    if not isinstance(img, Image.Image):
        img = tf.clip_by_value(img, 0, 255)
        img = Image.fromarray(tf.cast(img, tf.uint8).numpy())

    if dir_path is None:
        img.save("%s.jpg" % filename)
    else:
        img.save("%s\\%s.jpg" % (dir_path, filename))
    print("Saved as %s.jpg" % filename)


def plot_image(img, title=""):
    img = np.asarray(img)
    img = tf.clip_by_value(img, 0, 255)
    img = Image.fromarray(tf.cast(img, tf.uint8).numpy())
    plt.imshow(img)
    plt.axis("off")


def imgHiRes(img):
    img = img.copy()
    SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    model = hub.load(SAVED_MODEL_PATH)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    img = model(img)
    imgOutput = tf.squeeze(img)

    return imgOutput


if __name__ == "__main__":
    img = cv2.imread("../images/low_cat.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    model = hub.load(SAVED_MODEL_PATH)
    fake_img = model(img)
    fake_img = tf.squeeze(fake_img)
    plot_image(tf.squeeze(fake_img), title="Super Resolution cat")
    save_image(tf.squeeze(fake_img), filename="hi_cat")

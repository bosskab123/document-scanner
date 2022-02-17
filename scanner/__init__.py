import cv2
from scanner.image import getImgDoc, getContours, preprocessing, stackImages, imgHiRes, save_image
import datetime


class Scanner:
    def __init__(self, cam_id=0, frame_width=360, frame_height=480, brightness=100):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, frame_width)
        self.cap.set(4, frame_height)
        self.cap.set(10, brightness)
        self.success = False
        self.is_update_img = False
        self.img = None
        self.imgContour = None
        self.imgDoc = None

    def read(self):
        self.success, self.img = self.cap.read()
        self.is_update_img = True
        return self.success, self.img

    def analyze(self):
        if self.is_update_img:
            self.is_update_img = False
            self.imgContour = self.img.copy()
            imgThres = preprocessing(self.img)
            # Draw approximate contour
            contourApprox = getContours(imgThres)
            if contourApprox is not None:
                cv2.drawContours(self.imgContour, contourApprox, -1, (255, 0, 0), 3)
                self.imgDoc = getImgDoc(self.img, contourApprox)
            else:
                self.imgDoc = None

    def get(self, img_type=[], size=(360, 480)):
        width, height = size
        return_img = []
        if img_type is []:
            return
        else:
            for o in img_type:
                if o == 'camera':
                    if self.img is None:
                        return_img.append(self.img)
                    else:
                        return_img.append(cv2.resize(self.img, (width, height)))
                elif o == 'doc':
                    if self.imgDoc is None:
                        return_img.append(self.imgDoc)
                    else:
                        return_img.append(cv2.resize(self.imgDoc, (width, height)))
                elif o == 'doc_contour':
                    if self.imgContour is None:
                        return_img.append(self.imgContour)
                    else:
                        imgContour = cv2.resize(self.imgContour, (width, height))
                        return_img.append(imgContour)
                else:
                    print("No such {} type".format(o))
            return return_img

    @staticmethod
    def save(imgDoc, dir_path):
        print("Capturing document")
        imgHR = cv2.cvtColor(imgHiRes(imgDoc).numpy(), cv2.COLOR_RGB2BGR)
        ts = datetime.datetime.now()
        filename = "{}".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        save_image(imgHR, filename, dir_path=dir_path)
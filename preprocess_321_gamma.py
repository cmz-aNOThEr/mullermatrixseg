import os
from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2
import argparse
import polanalyser as pa
import numpy as np
import cv2
import numpy as np
# Read all images
root_path = "22test"

import cv2
import numpy as np

def gamma(x, r):
    y = pow(x / 255, r) * 255
    return y.astype(np.uint8)

def illum(img):
    # img = cv2.imread("test2.jpg")
    # img = img[532:768, 0:512]
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_bw, 180, 255, 0)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    img_zero = np.zeros(img.shape, dtype=np.uint8)
    # img[thresh == 255] = 150
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        img_zero[y:y+h, x:x+w] = 255
    # cv2.imshow("mask", mask)
    mask = img_zero
    # cv2.imshow("mask", mask)
    result = cv2.illuminationChange(img, mask, alpha=1, beta=2)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    return result


list_path = os.listdir(root_path)

for path in list_path:
    spath = os.path.join(root_path, path)
    if os.path.isfile(spath):
        if spath[-4:]=='.exr':
            image = cv2.imread(spath)

            #RGB转单通道（如需要gamma变换则改为image_gamma）
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            print("processing "+spath)
            gray_image = gray_image.astype(np.float32)
            cv2.imwrite(spath,gray_image)

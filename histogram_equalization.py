#直方图均衡化（会消除偏振信息）
import os

import OpenEXR
import Imath
import numpy as np
import cv2

def histogram_equalization_clahe(input_file, output_file):
    # 读取JPG图像
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    # 应用CLAHE
    equalized_img = clahe.apply(img)

    # 保存处理后的图像
    cv2.imwrite(output_file, equalized_img)

    print("CLAHE histogram equalization completed. Image saved at:", output_file)

#
root_path = "18"
output_path = "18test1"
if not os.path.exists(output_path):
    os.makedirs(output_path)
imgfiles = os.listdir(root_path)
for i in range(0, len(imgfiles)):
    path = os.path.join(root_path, imgfiles[i])
    print(imgfiles[i])
    if os.path.isfile(path):
        if (imgfiles[i].endswith(".jpg")):
            print("processing " + imgfiles[i])
            input_exr_file = root_path + "/" + imgfiles[i]
            output_exr_file = output_path + "/" + imgfiles[i]
            histogram_equalization_clahe(input_exr_file, output_exr_file)


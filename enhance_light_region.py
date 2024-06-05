#双边滤波+线性增强（效果最好）
import os

import cv2
import numpy as np

def enhance_brightness_and_details(image, brightness=0, contrast=5.0, sigma_color=10, sigma_space=10):
    #α：contrast,β：brightness
    # 使用双边滤波器平滑图像并增加细节
    #smoothed = cv2.bilateralFilter(image, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # 对平滑后的图像进行亮度和对比度的增强
    #enhanced_gray = cv2.convertScaleAbs(smoothed, alpha=contrast, beta=brightness)
    enhanced_gray = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    # 将灰度图像转换为彩色图像
    enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return enhanced_image

#
root_path = "22"
output_path = "22test"
if not os.path.exists(output_path):
    os.makedirs(output_path)
imgfiles = os.listdir(root_path)
for i in range(0, len(imgfiles)):
    path = os.path.join(root_path, imgfiles[i])
    print(imgfiles[i])
    if os.path.isfile(path):
        if (imgfiles[i].endswith(".jpg")):
            print("processing " + imgfiles[i])
            input_file = root_path + "/" + imgfiles[i]
            output_file = output_path + "/" + imgfiles[i]

            # 读取图像
            image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

            enhanced_image = enhance_brightness_and_details(image)

            cv2.imwrite(output_file,enhanced_image)


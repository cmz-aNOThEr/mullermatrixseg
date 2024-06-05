#retinex（会消除偏振信息）
import os

import numpy as np
import cv2


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img / 255.0)
        dst_Lblur = cv2.log(L_blur / 255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8
"""
def SSRmerge(b_gray, g_gray, r_gray,src_img,size):
    b_gray, g_gray, r_gray = cv2.split(src_img)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])
"""
if __name__ == '__main__':
    #
    root_path = "18"
    output_path = "18test"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    imgfiles = os.listdir(root_path)
    for i in range(0, len(imgfiles)):
        path = os.path.join(root_path, imgfiles[i])
        print(imgfiles[i])
        scales = [15, 101, 301]
        if os.path.isfile(path):
            if (imgfiles[i].endswith(".jpg")):
                print("processing " + imgfiles[i])
                input_file = root_path + "/" + imgfiles[i]
                output_file = output_path + "/" + imgfiles[i]

                # 读取图像
                image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

                b_gray, g_gray, r_gray = cv2.split(image)
                b_gray = MSR(b_gray, scales)
                g_gray = MSR(g_gray, scales)
                r_gray = MSR(r_gray, scales)
                enhanced_image = cv2.merge([b_gray, g_gray, r_gray])

                cv2.imwrite(output_file, enhanced_image)

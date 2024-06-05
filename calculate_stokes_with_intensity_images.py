
from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

import os,re

import numpy as np

import polanalyser as pa

import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# 读取36张黑白图像，每张图像代表了不同起偏器和检偏器的角度组合
# 存储JSON文件
output_path = "18test1"

# 用于获取文件名的文件夹路径
folder_path = "18test1"


def generate_colormap(color0, color1):
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[:128] = np.linspace(1, 0, 128)[..., None] * np.array(color0)
    colormap[128:] = np.linspace(0, 1, 128)[..., None] * np.array(color1)
    return np.clip(colormap, 0, 255)

def calculate_stokes():
    # 获取文件夹中所有的exr文件
    exr_files = [file for file in os.listdir(folder_path) if file.endswith('.exr')]

    # 定义正则表达式来匹配文件名中的整数a和b，(-?\d+)代表可能是负数
    pattern = re.compile(r'(-?\d+)_(-?\d+)\.exr')

    # 遍历每个exr文件
    for exr_file in exr_files:
        #读取图像
        image = cv2.imread(os.path.join(folder_path, exr_file), cv2.IMREAD_GRAYSCALE)

        # 尝试从文件名中提取整数a和b
        match = pattern.match(exr_file)
        if match:
            # 如果匹配成功，提取polarizer_angle和analyzer_angle的值
            polarizer_angle = int(match.group(1))
            analyzer_angle = int(match.group(2))

            # Calculate Stokes vector from intensity images and polarizer angles
            intensity_list = []
            muellers = []
            p_angle = np.deg2rad(polarizer_angle)
            a_angle = np.deg2rad(analyzer_angle)
            mueller = pa.polarizer(0) @ pa.qwp(a_angle)

            intensity_list.append(image)
            muellers.append(mueller)

    img_stokes = pa.calcStokes(intensity_list, muellers)

    stokes_I = img_stokes[:,:,0]
    stokes_Q = img_stokes[:,:,1]
    stokes_U = img_stokes[:,:,2]
    stokes_V = img_stokes[:,:,3]

    print(img_stokes[220, 220, :])

    img_dolp = np.sqrt(stokes_Q ** 2 + stokes_U ** 2) / stokes_I

    # Custom colormap (Positive -> Green, Negative -> Red)
    custom_colormap = generate_colormap((0, 0, 255), (0, 255, 0))

    # Apply colormap or adjust the brightness to export images
    img_s0_u8 = pa.applyColorMap(stokes_I, "viridis", vmin=0, vmax=np.max(stokes_I))
    img_s1_u8 = pa.applyColorMap(stokes_Q, "viridis", vmin=-1, vmax=1)  # normalized by s0
    img_s2_u8 = pa.applyColorMap(stokes_U, "viridis", vmin=-1, vmax=1)  # normalized by s0
    img_s3_u8 = pa.applyColorMap(stokes_V, "viridis", vmin=-1, vmax=1)  # normalized by s0

    # Apply colormap or adjust the brightness to export images
    img_dolp_u8 = np.clip(img_dolp * 255, 0, 255).astype(np.uint8)
    img_dolp_u8 = pa.applyColorMap(img_dolp_u8, "viridis", vmin=-1, vmax=1)
    # 将DoLP值映射到灰度范围
    # dolp_mapped =(dolp * 255).astype(np.uint8)
    #cv2.imwrite("s0.png", img_s0_u8)
    #cv2.imwrite("s1.png", img_s1_u8)
    #cv2.imwrite("s2.png", img_s2_u8)
    #cv2.imwrite(output_path + f"dolp.png", img_dolp_u8)


    # 添加颜色条的函数
    def add_colorbar(img, cmap, vmin, vmax, filename):
        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        plt.savefig(filename)
        plt.close()

    # 添加颜色条并保存每张图像
    add_colorbar(img_s0_u8, cm.viridis, 0, np.max(stokes_I), output_path + "/stokes_I_colormap.png")
    add_colorbar(img_s1_u8, cm.viridis, -1, 1, output_path + "/stokes_Q_colormap.png")
    add_colorbar(img_s2_u8, cm.viridis, -1, 1, output_path + "/stokes_U_colormap.png")
    add_colorbar(img_s3_u8, cm.viridis, -1, 1, output_path + "/stokes_V_colormap.png")
    add_colorbar(img_dolp_u8, cm.viridis, -1, 1, output_path + "/dolp_colormap.png")
    cv2.imshow("1", img_dolp_u8)

    cv2.waitKey(0)

calculate_stokes()
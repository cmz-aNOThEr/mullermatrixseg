from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2
import numpy as np

#计算灰度图像的dolp
# 读取两个正交方向的灰度图像
image1 = cv2.imread('dataset/png/apple/blue_0.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('dataset/png/apple/blue_90.png', cv2.IMREAD_GRAYSCALE)

# 计算DoLP值
dolp = np.abs(image1.astype(float) - image2.astype(float)) / \
       np.abs(image1.astype(float) + image2.astype(float) + 1e-8)
def generate_colormap(color0, color1):
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[:128] = np.linspace(1, 0, 128)[..., None] * np.array(color0)
    colormap[128:] = np.linspace(0, 1, 128)[..., None] * np.array(color1)
    return np.clip(colormap, 0, 255)

# Custom colormap (Positive -> Green, Negative -> Red)
custom_colormap = generate_colormap((0, 0, 255), (0, 255, 0))

# Apply colormap or adjust the brightness to export images
img_dolp_u8 = np.clip(img_dolp * 255, 0, 255).astype(np.uint8)
# 将DoLP值映射到灰度范围
#dolp_mapped =(dolp * 255).astype(np.uint8)

# 显示结果
cv2.imwrite('dataset/png/apple/blue_dolp.png', dolp_mapped)
cv2.waitKey(0)
cv2.destroyAllWindows()
from os import environ

from PIL import Image
import numpy as np
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import cv2 as cv
import polanalyser as pa

# Read all images
path = "toy_example_3x3_pc"
pcontainer = pa.PolarizationContainer(path)
image_list = pcontainer.get_list("image")

print(image_list[0].shape)  # (2048, 2448)

# Read all images
path = "3test"
pcontainer = pa.PolarizationContainer(path)
image_list = pcontainer.get_list("image")

print(image_list[0].shape)  # (2048, 2448)

img = cv.imread('3test/5_-25.exr')
print(img.shape)
img2 = cv.imread('toy_example_3x3_pc/image01.exr')
print(img2.shape)

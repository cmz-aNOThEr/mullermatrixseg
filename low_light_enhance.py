#暗部增强（会出现马赛克，不考虑使用）
import os

import OpenEXR
import Imath
import cv2
import numpy as np

def enhance_dark_regions(exr_file_path, output_file_path, enhancement_factor):
    # 读取EXR图像
    exr_file = OpenEXR.InputFile(exr_file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # 读取EXR图像通道数据
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_names = header['channels'].keys()
    channel_data = {}
    for channel in channel_names:
        data_str = exr_file.channel(channel, FLOAT)
        channel_data[channel] = np.fromstring(data_str, dtype=np.float32)
        channel_data[channel] = np.reshape(channel_data[channel], (size[1], size[0]))

    # 对暗部进行增强
    enhanced_data = {}
    for channel in channel_names:
        enhanced_data[channel] = np.log(1 + enhancement_factor * channel_data[channel])

    # 保存增强后的EXR图像
    output_exr = OpenEXR.OutputFile(output_file_path, header)
    output_exr.writePixels(enhanced_data)

    print("Enhanced image saved at:", output_file_path)


def test():
    rootpath = "18test"
    savepath = "18test"
    imgfiles = os.listdir(rootpath)
    for i in range(0, len(imgfiles)):
        path = os.path.join(rootpath, imgfiles[i])
        print(imgfiles[i])
        if os.path.isfile(path):
            if (imgfiles[i].endswith("exr")):
                # outputImg = outputImg * 255.0
                # 示例：增强暗部，增强因子为2
                print("processing" + imgfiles[i])
                enhance_dark_regions(rootpath + "/" + imgfiles[i], savepath + "/" + imgfiles[i], enhancement_factor=5)


if __name__ == '__main__':
    test()
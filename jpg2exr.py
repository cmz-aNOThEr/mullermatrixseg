import os
import cv2
import OpenEXR
import Imath
import numpy as np

def jpg_to_exr():
    input_folder = "22test"
    output_folder = "22test"
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(input_folder)

    # 获取文件夹下所有的JPG文件
    files = os.listdir(input_folder)
    jpg_files = [f for f in files if f.endswith('.jpg')]
    print(files)

    for jpg_file in jpg_files:
        jpg_path = os.path.join(input_folder, jpg_file)
        output_exr_path = os.path.join(output_folder, os.path.splitext(jpg_file)[0] + '.exr')

        # 读取JPG图像
        img = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
        print("read ", jpg_path)


        # 将图像数据转换为浮点类型
        img_float = img.astype(np.float32) / 255.0

        # 创建EXR文件
        header = OpenEXR.Header(img.shape[1], img.shape[0])
        exr_file = OpenEXR.OutputFile(output_exr_path, header)
        print("create ", output_exr_path)

        # 将图像数据写入EXR文件
        red = img_float[:,:,0].tostring()
        green = img_float[:,:,0].tostring()
        blue = img_float[:,:,0].tostring()
        exr_file.writePixels({'R': red, 'G': green, 'B': blue})

        #如果是灰度图会报错：通道数不匹配，改为以下两行
        #img = img_float.tostring()
        #exr_file.writePixels({'R': img})

        print("Converted", jpg_path, "to", output_exr_path)

    print("Conversion completed.")

# 示例：将input_folder文件夹下的所有JPG文件转换为EXR文件，并保存到output_folder文件夹中
jpg_to_exr()

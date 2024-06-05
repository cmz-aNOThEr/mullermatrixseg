import os
from PIL import Image as img
from tqdm import tqdm


path = '3test/'  # 待转换格式的图片所在文件夹
path2 = '3test/'  # 转换后的图片存储路径

files = os.listdir(path)
for n, filename in tqdm(enumerate(files), total=len(files)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # print(filename)
        png = img.open(path + filename)
        file_string = os.path.splitext(filename)[0]
        temp = file_string.split('.')  # 在 ‘.’ 处分割字符串
        png.save(path2 + temp[0] + ".bmp")  # 转换jpg格式就写 “.jpg”

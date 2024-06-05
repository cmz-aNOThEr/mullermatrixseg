#使用掩模修复高光
from os import environ

import numpy as np

environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import  cv2
import os,shutil
#找亮光位置
def create_mask(imgpath):
    image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return mask
#修复图片
def illum (imgpath, maskpath):
    src_ = cv2.imread(imgpath)
    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    #缩放因子(fx,fy)
    res_ = cv2.resize(src_,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    mask = cv2.resize(mask,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    dst = cv2.inpaint(res_, mask, 10, cv2.INPAINT_TELEA)
    return dst

if __name__=='__main__':
    rootpath = "no224"
    masksavepath="mask"
    savepath = "no224"
    imgfiles = os.listdir(rootpath)
    for i in range(0, len(imgfiles)):
        path = os.path.join(rootpath, imgfiles[i])
        print(imgfiles[i])
        if os.path.isfile(path):
            if (imgfiles[i].endswith("bmp")):
                maskpath =os.path.join(masksavepath, "mask_"+imgfiles[i])
                mask = create_mask(path)
                mask = mask.astype(np.float32)
                cv2.imwrite(maskpath, mask)
                dst=illum(path, maskpath)
                dst = dst.astype(np.float32)
                newname = 'illum_' + imgfiles[i].split(".")[0]
                cv2.imwrite(os.path.join(savepath, newname + ".bmp"), dst)
                #shutil.copyfile(os.path.join(rootpath, imgfiles[i].split(".")[0] + ".xml"),
                                #os.path.join(savepath, newname + ".xml"))

#读取文件中所有”a_b.exr“文件的文件名,使用正则表达式提取，获得起偏器角度a,检偏器角度b
#计算a,b角度PSG,PSA的穆勒矩阵，并存储于名为“a_b.json”的JSON文件中

import math
import numpy as np
import json
import os,re

# 定义一个函数来进行矩阵乘法
def matrix_multiply(matrix1, matrix2):
    # 确保矩阵尺寸匹配
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("矩阵尺寸不匹配，无法相乘")

    # 初始化结果矩阵
    result = np.zeros((len(matrix1), len(matrix2[0])))

    # 逐行逐列进行乘法累加
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result


# 存储JSON文件
output_json_path = "18test1"

# 用于获取文件名的文件夹路径
folder_path = "20"

# 获取文件夹中所有的exr文件
exr_files = [file for file in os.listdir(folder_path) if file.endswith('.exr')]

# 定义正则表达式来匹配文件名中的整数a和b，(-?\d+)代表可能是负数
pattern = re.compile(r'(-?\d+)_(-?\d+)\.exr')

# 遍历每个exr文件
for exr_file in exr_files:
    # 尝试从文件名中提取整数a和b
    match = pattern.match(exr_file)
    if match:
        # 如果匹配成功，提取a和b的值
        theta1 = int(match.group(1))
        theta2 = int(match.group(2))

        c1 = math.cos(2*theta1)
        s1 = math.sin(2*theta1)
        c2 = math.cos(2*theta2)
        s2 = math.sin(2*theta2)

        c1_sqr = math.pow(c1,2)
        s1_sqr = math.pow(s1,2)
        c2_sqr = math.pow(c2,2)
        s2_sqr = math.pow(s2,2)


        # 示例矩阵
        matrix1 = [[0.5, 0.5, 0, 0],
                   [0.5, 0.5, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]

        matrix2 = [[1, 0, 0, 0],
                   [0, c2_sqr, s2 * c2, -s2],
                   [0, s2 * c2, s2_sqr, c2],
                   [0, s2,-c2,0]]

        matrix3 =  [[1, 0, 0, 0],
                   [0, c1_sqr, s1 * c1, -s1],
                   [0, s1 * c1, s1_sqr, c1],
                   [0, s1,-c1,0]]

        # 调用函数进行矩阵乘法,计算PSA，PSG的穆勒矩阵
        result_PSA = matrix_multiply(matrix1, matrix2)
        result_PSG = matrix_multiply(matrix3, matrix1)

        print("矩阵相乘的结果：")
        print(result_PSA)
        print(result_PSG)

        # 将ndarrays转换为Python的列表
        mueller_psa_list = result_PSA.tolist()
        mueller_psg_list = result_PSG.tolist()


        # 创建一个名为'a.json'的文件路径
        file_path = os.path.join(output_json_path, "{}".format(theta1) + "_" + "{}".format(theta2) + ".json")

        # 检查文件是否存在
        if os.path.exists(file_path):
            # 如果文件存在，读取文件中的数据
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            # 如果文件不存在，则创建一个空的数据结构
            data = {}

        # 更改写入JSON文件的数据
        json_data = {
            "filename": "{}".format(theta1)+"_"+"{}".format(theta2)+".exr",
            "mueller_psa": {
                "type": "ndarray",
                "values": mueller_psa_list
            },
            "mueller_psg": {
                "type": "ndarray",
                "values": mueller_psg_list
            }
        }

        # 将数据写入JSON文件
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=4)

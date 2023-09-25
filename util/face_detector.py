import os
# from tensorflow import torch
import torch.nn as nn
import tensorflow as tf

from PIL import Image
from mtcnn import MTCNN

import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2

def save_txt_with_image(output_path,parameter_famele):
    # txt_content = "1girl,real face"
    if parameter_famele:
        txt_content = "a woman,real face"
    else:
        txt_content = "a man,real face"

    txt_path = os.path.splitext(output_path)[0] + ".txt"
    with open(txt_path, 'w') as txt_file:
        txt_file.write(txt_content)

def detect_faces(image_path, save_path,parameter_famele,crop_size=(512, 512)):
    # 加载人脸检测器的预训练模型
    # 初始化MTCNN模型
    # detector = MTCNN()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    # 读取图像
    img = cv2.imread(image_path)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 使用人脸检测器检测人脸
    # faces = detector.detect_faces(img_rgb)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))

    # 定义 scale_factor 大小
    # scale_factor = 1.5

    # 裁剪并保存人脸
    for (x, y, w, h) in faces:
        # 计算裁剪区域的边界
        center_x = x + w // 2
        center_y = y + h // 2
        half_width = crop_size[0] // 2
        half_height = crop_size[1] // 2
        roi_x1 = center_x - half_width
        roi_y1 = center_y - half_height
        roi_x2 = roi_x1 + crop_size[0]
        roi_y2 = roi_y1 + crop_size[1]

        # 裁剪并保存人脸
        face = img[roi_y1:roi_y2, roi_x1:roi_x2]
        cv2.imwrite(save_path, face)
        save_txt_with_image(save_path,parameter_famele)

# def detect_faces(image_path, save_path, parameter_famele,crop_size=(512, 512)):
#     try:
#         # 设置特定GPU
#         # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#         # os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
#         # 创建 MTCNN 模型
#         model = MTCNN()
#
#         # 使用 OpenCV 读取图像
#         image = cv2.imread(image_path)
#
#         # 转换图像通道顺序（如果需要）
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # 执行人脸检测
#         faces = model.detect_faces(image)
#
#         # # 检测到的人脸数量
#         # num_faces = len(faces)
#         #
#         # if num_faces == 0:
#         #     print("未检测到人脸")
#         #     return
#         # elif num_faces > 1:
#         #     print("检测到多张人脸，选择第一张")
#
#         # # 获取第一张人脸的位置
#         # x, y, w, h = faces[0]['box']
#         #
#         # # 计算裁剪区域的边界
#         # center_x = x + w // 2
#         # center_y = y + h // 2
#         # half_width = crop_size[0] // 2
#         # half_height = crop_size[1] // 2
#         # roi_x1 = center_x - half_width
#         # roi_y1 = center_y - half_height
#         # roi_x2 = roi_x1 + crop_size[0]
#         # roi_y2 = roi_y1 + crop_size[1]
#
#         for face in faces:
#             x, y, w, h = face['box']  # 获取人脸位置信息
#             cropped_face = image[y:y + h, x:x + w]  # 裁剪人脸图像
#             # 在此处可以对裁剪后的人脸图像进行进一步处理或保存
#
#             # 裁剪并保存人脸
#             # face = image[roi_y1:roi_y2, roi_x1:roi_x2]
#             cv2.imwrite(save_path, cropped_face)
#
#         # 保存文本文件
#         save_txt_with_image(save_path,parameter_famele)
#
#         print("人脸已裁剪并保存到", save_path)
#
#         # del model  # 删除模型对象
#         # torch.cuda.empty_cache()  # 清空GPU缓存
#
#     except Exception as e:
#         print("发生错误:", str(e))
#
#     finally:
#         # 释放模型和显存
#         del model  # 删除模型对象
#         tf.keras.backend.clear_session()

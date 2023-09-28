import os
# from tensorflow import torch
import dlib
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
        txt_content = "a woman,real face,black background"
    else:
        txt_content = "a man,real face,black background"

    txt_path = os.path.splitext(output_path)[0] + ".txt"
    with open(txt_path, 'w') as txt_file:
        txt_file.write(txt_content)

# def detect_faces(image_path, save_path,parameter_famele,crop_size=(512, 512)):
#     # 加载人脸检测器的预训练模型
#     # 初始化MTCNN模型
#     # detector = MTCNN()
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
#
#     # 读取图像
#     img = cv2.imread(image_path)
#     # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
#     # 使用人脸检测器检测人脸
#     # faces = detector.detect_faces(img_rgb)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))
#
#     # 定义 scale_factor 大小
#     # scale_factor = 1.5
#
#     # 裁剪并保存人脸
#     for (x, y, w, h) in faces:
#         # 计算裁剪区域的边界
#         center_x = x + w // 2
#         center_y = y + h // 2
#         half_width = crop_size[0] // 2
#         half_height = crop_size[1] // 2
#         roi_x1 = center_x - half_width
#         roi_y1 = center_y - half_height
#         roi_x2 = roi_x1 + crop_size[0]
#         roi_y2 = roi_y1 + crop_size[1]
#
#         # 裁剪并保存人脸
#         face = img[roi_y1:roi_y2, roi_x1:roi_x2]
#
#         face = remove_bg(face)
#         cv2.imwrite(save_path, face)
#         save_txt_with_image(save_path,parameter_famele)


def detect_faces(image_path, save_path, parameter_famele,crop_size=(512, 512)):
    try:
        # 先扣人像
        face=remove_bg(image_path)

        # 使用dlib检测人脸
        detector = dlib.get_frontal_face_detector()
        faces = detector(face)

        # face_rgb = face[:, :, :3]  # 取前三个通道 (RGB)
        # faces = detector(cv2.cvtColor(face_rgb, cv2.COLOR_BGR2GRAY))

        if len(faces) > 0:
            x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
            cropped_face = face[y:y + h, x:x + w]

            # 调整大小为512x512
            resized_face = resize_image(cropped_face, 512,512)

            # 使用Pillow保存带有透明背景的PNG图像
            pil_image = Image.fromarray(cv2.cvtColor(resized_face, cv2.COLOR_BGRA2RGBA))
            pil_image.save(save_path)

            # 保存文本文件
            save_txt_with_image(save_path, parameter_famele)
        else:
            resized_face = resize_image(face, 512,512)

            cv2.imwrite(save_path, resized_face)
            save_txt_with_image(save_path,parameter_famele)

            print("No face detected.")

        print("人脸已裁剪并保存到", save_path)

    except Exception as e:
        print("发生错误:", str(e))


# -------------------------------------------------------------------------------
# 扣出人脸
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101


def remove_bg(image_path):
    model = deeplabv3_resnet101(pretrained=True).eval()

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = resize_image(image, 1024, 1024)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    output_predictions = torch.argmax(output, dim=0)

    mask = output_predictions == 15
    mask_np = mask.cpu().numpy().astype(np.uint8)
    mask_colored = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
    no_bg = image * mask_colored

    # 清理显存
    del input_tensor, output, output_predictions
    torch.cuda.empty_cache()

    model = None
    import gc
    gc.collect()

    return no_bg

def resize_image(image, width, height):
    """调整图像大小并保持比例。"""
    aspect_ratio = image.shape[1] / float(image.shape[0])
    target_aspect_ratio = width / float(height)

    if aspect_ratio > target_aspect_ratio:
        # 以宽度为基准进行缩放
        scale = width / float(image.shape[1])
    else:
        # 以高度为基准进行缩放
        scale = height / float(image.shape[0])

    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = 255 * np.ones(shape=[height, width, image.shape[2]], dtype=np.uint8)  # 根据输入图像的通道数创建背景

    y_offset = (height - resized_image.shape[0]) // 2
    x_offset = (width - resized_image.shape[1]) // 2

    canvas[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image
    return canvas


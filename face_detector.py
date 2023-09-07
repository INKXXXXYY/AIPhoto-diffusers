import os

from mtcnn import MTCNN
import cv2

def save_txt_with_image(output_path):
    txt_content = "1girl,real face"
    txt_path = os.path.splitext(output_path)[0] + ".txt"
    with open(txt_path, 'w') as txt_file:
        txt_file.write(txt_content)

def detect_faces(image_path, save_path,crop_size=(512, 512)):

    # 加载人脸检测器的预训练模型
    # 初始化MTCNN模型
    detector = MTCNN()

    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 使用人脸检测器检测人脸
    faces = detector.detect_faces(img_rgb)

    # 定义 scale_factor 大小
    scale_factor = 1.5

    # 裁剪并保存人脸
    for face in faces:
        confidence = face['confidence']
        if confidence < 0.9:
            continue

        x, y, w, h = face['box']

        # 计算裁剪区域的边界
        center_x = x + w // 2
        center_y = y + h // 2

        # 计算缩放后的人脸尺寸
        scaled_width = int(w * scale_factor)
        scaled_height = int(h * scale_factor)

        # 计算缩放后的人脸区域的边界
        roi_x1 = max(0, center_x - scaled_width // 2)
        roi_y1 = max(0, center_y - scaled_height // 2)
        roi_x2 = min(img_rgb.shape[1], roi_x1 + scaled_width)
        roi_y2 = min(img_rgb.shape[0], roi_y1 + scaled_height)

        # 缩放并截取人脸区域
        face = cv2.resize(img_rgb[roi_y1:roi_y2, roi_x1:roi_x2], crop_size, interpolation=cv2.INTER_LINEAR)

        if face.size > 0:
            # 创建一个新窗口并显示裁剪后的人脸图像
            # cv2.namedWindow('Cropped Face', cv2.WINDOW_NORMAL)
            # cv2.imshow('Cropped Face', face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 裁剪并保存人脸
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 转换为 RGB 色彩空间
            cv2.imwrite(save_path, rgb_face)
            save_txt_with_image(save_path)  # 保存.txt文件

        else:
            print("无法检测到人脸")



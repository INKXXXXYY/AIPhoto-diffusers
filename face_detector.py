import os
#from mtcnn import MTCNN
import cv2

def save_txt_with_image(output_path):
    txt_content = "1girl,real face"
    txt_path = os.path.splitext(output_path)[0] + ".txt"
    with open(txt_path, 'w') as txt_file:
        txt_file.write(txt_content)

def detect_faces(image_path, save_path,crop_size=(512, 512)):

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
        save_txt_with_image(save_path)



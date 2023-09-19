from werkzeug.utils import secure_filename
from face_detector import detect_faces
import os

def save_photo_to_server(user_id, photos, upload_folder, pretech_folder):
    # 构建保存路径
    save_path = os.path.join(upload_folder, user_id)
    save_pre_path = os.path.join(pretech_folder, user_id, "10_face/",)

    # 检查文件夹是否存在，如果不存在则创建它
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_pre_path, exist_ok=True)

    for photo in photos:
        if photo.filename == '':
            continue  # 跳过没有文件名的字节对象

        # 获取上传照片的文件名
        filename = secure_filename(photo.filename)

        # 写入照片数据到本地文件
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(photo.read())

        # 构建处理后的文件名
        processed_filename = 'processed_' + filename

        # 执行人脸检测并保存结果图像
        detect_faces(os.path.join(save_path, filename), os.path.join(save_pre_path, processed_filename))

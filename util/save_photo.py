import os

import torch
from werkzeug.utils import secure_filename

from util.face_detector import detect_faces


def save_photos_to_server(user_id, param_name,file_list,parameter_famele, upload_folder, pretech_folder):
    # 构建保存路径
    save_path = os.path.join(upload_folder, user_id,str(param_name))
    save_pre_path = os.path.join(pretech_folder, user_id, str(param_name),"10_face/")

    # 检查文件夹是否存在，如果不存在则创建它
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_pre_path, exist_ok=True)

    for file in file_list:
        if file.filename == '':
            continue  # 跳过没有文件名的字节对象

        # 获取上传照片的文件名
        filename = secure_filename(file.filename)

        # 更改文件扩展名为 .png
        basename, ext = os.path.splitext(filename)
        filename_png = basename + '.png'

        # 写入文件数据到本地文件---
        file.save(os.path.join(save_path, filename_png))

        # 构建处理后的文件名
        processed_filename = 'processed_' + filename_png

        # 执行人脸检测并保存结果图像
        detect_faces(os.path.join(save_path, filename_png), os.path.join(save_pre_path, processed_filename),parameter_famele)
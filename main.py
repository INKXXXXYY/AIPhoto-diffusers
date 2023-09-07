import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from face_detector import detect_faces

app = Flask(__name__)

# 保存上传照片的目录
UPLOAD_FOLDER = 'upload/origin'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PRETECT_FOLDER = 'upload/detected'
app.config['PRETECT_FOLDER'] = PRETECT_FOLDER


@app.route('/upload', methods=['POST'])
def upload_photo():
    try:
        content_type = request.content_type
        if 'multipart/form-data' not in content_type:
            return jsonify({'error': 'Invalid Content-Type'})

        if 'photo' not in request.files:
            return jsonify({'error': 'No photo uploaded'})

        photos = request.files.getlist('photo')
        print(photos)

        # 保存照片到服务器
        save_photo_to_server(photos)

        # 返回处理后的结果
        result = {'message': 'Photo processed successfully'}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


def save_photo_to_server(photos):
    for photo in photos:
        if photo.filename == '':
            continue  # 跳过没有文件名的字节对象

        # 获取上传照片的文件名
        filename = secure_filename(photo.filename)

        # 构建保存路径
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 写入照片数据到本地文件
        with open(save_path, 'wb') as f:
            f.write(photo.read())

        # 构建处理后的文件名
        processed_filename = 'processed_' + filename

        # 构建保存路径
        save_pre_path = os.path.join(app.config['PRETECT_FOLDER'], processed_filename)

        # 执行人脸检测并保存结果图像
        detect_faces(save_path, save_pre_path)


if __name__ == '__main__':
    app.run()

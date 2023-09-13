import os
import select

from flask import Flask, request, jsonify,send_file
from werkzeug.utils import secure_filename
import subprocess

from face_detector import detect_faces
from fashi_model import generate_image
from adetailer import generate_ad_output
from super_resolution import upscale_image

import time


app = Flask(__name__)

# 保存上传照片的目录
UPLOAD_FOLDER = 'upload/origin'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PRETECT_FOLDER = '/root/autodl-tmp/lora-scripts-main/train/face'
app.config['PRETECT_FOLDER'] = PRETECT_FOLDER


@app.route('/upload', methods=['POST'])
def upload_photo():
    start_time = time.time()
    try:
        # 獲取上傳的照片
        content_type = request.content_type
        if 'multipart/form-data' not in content_type:
            return jsonify({'error': 'Invalid Content-Type'})

        if 'photo' not in request.files:
            return jsonify({'error': 'No photo uploaded'})

        photos = request.files.getlist('photo')
        # print(photos)

        print("---------------------------")
        print("将照片保存到服务器中")
        # 保存照片到服务器中，方便後續處理
        save_photo_to_server(photos)

        # 訓練人臉lora
        print("---------------------------")
        print("开始训练人脸lora......")
        # train_lora()


        print("---------------------------")
        print("调用风格模型进行推理")

        # 調用風格模型
        generate_image()

        print("---------------------------")
        print("高清修复中......")


        # 高清修复
        upscale_image()

        print("---------------------------")
        print("使用adetail进行人脸修复......")

        # 調用adetail對人臉進行修復
        generate_ad_output()

        # 返回处理后的结果
        # result = {'message': 'Photo processed successfully'}
        end_time = time.time()
        execution_time = end_time - start_time

        print("接口调用到返回的时间：", execution_time, "秒")

        return send_file('final.png', mimetype='image/png')

        # return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


def save_photo_to_server(photos):
    print("---------------------------")
    print("人脸数据预处理......")
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
        save_pre_path = os.path.join(app.config['PRETECT_FOLDER'], "10_face/", processed_filename)


        # 执行人脸检测并保存结果图像
        detect_faces(save_path, save_pre_path)

def train_lora():
    # 开始训练lora
    print("开始训练人脸lora")

    # 执行 bash 命令并输出结果
    # result = subprocess.run(
    #     ["bash", "-c", "cd /root/autodl-tmp/lora-scripts-main && bash train.sh"],
    #     capture_output=True, text=True, shell=True)

    # 切换目录到 /root/autodl-tmp/lora-scripts-main
    os.chdir("/root/autodl-tmp/lora-scripts-main")

    # 获取当前目录地址并输出
    curr_dir = os.getcwd()
    print(f"当前目录地址: {curr_dir}")

    # 执行 bash train.sh 命令
    # train_cmd = ["bash", "-c", "bash train.sh"]
    # result = subprocess.run(train_cmd, capture_output=True, text=True, shell=True, env=os.environ)
    result = os.system("bash train.sh")
    print(result)

    # 输出结果
    print("标准输出：", result.stdout)
    print("错误输出：", result.stderr)
    print("返回码：", result.returncode)




if __name__ == '__main__':
    app.run(port=6006)


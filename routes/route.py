from flask import Blueprint, request, jsonify, send_file, current_app
import uuid
import time

import os
import cv2

from src.infer import infer
from src.lora import train_lora
from util.get_upload_img import get_all_file
from util.image_util import get_first_image
from src.face_fusion import image_face_fusion
from util.virtualenv_util import activate_virtual_environment


single_upload_bp = Blueprint('single', __name__)
people_upload_bp = Blueprint('people', __name__)

@single_upload_bp.route('/upload', methods=['POST'])
# @app.route('/upload', methods=['POST'])
def upload_photo():
    start_time = time.time()
    user_id = str(uuid.uuid4())

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
        # save_photo_to_server(user_id,photos,app.config['UPLOAD_FOLDER'], app.config['PRETECT_FOLDER'])

        # 訓練人臉lora
        print("---------------------------")
        print("开始训练人脸lora......")
        # train_lora()


        print("---------------------------")
        print("调用风格模型进行推理")

        # 調用風格模型
        # process_image()

        print("---------------------------")
        print("高清修复中......")


        # 高清修复
        # upscale_image()

        print("---------------------------")
        print("使用adetail进行人脸修复......")

        # 調用adetail對人臉進行修復
        # generate_ad_output()

# -------------------------------------------------------------------
        # 調用face_fusion進行人臉融合
        print("切換環境並調用face_fusion")

        import cv2
        from util.image_util import get_first_image
        from src.face_fusion import image_face_fusion
        from util.virtualenv_util import activate_virtual_environment
        # 切换虚拟环境
        # 调用封装的函数
        path = '/root/autodl-tmp/lora-scripts-main/train/face/10_face'
        user_path = get_first_image(path)
        # print(user_path)
        template_path = 'final.png'

        # 调用工具函数来切换虚拟环境
        virtual_env_name = "modelscope"
        activate_virtual_environment(virtual_env_name,template_path,user_path)

        # finish

        # 返回处理后的结果
        end_time = time.time()
        execution_time = end_time - start_time

        print("接口调用到返回的时间：", execution_time, "秒")

        return send_file('result.png', mimetype='image/png')

        # return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@people_upload_bp.route('/people_upload', methods=['POST'])
def process_photos(*args):
    start_time = time.time()
    # 生成用户ID
    user_id = str(uuid.uuid4())

    try:
        # 获取上传的两个照片文件
        content_type = request.content_type
        if 'multipart/form-data' not in content_type:
            return jsonify({'error': 'Invalid Content-Type'})

        if 'photo_1' not in request.files:
            return jsonify({'error': 'no photos uploaded'})

        # 获取上传的照片
        print("-----------------------------------------------------------------")
        print("获取上传照片")
        files = request.files

        # 保存上传的文件并进行預处理
        # get_all_file(user_id, len(files),files, current_app.config['UPLOAD_FOLDER'], current_app.config['PRETECT_FOLDER'])

        # 训练人脸Lora
        print("-----------------------------------------------------------------")
        print("開始訓練人臉lora")
        # train_lora(os.path.join(current_app.config['PRETECT_FOLDER'],user_id),len(files),os.path.join(current_app.config['LORA_MODEL'],user_id))

        # 调用风格模型进行推理等处理
        print("---------------------------")
        print("调用风格模型进行推理")
        infer(user_id)

        # 返回处理后的结果
        end_time = time.time()
        execution_time = end_time - start_time

        print("接口调用到返回的时间：", execution_time, "秒")

        return send_file('result.png', mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)})
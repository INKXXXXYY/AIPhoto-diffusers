import os
import uuid
import time

from flask import Flask, request, jsonify,send_file
from util.save_photo import save_photo_to_server
from lora import train_lora


app = Flask(__name__)

# 保存上传照片的目录
UPLOAD_FOLDER = 'upload/origin'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PRETECT_FOLDER = '/root/autodl-tmp/lora-scripts-main/train/'
app.config['PRETECT_FOLDER'] = PRETECT_FOLDER


@app.route('/upload', methods=['POST'])
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
        save_photo_to_server(user_id,photos,app.config['UPLOAD_FOLDER'], app.config['PRETECT_FOLDER'])

        # 訓練人臉lora
        print("---------------------------")
        print("开始训练人脸lora......")
        # train_lora()


        print("---------------------------")
        print("调用风格模型进行推理")

        # 調用風格模型
        # generate_image()
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
        from face_fusion import image_face_fusion
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

@app.route('/people_upload', methods=['POST'])
def process_photos():
    start_time = time.time()

    # 生成用户ID
    user_id = str(uuid.uuid4())

    try:
        # 获取上传的两个照片文件
        content_type = request.content_type
        if 'multipart/form-data' not in content_type:
            return jsonify({'error': 'Invalid Content-Type'})

        if 'photo_girl' not in request.files or 'photo_boy' not in request.files:
            return jsonify({'error': 'Two photos not uploaded'})


        photo_girl = request.files.getlist('photo_girl')
        photo_boy = request.files.getlist('photo_boy')

        # 保存照片到服务器

        print("-----------------------------------------------------------------")
        print("將照片保存到服務器")
        save_photo_to_server(user_id+"_girl",photo_girl,app.config['UPLOAD_FOLDER'], app.config['PRETECT_FOLDER'])
        save_photo_to_server(user_id+"_boy",photo_boy,app.config['UPLOAD_FOLDER'], app.config['PRETECT_FOLDER'])

        # 训练人脸Lora
        print("-----------------------------------------------------------------")
        print("開始訓練人臉lora")
        train_lora(os.path.join(app.config['PRETECT_FOLDER'], user_id+"_girl"),user_id+"_girl")
        train_lora(os.path.join(app.config['PRETECT_FOLDER'], user_id+"_boy"),user_id+"_boy")


        # 调用风格模型进行推理等处理

        # 高清修复等其他处理

        # 返回处理后的结果
        return send_file('result.png', mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(port=6006)


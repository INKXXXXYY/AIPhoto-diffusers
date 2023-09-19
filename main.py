import os
import uuid
import time

import torch
from flask import Flask, request, jsonify,send_file
from ModelManager import ModelManager

from controlnet_model import process_image
from util.get_upload_img import get_all_file
from lora import train_lora
TF_ENABLE_ONEDNN_OPTS=0


app = Flask(__name__)

# 保存上传照片的目录
UPLOAD_FOLDER = '/root/autodl-tmp/aifamily_demo/upload/origin'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PRETECT_FOLDER = '/root/autodl-tmp/aifamily_demo/upload/detected'
app.config['PRETECT_FOLDER'] = PRETECT_FOLDER
#loramodel地址
LORA_MODEL = '/root/autodl-tmp/aifamily_demo/upload/lora_model'
app.config['LORA_MODEL'] = LORA_MODEL


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
def process_photos(*args):
    start_time = time.time()
    # 生成用户ID
    user_id = str(uuid.uuid4())
    manager = ModelManager()

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
        # get_all_file(user_id, len(files),files, app.config['UPLOAD_FOLDER'], app.config['PRETECT_FOLDER'])

        # 训练人脸Lora
        print("-----------------------------------------------------------------")
        print("開始訓練人臉lora")
        #train_lora(user_id,os.path.join(app.config['PRETECT_FOLDER'],user_id),len(files),os.path.join(app.config['LORA_MODEL'],user_id))


        # 调用风格模型进行推理等处理
        print("---------------------------")
        print("调用风格模型进行推理")


        # 添加模型
        manager.add_model("pose_model", "/root/autodl-tmp/controlnet/sd-controlnet-openpose")
        manager.add_model("diffuser_model1", "/root/autodl-tmp/diffusers_model/newchinese/newchinese1_3")
        manager.add_model("diffuser_model2", "/root/autodl-tmp/diffusers_model/newchinese/newchinese2_1")
        manager.add_model("diffuser_model3", "/root/autodl-tmp/diffusers_model/newchinese/newchinese3_1")

        # 修改模型参数
        # manager.modify_model_parameters("diffuser_model1", torch_dtype=torch.float32)

        # 处理图像并调用所有模型
        output_images = manager.process_image_with_models(
            "pose.jpg",
            "pose_model",
            ["diffuser_model1","diffuser_model2","diffuser_model3"],
            "solo,realistic,1girl,smile,French hairstyle,Clear eyes,ultra realistic skin,exquisite facial features,goddess women, Off shoulder dress,Backlit, Contrast Filters, looking at viewer,",
            [
                "Two people,（deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, (worst quality, low quality:2), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, (tree,wood:1.2), (stone:1.2), (green,black,white:1.4), (sandals:1.4),"],
            generator_seed=2,
            num_inference_steps=20,
            width=1024,
            height=1536
        )

        # 保存输出图像
        for i, output_image in enumerate(output_images):
            output_image.save(f"output_{i}.png")

        # manager.add_model("diffuser_model2", "/root/autodl-tmp/diffusers_model/other_model")

        # 高清修复等其他处理

        # 返回处理后的结果
        return send_file('result.png', mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(port=6006)


from flask import Flask, request, jsonify,send_file
from routes.route import single_upload_bp,people_upload_bp

TF_ENABLE_ONEDNN_OPTS=0

app = Flask(__name__)

# 保存上传照片的目录
UPLOAD_FOLDER = '/root/autodl-tmp/aifamily_demo/user/origin'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PRETECT_FOLDER = '/root/autodl-tmp/aifamily_demo/user/detected'
app.config['PRETECT_FOLDER'] = PRETECT_FOLDER
#loramodel地址
LORA_MODEL = '/root/autodl-tmp/aifamily_demo/user/lora_model'
app.config['LORA_MODEL'] = LORA_MODEL

app.register_blueprint(single_upload_bp, url_prefix="")
app.register_blueprint(people_upload_bp, url_prefix="")


if __name__ == '__main__':
    app.run(port=6006)


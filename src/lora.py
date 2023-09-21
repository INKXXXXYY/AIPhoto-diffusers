import json
import subprocess
import os

from functools import partial
import torch
from asdff import AdPipeline, yolo_detector
from diffusers import KDPM2DiscreteScheduler
from diffusers.utils import load_image
import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置使用的 GPU 设备编号


class LoraPipeline:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.base_path = "/root/autodl-tmp"
        self.lora_script_path = os.path.join(self.base_path, "lora-scripts-main")
        self.lora_model_path = os.path.join(self.base_path, "lora_models")
        self.diffusers_model_path = os.path.join(self.base_path, "diffusers_model")

    def train_lora(self, para_num: int):
        if not os.path.exists(self.lora_model_path):
            os.makedirs(self.lora_model_path)

        train_data_dir = os.path.join(
            self.base_path, "aifamily_demo", "user", "detected", self.user_id
        )
        if not os.path.exists(train_data_dir):
            print("error: user data does not exist")

        for num in range(para_num):
            # 构建新的 train_data_dir
            output_path = os.path.join(
                self.lora_model_path, self.user_id, f"{self.user_id}_{num}.safetensors"
            )
            modified_train_data_dir = os.path.join(
                train_data_dir, str(num + 1), "10_face"
            )
            print("预处理后用户数据数据集路径：", modified_train_data_dir)

            result = subprocess.run(
                [
                    "bash",
                    os.path.join(self.lora_script_path, "train.sh"),
                    modified_train_data_dir,
                    str(num),
                    output_path,
                ],
                capture_output=True,
                text=True,
            )
            print(result)
            # 输出结果
            print("标准输出：", result.stdout)

    def generate_ad_output(self):
        # 第一层循环：遍历 self.diffusers_model_path 下的文件夹
        for folder in os.listdir(self.diffusers_model_path):
            first_level_path = os.path.join(self.diffusers_model_path, folder)

            # 判断是否为文件夹
            if os.path.isdir(first_level_path):
                # 第二层循环：遍历子文件夹
                for sub_folder in os.listdir(first_level_path):
                    model_dir = os.path.join(first_level_path, sub_folder)

                    if os.path.isdir(model_dir):  # 确保是文件夹
                        pipe = AdPipeline.from_pretrained(
                            model_dir, torch_dtype=torch.float16
                        )
                        pipe.load_lora_weights(self.lora_output_path)

                        pipe.safety_checker = None

                        lora_w = 1
                        pipe._lora_scale = lora_w

                        state_dict, network_alphas = pipe.lora_state_dict(
                            self.lora_output_path
                        )

                        for key in network_alphas:
                            network_alphas[key] = network_alphas[key] * lora_w

                        pipe.load_lora_into_unet(
                            state_dict=state_dict,
                            network_alphas=network_alphas,
                            unet=pipe.unet,
                        )
                        pipe.load_lora_into_text_encoder(
                            state_dict=state_dict,
                            network_alphas=network_alphas,
                            text_encoder=pipe.text_encoder,
                        )

                        pipe.scheduler = KDPM2DiscreteScheduler.from_config(
                            pipe.scheduler.config
                        )
                        pipe = pipe.to("cuda")

                        person_model_path = os.path.join(
                            self.base_path, "face_yolov8n.pt"
                        )
                        person_detector = partial(
                            yolo_detector, model_path=person_model_path
                        )

                        with open(
                            os.path.join(model_dir, "config.json"),
                            "r",
                            encoding="utf-8",
                        ) as f:
                            common = json.load(f)

                        model_id = common["model_id"]
                        generate_basic_image_path = os.path.join(
                            self.base_path,
                            "user",
                            "output",
                            "generate_basic_image",
                            self.user_id,
                            model_id,
                        )
                        for file in os.listdir(generate_basic_image_path):
                            image_path = os.join(generate_basic_image_path, file)
                            images = load_image(image_path)
                            result = pipe(
                                common=common["config"],
                                images=[images],
                                detectors=[person_detector, pipe.default_detector],
                            )

                        for img in result.images:
                            ad_output_path = os.path.join(
                                self.base_path,
                                "user",
                                "output",
                                "ad_output",
                                self.user_id,
                                model_id,
                                f"{self.user_id}_{model_id}.png"
                            )
                            img.save(ad_output_path)
                            ff_output_path = os.path.join(
                                self.base_path, "user", "output", "ff_output", self.user_id, model_id,
                                f"{self.user_id}_{model_id}.png"
                            )
                            img.save(ad_output_path)
                            self.image_face_fusion(ad_image_path=ad_output_path, output_path=ff_output_path)


    def image_face_fusion(self, ad_image_path:str, output_path:str):
        # 这里假设'pipeline'和'Tasks'已经被正确导入
        image_face_fusion_pipeline = pipeline(
            Tasks.image_face_fusion, model="damo/cv_unet-image-face-fusion_damo"
        )
        # 读取 ad 处理后的照片
        from util.image_util import get_first_image
        user_path = os.path.join(self.base_path, "user", "detected", self.user_id, "1", "10_face")
        user_image_path = get_first_image(user_path)
        result = image_face_fusion_pipeline(
            dict(template=ad_image_path, user=user_image_path)
        )

        result_rgb = cv2.cvtColor(result[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, result_rgb)

        print("finished!")

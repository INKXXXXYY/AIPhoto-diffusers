import io
import json
import os
from datetime import datetime
from typing import List
from PIL import Image
import torch
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DPMSolverMultistepScheduler
import warnings


class ModelManager:
    def __init__(self):
        self.models = {}  # 存储所有模型
        self.model_paths = {}  # 存储模型路径
        self.image_generator = ImageGenerator(self)  # 实例化ImageGenerator类


    def add_model(self, model_id: str, model_path: str):
        # 添加一个模型
        if model_id == "pose_model":
            model = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16)
            self.models[model_id] = model
            self.model_paths[model_id] = model_path
            print("成功添加", model_id)
        else:
            controlnet = self.models.get("pose_model")
            model = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet,
                                                                      torch_dtype=torch.float16)
            self.models[model_id] = model
            self.model_paths[model_id] = model_path
            print("成功添加", model_id)

    def get_model_path(self, model_id: str) -> str:
        # 获取模型路径
        if model_id in self.model_paths:
            return self.model_paths[model_id]
        else:
            raise ValueError("Invalid model ID")

    def remove_model(self, model_id: str):
        # 移除一个模型
        if model_id in self.models:
            del self.models[model_id]

    def modify_model_parameters(self, model_id: str, **kwargs):
        # 修改模型参数
        if model_id in self.models:
            model = self.models[model_id]
            for key, value in kwargs.items():
                setattr(model, key, value)

    def process_image_with_models(self, user_id: str,config_file: str, diffuser_model_ids: List[str], num_inference_steps: int,
                                  width: int, height: int):
        self.image_generator.generate_basic_image(user_id, config_file, diffuser_model_ids, num_inference_steps,
                                            width, height)


class ImageGenerator:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def generate_basic_image(self, user_id: str, config_file: str, diffuser_model_ids: List[str], num_inference_steps: int,
                       width: int, height: int):
        # 读取配置文件
        with open(config_file, "r") as f:
            config = json.load(f)
        print("读取config文件")

        warnings.filterwarnings("ignore", category=FutureWarning)

        # 循环遍历 diffuser_model_ids 列表中的模型ID，并逐步叠加
        # 循环遍历模型的配置，并逐步叠加
        for model_id, model_config in config.items():
            if model_id in diffuser_model_ids:
                print("正在使用", model_id, "进行推理")
                # 处理 diffuser_model
                pose_image_path = model_config["pose_image"]
                pose_image = Image.open(f"{self.model_manager.get_model_path(model_id)}/{pose_image_path}").convert(
                    "RGB")
                print("成功导入骨架图")

                prompt = model_config["prompt"]
                negative_prompt = model_config["negative_prompt"]
                generator_seed = model_config["generator_seed"]
                print(f"成功读取{model_id}的模型参数")

                # 加载 diffuser 模型
                pipe = self.model_manager.models.get(model_id)

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.enable_model_cpu_offload()

                generator = torch.Generator(device="cuda").manual_seed(2)

                # 使用 diffuser_model 处理图像
                try:
                    print(f"Processing image with model {model_id}")
                    # 使用 diffuser_model 处理图像
                    output_image = pipe(
                        prompt if isinstance(prompt, str) else [prompt],  # 确保 prompt 是 str 或包含字符串的列表
                        pose_image,
                        generator=generator,
                        negative_prompt=negative_prompt,
                        # generator_seed=generator_seed,
                        num_inference_steps=num_inference_steps,
                        width=width,
                        height=height
                    ).images[0]

                    # 生成唯一的文件名
                    now = datetime.now()
                    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                    output_image_path = f"user/output/{user_id}/{model_id}_{timestamp_str}.png"

                    # 如果输出目录不存在则创建
                    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                    # 保存输出图像
                    output_image.save(output_image_path)
                    print(f"Processed image saved at {output_image_path}")

                except Exception as e:
                    # 捕获并记录异常
                    error_message = f"An error occurred while processing the image with model {model_id}: {e}"
                    print(error_message)
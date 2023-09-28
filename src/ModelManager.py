import io
import json
import os
from typing import List, Optional
from PIL import Image
import torch
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DPMSolverMultistepScheduler
import warnings

from transformers import CLIPTextModel

from src.ImageGenerator import ImageGenerator


class ModelManager:
    def __init__(self):
        self.models = {}  # 存储所有模型
        self.model_paths = {}  # 存储模型路径
        self.image_generator = ImageGenerator(self)  # 实例化ImageGenerator类


    def add_model(self, model_id: str, model_path: str):
        # 添加一个模型
        if model_id == "pose_model":
            model = ControlNetModel.from_pretrained(model_path, conditioning_scale =1.5,torch_dtype=torch.float16)
            self.models[model_id] = model
            self.model_paths[model_id] = model_path
        elif model_id == "canny_model":
            model = ControlNetModel.from_pretrained(model_path, conditioning_scale =1.5,torch_dtype=torch.float16)
            self.models[model_id] = model
            self.model_paths[model_id] = model_path
        else:
            controlnet = [self.models.get("pose_model"),self.models.get("canny_model")]
            text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
                                                         num_hidden_layers=11, torch_dtype=torch.float16)

            model = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet,text_encoder = text_encoder,

                                                                      torch_dtype=torch.float16)
            # model.text_encoder_name=text_encoder
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

    def process_image_with_models(self, user_id: str,diffuser_model_ids: List[str], num_inference_steps: int,
                                  width: int, height: int):
        self.image_generator.generate_basic_image(user_id, diffuser_model_ids, num_inference_steps,
                                            width, height)

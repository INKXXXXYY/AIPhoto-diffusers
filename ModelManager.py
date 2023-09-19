from typing import List
from PIL import Image
import torch
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import warnings

class ModelManager:
    def __init__(self):
        self.models = {}  # 存储所有模型

    def add_model(self, model_id: str, model_path: str):
        # 添加一个模型
        model = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.models[model_id] = model

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

    def process_image_with_models(self, image_path: str, pose_model_id: str, diffuser_model_ids: List[str], prompt: str,
                                  negative_prompt: List[str], generator_seed: int, num_inference_steps: int,
                                  width: int, height: int):
        # 处理图像并调用所有模型
        image = Image.open(image_path).convert("RGB")

        pose_model = self.models.get(pose_model_id)
        if pose_model is None:
            raise ValueError(f"Pose model with id {pose_model_id} does not exist.")

        pose = pose_model(image)
        pose.save("bone_pose.png")

        warnings.filterwarnings("ignore", category=FutureWarning)

        diffusers = []
        for model_id in diffuser_model_ids:
            diffuser_model = self.models.get(model_id)
            if diffuser_model is None:
                raise ValueError(f"Diffuser model with id {model_id} does not exist.")
            diffusers.append(diffuser_model)

        generator = torch.Generator(device="cuda").manual_seed(generator_seed)

        output_images = []
        for diffuser_model in diffusers:
            pipe = StableDiffusionControlNetPipeline.from_config(diffuser_model.config)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()

            output = pipe(
                [prompt],
                pose,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height
            ).images[0]
            output_images.append(output)

        return output_images

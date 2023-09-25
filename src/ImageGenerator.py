from datetime import datetime
import json
import os
import warnings
from typing import List

import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline
from flask import current_app


class ImageGenerator:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.upscale_model_path="/root/autodl-tmp/4x-UltraSharp.pth"

    def generate_basic_image(self, user_id: str,diffuser_model_ids: List[str], num_inference_steps: int,
                       width: int, height: int):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        # # 读取配置文件
        # with open(config_file, "r") as f:
        #     config = json.load(f)
        # print("读取config文件")

        # warnings.filterwarnings("ignore", category=FutureWarning)

        # 循环遍历 diffuser_model_ids 列表中的模型ID，并逐步叠加
        for model_id in diffuser_model_ids:
            model_config_file = os.path.join(self.model_manager.get_model_path(model_id), f"config.json")
            if not os.path.isfile(model_config_file):
                print(f"No configuration file found for model {model_id}")
                continue

            print("正在使用", model_id, "进行推理")

            # 读取模型的配置
            with open(model_config_file, "r") as f:
                model_config = json.load(f)

            # 处理 diffuser_model
            pose_image_path = model_config["config"]["pose_image"]
            pose_image = Image.open(f"{self.model_manager.get_model_path(model_id)}/{pose_image_path}").convert("RGB")
            print("成功导入骨架图")

            prompt = model_config["config"]["prompt"]
            negative_prompt = model_config["config"]["negative_prompt"]
            generator_seed = model_config["config"]["generator_seed"]
            print(f"成功读取{model_id}的模型参数")

            # 加载 diffuser 模型
            pipe = self.model_manager.models.get(model_id)

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()

            # generator = torch.Generator(device="cuda:1").manual_seed(2)
            generator = torch.Generator(device="cuda").manual_seed(generator_seed)

            # 使用 diffuser_model 处理图像
            try:
                print(f"Processing image with model {model_id}")
                # 使用 diffuser_model 处理图像
                output_image = pipe(
                    prompt if isinstance(prompt, str) else [prompt],  # 确保 prompt 是 str 或包含字符串的列表
                    pose_image,
                    # generator=generator.manual_seed(generator_seed),
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
                # output_image_path = f"user/output/generate_basic_image/{user_id}/{model_id}/{model_id}_{timestamp_str}.png"
                output_image_path = os.path.join(current_app.config['OUTPUT_FOLDER'], "generate_basic_image", user_id, model_id,
                                                 f"{model_id}_{timestamp_str}.png")

                # 如果输出目录不存在则创建
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                # 保存输出图像
                output_image.save(output_image_path)
                # print("qqqqqqqqqqqqqqqqqqqqqqqqqqq")
                print(f"Processed image saved at {output_image_path}")

                print("运行完后释放显存")
                torch.cuda.empty_cache()

                print("--------------------------------------------")
                print("對圖像進行超分修復")
                self.sp_image(user_id,model_id,output_image,prompt,negative_prompt,num_inference_steps,guidance_scale=7.5)


            except Exception as e:
                # 捕获并记录异常
                error_message = f"An error occurred while processing the image with model {model_id}: {e}"
                print(error_message)

    # def sp_image(self,user_id,model_id,image,prompt,negative_prompt,num_inference_steps,guidance_scale=7.5):
    #
    #     warnings.filterwarnings("ignore", category=FutureWarning)
    #
    #     print("正在使用", model_id, "进行高清修復")
    #
    #     # 加载 diffuser 模型
    #     pipe = self.model_manager.models.get(model_id)
    #     pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_manager.get_model_path(model_id), torch_dtype=torch.float16)
    #
    #     # 将模型转移到GPU上进行推理
    #     # device = torch.device("cuda")
    #     # pipe.load_textual_inversion("root/autodl-tmp/4x-UltraSharp.pth")
    #     pipe = pipe.to("cuda")
    #
    #     # 使用 diffuser_model 处理图像
    #     try:
    #         # 打开图像
    #         # image = Image.open(image_path)
    #         # 放大算法
    #         image = image.resize((1024, 1536))
    #         # image=self.upscale_image(user_id, self.upscale_model_path, image)
    #
    #         print("输入照片进行修复")
    #         images = pipe(
    #             prompt=prompt,
    #             negative_prompt=negative_prompt,
    #             image=image,
    #             strength=0.3,
    #             guidance_scale=guidance_scale,
    #             num_inference_steps=num_inference_steps,
    #             # width=1024,
    #             # height=1536
    #         ).images[0]
    #
    #         # 生成唯一的文件名
    #         now = datetime.now()
    #         timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    #         # output_image_path = f"user/output/upscale_image/{user_id}/{model_id}/super_{model_id}_{timestamp_str}.png"
    #         output_image_path = os.path.join(current_app.config['OUTPUT_FOLDER'], "upscale_image", user_id, model_id,
    #                                          f"super_{model_id}_{timestamp_str}.png")
    #
    #         # 如果输出目录不存在则创建
    #         os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    #
    #         # 保存输出图像
    #         images.save(output_image_path)
    #         print(f"Processed image saved at {output_image_path}")
    #
    #
    #     except Exception as e:
    #         # 捕获并记录异常
    #         error_message = f"An error occurred while processing the image with model {model_id}: {e}"
    #         print(error_message)


    def sp_image(self,user_id,model_id,image,prompt,negative_prompt,num_inference_steps,guidance_scale=7.5):

        warnings.filterwarnings("ignore", category=FutureWarning)

        print("正在使用", model_id, "进行高清修復")

        # 加载 diffuser 模型
        model_id = "root/autodl-tmp/stable-diffusion-x4-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        # 将模型转移到GPU上进行推理
        # device = torch.device("cuda")
        # pipe.load_textual_inversion("root/autodl-tmp/4x-UltraSharp.pth")
        pipeline = pipeline.to("cuda")

        # 使用 diffuser_model 处理图像
        try:
            # 打开图像
            # image = Image.open(image_path)
            # 放大算法
            image = image.resize((1024, 1536))
            # image=self.upscale_image(user_id, self.upscale_model_path, image)

            print("输入照片进行修复")
            images = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=0.3,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                # width=1024,
                # height=1536
            ).images[0]

            # 生成唯一的文件名
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            # output_image_path = f"user/output/upscale_image/{user_id}/{model_id}/super_{model_id}_{timestamp_str}.png"
            output_image_path = os.path.join(current_app.config['OUTPUT_FOLDER'], "upscale_image", user_id, model_id,
                                             f"super_{model_id}_{timestamp_str}.png")

            # 如果输出目录不存在则创建
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

            # 保存输出图像
            images.save(output_image_path)
            print(f"Processed image saved at {output_image_path}")


        except Exception as e:
            # 捕获并记录异常
            error_message = f"An error occurred while processing the image with model {model_id}: {e}"
            print(error_message)


    def upscale_image(self,user_id, model_path, image):
        try:
            # 加载.pth模型文件
            model = torch.load(model_path)
            # model.eval()

            # 打开图像
            # image = Image.open(image_path)

            # 放大算法
            image = image.resize((1024, 1536))  # 根据需要调整大小

            print("输入照片进行修复")

            # 将图像转换为PyTorch张量
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            # 使用模型进行图像放大处理
            with torch.no_grad():
                output_image = model(image)

            # 将输出图像从PyTorch张量转换为PIL图像
            output_image = output_image.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
            output_image = Image.fromarray(output_image)

            # 生成唯一的文件名
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            output_image_path = os.path.join("user/output/upscale_image", user_id,
                                             f"super_{user_id}_{timestamp_str}.png")

            # 如果输出目录不存在则创建
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

            # 保存输出图像
            output_image.save(output_image_path)
            print(f"Processed image saved at {output_image_path}")

            return image

        except Exception as e:
            # 捕获并记录异常
            error_message = f"An error occurred while processing the image: {e}"
            print(error_message)


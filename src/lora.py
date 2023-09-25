import json
import subprocess
import os

from functools import partial
import torch
from asdff import AdPipeline, yolo_detector
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
import cv2

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from src.face_fusion import image_face_fusion
from util.virtualenv_util import activate_virtual_environment

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置使用的 GPU 设备编号


class LoraPipeline:
    def __init__(self, user_id: str,len_num: int):
        self.user_id = user_id
        self.base_path = "/root/autodl-tmp"
        self.lora_script_path = os.path.join(self.base_path, "lora-scripts-main")
        if len_num>1:
            self.diffusers_model_path = os.path.join(self.base_path, "diffusers_model","people")
        else:
            self.diffusers_model_path = os.path.join(self.base_path, "diffusers_model","single")

        self.lora_model_path = os.path.join(self.base_path, "lora_models")
        self.lora_output_path = os.path.join(self.base_path, "lora_models")


    def train_lora(self, para_num: int):
        # 开始训练lora
        # 切换目录到指定路径
        os.chdir(self.lora_script_path)

        # 获取当前目录地址并输出
        curr_dir = os.getcwd()
        print(f"当前目录地址: {curr_dir}")

        if not os.path.exists(self.lora_model_path):
            os.makedirs(self.lora_model_path)

        train_data_dir = os.path.join(
            self.base_path, "user", "detected", self.user_id
        )

        if not os.path.exists(train_data_dir):
            print("error: user data does not exist")

        for num in range(para_num):
            # 构建新的 train_data_dir
            output_path = os.path.join(
                self.lora_model_path, self.user_id
            )
            modified_train_data_dir = os.path.join(
                train_data_dir, str(num + 1)
            )
            print("预处理后用户数据数据集路径：", modified_train_data_dir)

            # 改为流输出
            # 定义命令和参数
            command = [
                "bash",
                os.path.join(self.lora_script_path, "train.sh"),
                modified_train_data_dir,
                f"{self.user_id}_{num}",
                output_path,
            ]

            # 启动子进程并捕获标准输出
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # 逐行读取标准输出并输出到当前进程的标准输出
            for line in process.stdout:
                print(line, end='')

            # 等待子进程完成
            process.wait()

            # 检查子进程的返回代码
            if process.returncode == 0:
                print("lora训练已完成！")
            else:
                print("lora训练失败！")

            # result = subprocess.run(
            #     [
            #         "bash",
            #         os.path.join(self.lora_script_path, "train.sh"),
            #         modified_train_data_dir,
            #         f"{self.user_id}_{num}",
            #         output_path,
            #     ],
            #     capture_output=True,
            #     text=True,
            # )
            # self.lora_output_path=os.path.join(
            #     output_path,f"{self.user_id}_{num}.safetensors"
            # )
            #

            # print(result)
            # # 输出结果
            # print("标准输出：", result.stdout)
            # print("lora訓練已完成！")
            # print("运行完后释放显存")
            # torch.cuda.empty_cache()

    # def generate_ad_output(self):
    #     # 第一层循环：遍历 self.diffusers_model_path 下的文件夹
    #     for folder in os.listdir(self.diffusers_model_path):
    #         if folder != ".ipynb_checkpoints":
    #
    #             first_level_path = os.path.join(self.diffusers_model_path, folder)
    #             # print(folder)
    #
    #             # 判断是否为文件夹
    #             if os.path.isdir(first_level_path):
    #                 # print(first_level_path)
    #
    #                 # 第二层循环：遍历子文件夹
    #                 for sub_folder in os.listdir(first_level_path):
    #                     if sub_folder != ".ipynb_checkpoints":
    #
    #                         model_dir = os.path.join(first_level_path, sub_folder)
    #                         print(model_dir)
    #
    #                         if os.path.isdir(model_dir):  # 确保是文件夹
    #                             pipe = AdPipeline.from_pretrained(
    #                                 model_dir, torch_dtype=torch.float16
    #                             )
    #                             print("成功建立ad_pipe")
    #                             print(self.lora_output_path)
    #
    #                             # pipe.load_lora_weights(self.lora_output_path)
    #                             # print(self.lora_output_path)
    #
    #                             pipe.safety_checker = None
    #
    #                             lora_w = 1.5
    #                             pipe._lora_scale = lora_w
    #                             print("lora接入成功！")
    #
    #                             # state_dict, network_alphas = pipe.lora_state_dict(
    #                             #     self.lora_output_path
    #                             # )
    #                             #
    #                             # for key in network_alphas:
    #                             #     network_alphas[key] = network_alphas[key] * lora_w
    #                             #
    #                             # pipe.load_lora_into_unet(
    #                             #     state_dict=state_dict,
    #                             #     network_alphas=network_alphas,
    #                             #     unet=pipe.unet,
    #                             # )
    #                             # pipe.load_lora_into_text_encoder(
    #                             #     state_dict=state_dict,
    #                             #     network_alphas=network_alphas,
    #                             #     text_encoder=pipe.text_encoder,
    #                             # )
    #
    #                             pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #                                 pipe.scheduler.config
    #                             )
    #                             pipe = pipe.to("cuda")
    #                             print("使用cuda")
    #
    #                             person_model_path = os.path.join(
    #                                 self.base_path, "face_yolov8n.pt"
    #                             )
    #                             person_detector = partial(
    #                                 yolo_detector, model_path=person_model_path
    #                             )
    #
    #                             with open(
    #                                 os.path.join(model_dir, "config.json"),
    #                                 "r",
    #                                 encoding="utf-8",
    #                             ) as f:
    #                                 common = json.load(f)
    #                             print("成功读取config文件")
    #                             print(common)
    #
    #                             model_id = common["model_id"]
    #                             common = common["config"]
    #                             new_commmon = {
    #                                 "prompt": common["prompt"],
    #                                 "negative_prompt":common["negative_prompt"]
    #                             }
    #                             generate_basic_image_path = os.path.join(
    #                                 self.base_path,
    #                                 "user",
    #                                 "output",
    #                                 "upscale_image",
    #                                 self.user_id,
    #                                 model_id,
    #                             )
    #
    #                             for file in os.listdir(generate_basic_image_path):
    #                                 print("循环修复文件夹中人脸")
    #                                 image_path = os.path.join(generate_basic_image_path, file)
    #                                 images = load_image(image_path)
    #                                 result = pipe(
    #                                     common = new_commmon,
    #                                     images=[images],
    #                                     detectors=[person_detector, pipe.default_detector],
    #                                 ).images
    #
    #                                 print("保存")
    #                                 ad_output_path = os.path.join(
    #                                     self.base_path,
    #                                     "user",
    #                                     "output",
    #                                     "ad_output",
    #                                     self.user_id,
    #                                     model_id
    #                                 )
    #                                 # 创建路径
    #                                 if not os.path.exists(ad_output_path):
    #                                     os.makedirs(ad_output_path, exist_ok=True)
    #                                     print(f"路径 {ad_output_path} 创建成功")
    #
    #                                 for img in result:
    #                                     img.save(os.path.join(ad_output_path, f"{self.user_id}_{model_id}.png"))
    #                                     # result.save(ad_output_path)
    #                                 print("保存至", ad_output_path)
    #                                 ff_output_path = os.path.join(
    #                                     self.base_path, "user", "output", "ff_output", self.user_id, model_id
    #                                 )
    #                                 # img.save(ad_output_path)
    #                                 # print()
    #                                 self.image_face_fusion(ad_image_path=os.path.join(ad_output_path, f"{self.user_id}_{model_id}.png"), output_path=ff_output_path,model_id=model_id)
    #                                 # image_face_fusion()
    #                             # for img in result.images:

    # 以上是备份

    def generate_ad_output(self):
        # 第一层循环：遍历 self.diffusers_model_path 下的文件夹
        for folder in os.listdir(self.diffusers_model_path):
            if folder != ".ipynb_checkpoints":

                first_level_path = os.path.join(self.diffusers_model_path, folder)
                # print(folder)

                # 判断是否为文件夹
                if os.path.isdir(first_level_path):
                    # print(first_level_path)

                    # 第二层循环：遍历子文件夹
                    for sub_folder in os.listdir(first_level_path):
                        if sub_folder != ".ipynb_checkpoints":
                            print(sub_folder)

                            model_dir = os.path.join(first_level_path, sub_folder)
                            print(model_dir)

                            if os.path.isdir(model_dir):  # 确保是文件夹
                                pipe = AdPipeline.from_pretrained(
                                    model_dir, torch_dtype=torch.float16
                                )
                                print("成功建立ad_pipe")
                                print(self.lora_output_path)

                                # 这玩意现在还是单人的，没有遍历lora文件。。
                                print(os.path.join(self.lora_output_path,self.user_id,f"{self.user_id}_0"))
                                # pipe.load_lora_weights(os.path.join(self.lora_output_path,self.user_id,f"{self.user_id}_0.safetensors"))

                                pipe.safety_checker = None

                                lora_w = 1.5
                                pipe._lora_scale = lora_w
                                print("lora接入成功！")

                                # state_dict, network_alphas = pipe.lora_state_dict(
                                #     self.lora_output_path
                                # )
                                #
                                # for key in network_alphas:
                                #     network_alphas[key] = network_alphas[key] * lora_w
                                #
                                # pipe.load_lora_into_unet(
                                #     state_dict=state_dict,
                                #     network_alphas=network_alphas,
                                #     unet=pipe.unet,
                                # )
                                # pipe.load_lora_into_text_encoder(
                                #     state_dict=state_dict,
                                #     network_alphas=network_alphas,
                                #     text_encoder=pipe.text_encoder,
                                # )

                                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                                    pipe.scheduler.config
                                )
                                pipe = pipe.to("cuda")
                                pipe.unet = pipe.unet.to("cuda")  # 将unet模型移动到CUDA设备

                                # generator = torch.Generator(device="cuda").manual_seed(2)

                                print("使用cuda")

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
                                print("成功读取config文件")
                                print(common)

                                model_id = common["model_id"]
                                common = common["config"]
                                new_commmon = {
                                    "prompt": common["prompt"],
                                    "negative_prompt": common["negative_prompt"]
                                }
                                generate_basic_image_path = os.path.join(
                                    self.base_path,
                                    "user",
                                    "output",
                                    "upscale_image",
                                    self.user_id,
                                    model_id,
                                )

                                for file in os.listdir(generate_basic_image_path):
                                    print("循环修复文件夹中人脸")
                                    image_path = os.path.join(generate_basic_image_path, file)
                                    images = load_image(image_path)
                                    result = pipe(
                                        common=new_commmon,
                                        images=[images],
                                        detectors=[person_detector, pipe.default_detector]
                                        # generator=generator
                                    ).images

                                    print("保存")
                                    ad_output_path = os.path.join(
                                        self.base_path,
                                        "user",
                                        "output",
                                        "ad_output",
                                        self.user_id,
                                        model_id
                                    )
                                    # 创建路径
                                    if not os.path.exists(ad_output_path):
                                        os.makedirs(ad_output_path, exist_ok=True)
                                        print(f"路径 {ad_output_path} 创建成功")

                                    for img in result:
                                        img.save(os.path.join(ad_output_path, f"{self.user_id}_{model_id}.png"))
                                        # result.save(ad_output_path)
                                    print("保存至", ad_output_path)
                                    ff_output_path = os.path.join(
                                        self.base_path, "user", "output", "ff_output", self.user_id, model_id
                                    )
                                    # img.save(ad_output_path)
                                    # print()
                                    self.image_face_fusion(
                                        ad_image_path=os.path.join(ad_output_path, f"{self.user_id}_{model_id}.png"),
                                        output_path=ff_output_path, model_id=model_id)
                                    # image_face_fusion()
                                # for img in result.images:

    def image_face_fusion(self, ad_image_path:str, output_path:str,model_id:str):

        # 这里假设'pipeline'和'Tasks'已经被正确导入
        # image_face_fusion_pipeline = pipeline(
        #     Tasks.image_face_fusion, model="damo/cv_unet-image-face-fusion_damo"
        # )

        # 创建路径
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            print(f"路径 {output_path} 创建成功")


        # 读取 ad 处理后的照片
        from util.image_util import get_first_image
        user_path = os.path.join(self.base_path, "user", "detected", self.user_id, "1", "10_face")
        user_image_path = get_first_image(user_path)

        #         # 调用工具函数来切换虚拟环境
        virtual_env_name = "modelscope"
        activate_virtual_environment(virtual_env_name,ad_image_path,user_image_path,os.path.join(output_path,f"{self.user_id}_{model_id}.png"))
        # image_face_fusion(ad_image_path,user_image_path,os.path.join(output_path,f"{self.user_id}_{model_id}.png"))
        #
        # result = image_face_fusion_pipeline(
        #     dict(template=ad_image_path, user=user_image_path)
        # )
        #
        # result_rgb = cv2.cvtColor(result[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
        # cv2.imwrite(os.path.join(output_path,f"{self.user_id}_{model_id}.png"), result_rgb)

        print("finished!")

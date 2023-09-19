from PIL import Image
import torch
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import warnings

def process_image():
    # 打开图像
    image = Image.open("pose.jpg").convert("RGB")

    # 初始化模型
    model = OpenposeDetector.from_pretrained("/root/autodl-tmp/controlnet")

    # 进行姿势检测
    pose = model(image)

    # 保存姿势检测结果
    pose.save("bone_pose.png")

    # 关闭 FutureWarning 警告
    warnings.filterwarnings("ignore", category=FutureWarning)

    controlnet = ControlNetModel.from_pretrained(
        "/root/autodl-tmp/controlnet/sd-controlnet-openpose", torch_dtype=torch.float16
    )

    model_id = "/root/autodl-tmp/diffusers_model/real_fashi"
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(2)

    prompt = "solo,realistic,1girl,smile,French hairstyle,Clear eyes,ultra realistic skin,exquisite facial features,goddess women, Off shoulder dress,Backlit, Contrast Filters, looking at viewer,"
    negative_prompt = ["Two people,（deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, (worst quality, low quality:2), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, (tree,wood:1.2), (stone:1.2), (green,black,white:1.4), (sandals:1.4),"]

    output = pipe(
        [prompt],
        pose,
        negative_prompt=negative_prompt,
        generator=generator,
        num_inference_steps=20,
        width = 1024,
        height = 1536
    ).images[0]
    output.save("test.png")
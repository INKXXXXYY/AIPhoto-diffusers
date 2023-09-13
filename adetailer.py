from functools import partial
import torch
from asdff import AdPipeline, yolo_detector
from huggingface_hub import hf_hub_download
from diffusers import KDPM2DiscreteScheduler
from diffusers.utils import load_image
from PIL import Image

def generate_ad_output():
    # 引入lora（无控制）
    lora_path = "/root/autodl-tmp/lora-scripts-main/output/test.safetensors"

    pipe = AdPipeline.from_pretrained("/root/autodl-tmp/diffusers_model/real_fashi", torch_dtype=torch.float16)
    pipe.load_lora_weights(lora_path)

    pipe.safety_checker = None

    lora_w = 1
    pipe._lora_scale = lora_w

    state_dict, network_alphas = pipe.lora_state_dict(lora_path)

    for key in network_alphas:
        network_alphas[key] = network_alphas[key] * lora_w

    pipe.load_lora_into_unet(
        state_dict=state_dict,
        network_alphas=network_alphas,
        unet=pipe.unet
    )

    pipe.load_lora_into_text_encoder(
        state_dict=state_dict,
        network_alphas=network_alphas,
        text_encoder=pipe.text_encoder
    )

    pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    person_model_path = "/root/autodl-tmp/face_yolov8n.pt"
    person_detector = partial(yolo_detector, model_path=person_model_path)

    image_path = "super_test.png"


    common = {"prompt": "realistic,1girl, smile,French hairstyle,Clear eyes, ultra realistic skin, exquisite facial features,goddess women, Off shoulder dress,Backlit, Contrast Filters, looking at viewer,",     "negative_prompt":"Two people,（deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, (worst quality, low quality:2), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, (tree,wood:1.2), (stone:1.2), (green,black,white:1.4), (sandals:1.4),","num_inference_steps": 25}
    images = load_image(image_path)
    result = pipe(common=common, images=[images], detectors=[person_detector, pipe.default_detector])

    # 假设您的ADOutput对象是ad_output
    for img in result.images:
        img.save("final.png")

    return result
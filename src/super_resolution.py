import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

def upscale_image():

    device = "cuda"
    model_id_or_path = "/root/autodl-tmp/diffusers_model/real_fashi"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)


    # prompt = "A fantasy landscape, trending on artstation"
    prompt = "solo,realistic,1girl,smile,French hairstyle,Clear eyes,ultra realistic skin,exquisite facial features,goddess women, Off shoulder dress,Backlit, Contrast Filters, looking at viewer,"
    negative_prompt="""Two people,（deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, (worst quality, low quality:2), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, (tree,wood:1.2), (stone:1.2), (green,black,white:1.4), (sandals:1.4),"""

    image_path = "test.png"

    image = Image.open(image_path)

    print("输入照片进行修复")
    images = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, strength=0.3, guidance_scale=7.5).images[0]
    images.save("super_test.png")
    # images


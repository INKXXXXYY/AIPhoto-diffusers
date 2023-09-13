# import torch
# from diffusers import StableDiffusionPipeline
#
# text2img_pipe = StableDiffusionPipeline.from_pretrained(
#     "lightspaceai/AWPortrait_v12"
#     , torch_dtype = torch.float16
#     , safety_checker = None
# ).to("cuda:0")
#
# lora_path = "</root/autodl-tmp/stable-diffusion-webui/models/Lora/face1.safetensors>"
# text2img_pipe.load_lora_weights(lora_path)
#
# from diffusers import EulerDiscreteScheduler
#
# prompt = """
# Маша making extreme selfie on skyscraper, bird's eye view, from above, night, smiling
# """
# neg_prompt = """
# NSFW,deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation
# """
#
# text2img_pipe.scheduler = EulerDiscreteScheduler.from_config(text2img_pipe.scheduler.config)
#
# image = text2img_pipe(
#     prompt = prompt
#     , negative_prompt = neg_prompt
#     , generator = torch.Generator("cuda:0").manual_seed(3135098381)
#     , num_inference_steps = 28
#     , guidance_scale = 8
#     , width = 512
#     , height = 768
# ).images[0]
# # display(image)


from diffusers import StableDiffusionPipeline

image_pipe = StableDiffusionPipeline.from_pretrained("lightspaceai/AWPortrait_v12")
# 加载本地模型：
# image_pipe = StableDiffusionPipeline.from_pretrained("./models/Stablediffusion/stable-diffusion-v1-4")
image_pipe.to("cuda")

prompt = "a photograph of an astronaut riding a horse"
pipe_out = image_pipe(prompt)

image = pipe_out.images[0]
# you can save the image with
# image.save(f"astronaut_rides_horse.png")
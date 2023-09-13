import torch
from diffusers import StableDiffusionPipeline

def generate_image():


    text2img_pipe = StableDiffusionPipeline.from_pretrained("/root/autodl-tmp/diffusers_model/real_fashi").to("cuda:1")


    # text2img_pipe.unload_lora_weights()
    # lora_path = "/root/autodl-tmp/stable-diffusion-webui/models/Lora/face1.safetensors"

    # lora_w = 0.8
    # text2img_pipe._lora_scale = lora_w

    # state_dict, network_alphas = text2img_pipe.lora_state_dict(
    #     lora_path
    # )

    # for key in network_alphas:
    #     network_alphas[key] = network_alphas[key] * lora_w

    # #network_alpha = network_alpha * lora_w
    # text2img_pipe.load_lora_into_unet(
    #     state_dict = state_dict
    #     , network_alphas = network_alphas
    #     , unet = text2img_pipe.unet
    # )

    # text2img_pipe.load_lora_into_text_encoder(
    #     state_dict = state_dict
    #     , network_alphas = network_alphas
    #     , text_encoder = text2img_pipe.text_encoder
    # )

    from diffusers import DPMSolverSDEScheduler

    prompt = "solo,realistic,1girl,smile,French hairstyle,Clear eyes,ultra realistic skin,exquisite facial features,goddess women, Off shoulder dress,Backlit, Contrast Filters, looking at viewer,"
    negative_prompt="""Two people,（deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, (worst quality, low quality:2), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, (tree,wood:1.2), (stone:1.2), (green,black,white:1.4), (sandals:1.4),"""

    text2img_pipe.scheduler = DPMSolverSDEScheduler.from_config(text2img_pipe.scheduler.config)

    # 把推理分發到不同是gpu上
    # 待測試
    # distributed_state = text2img_pipe()
    # text2img_pipe.to(distributed_state.device)

    image = text2img_pipe(
        prompt = prompt
        , negative_prompt = negative_prompt
        , generator = torch.Generator("cuda:1")
        , num_inference_steps = 28
        , guidance_scale = 7
        , width = 512
        , height = 512
    ).images[0]

    image.save('test.png')
    # image
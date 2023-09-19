import torch

from ModelManager import ModelManager

manager = ModelManager()

# 添加模型
manager.add_model("pose_model", "/root/autodl-tmp/controlnet")
manager.add_model("diffuser_model1", "/root/autodl-tmp/diffusers_model/real_fashi")
manager.add_model("diffuser_model2", "/root/autodl-tmp/diffusers_model/other_model")

# 修改模型参数
manager.modify_model_parameters("diffuser_model1", torch_dtype=torch.float32)

# 处理图像并调用所有模型
output_images = manager.process_image_with_models(
    "pose.jpg",
    "pose_model",
    ["diffuser_model1", "diffuser_model2"],
    "solo,realistic,1girl,smile,French hairstyle,Clear eyes,ultra realistic skin,exquisite facial features,goddess women, Off shoulder dress,Backlit, Contrast Filters, looking at viewer,",
    ["Two people,（deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, (worst quality, low quality:2), jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, (tree,wood:1.2), (stone:1.2), (green,black,white:1.4), (sandals:1.4),"],
    generator_seed=2,
    num_inference_steps=20,
    width=1024,
    height=1536
)

# 保存输出图像
for i, output_image in enumerate(output_images):
    output_image.save(f"output_{i}.png")

from src.ModelManager import ModelManager

def infer(user_id):

    manager = ModelManager()

    # 添加模型
    manager.add_model("pose_model", "/root/autodl-tmp/controlnet/sd-controlnet-openpose")

    manager.add_model("diffuser_model1", "/root/autodl-tmp/diffusers_model/newchinese/newchinese1_3")
    manager.add_model("diffuser_model2", "/root/autodl-tmp/diffusers_model/newchinese/newchinese2_1")
    manager.add_model("diffuser_model3", "/root/autodl-tmp/diffusers_model/newchinese/newchinese3_1")

    # 处理图像并调用所有模型
    manager.process_image_with_models(
        user_id,
        "/root/autodl-tmp/diffusers_model/config.json",
        ["diffuser_model1", "diffuser_model2", "diffuser_model3"],
        num_inference_steps=30,
        width=512,
        height=768
    )

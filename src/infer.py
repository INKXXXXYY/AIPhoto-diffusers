from src.ModelManager import ModelManager

def infer(user_id):

    manager = ModelManager()

    # 添加模型
    manager.add_model("pose_model", "/root/autodl-tmp/controlnet/sd-controlnet-openpose")

    manager.add_model("newchinese1_3", "/root/autodl-tmp/diffusers_model/newchinese/newchinese1_3")
    manager.add_model("newchinese2_1", "/root/autodl-tmp/diffusers_model/newchinese/newchinese2_1")
    manager.add_model("newchinese3_1", "/root/autodl-tmp/diffusers_model/newchinese/newchinese3_1")

    # 处理图像并调用所有模型
    manager.process_image_with_models(
        user_id,
        ["newchinese1_3", "newchinese2_1", "newchinese3_1"],
        num_inference_steps=30,
        width=512,
        height=768
    )

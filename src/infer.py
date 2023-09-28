import os

from src.ModelManager import ModelManager

def infer(user_id,len_num):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if len_num>1:
        manager = ModelManager()

        # 添加模型
        manager.add_model("pose_model", "/root/autodl-tmp/controlnet/sd-controlnet-openpose")

        manager.add_model("newchinese1_3", "/root/autodl-tmp/diffusers_model/people/newchinese/newchinese1_3")
        manager.add_model("newchinese2_1", "/root/autodl-tmp/diffusers_model/people/newchinese/newchinese2_1")
        manager.add_model("newchinese3_1", "/root/autodl-tmp/diffusers_model/people/newchinese/newchinese3_1")

        # 处理图像并调用所有模型
        manager.process_image_with_models(
            user_id,
            ["newchinese1_3", "newchinese2_1", "newchinese3_1"],
            num_inference_steps=50,
            width=512,
            height=768
        )
    else:
        manager = ModelManager()

        # 添加模型
        manager.add_model("pose_model", "/root/autodl-tmp/controlnet/control_v11p_sd15_openpose")
        manager.add_model("canny_model", "/root/autodl-tmp/controlnet/control_v11p_sd15_canny")
        # manager.add_model("pose_model", "/root/autodl-tmp/controlnet/sd-controlnet-openpose")

        # manager.add_model("fashi", "/root/autodl-tmp/diffusers_model/single/fashi/fashi_1")
        manager.add_model("1person_100", "/root/autodl-tmp/diffusers_model/single/1person/1person")
        manager.add_model("1person_2_100", "/root/autodl-tmp/diffusers_model/single/1person/1person_2")
        manager.add_model("1person_4_100", "/root/autodl-tmp/diffusers_model/single/1person/1person_4")

        # manager.add_model("fashi", "/root/autodl-tmp/diffusers_model/single/fashi/fashi_1")

        # manager.add_model("newchinese2_1", "/root/autodl-tmp/diffusers_model/newchinese/newchinese2_1")
        # manager.add_model("newchinese3_1", "/root/autodl-tmp/diffusers_model/newchinese/newchinese3_1")

        # 处理图像并调用所有模型
        manager.process_image_with_models(
            user_id,
            ["1person_100","1person_2_100","1person_4_100"],
            # ["fashi","1person_100","1person_2_100"],
            num_inference_steps=40,
            width=512,
            height=768
        )


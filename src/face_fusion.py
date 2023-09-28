import cv2
import sys

import numpy as np
from PIL.Image import Image
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def image_face_fusion(template_path, user_path, output_path='result.png'):
    # 创建图像融合的管道
    image_face_fusion_pipeline = pipeline(
        Tasks.image_face_fusion,
        model='damo/cv_unet-image-face-fusion_damo'
    )

    # print("输出两张照片看看")
    # try:
    #     image1 = cv2.imread(template_path)
    #     image2 = cv2.imread(user_path)
    #
    #     cv2.imwrite("root/autodl-tmp/1.png", image1)
    #     cv2.imwrite("root/autodl-tmp/2.png", image2)
    #
    #     image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite("root/autodl-tmp/2_rgb.png", image2_rgb)
    # except Exception as e:
    #     print("Error:", str(e))


    # 执行图像融合
    result = image_face_fusion_pipeline(dict(template=template_path, user=user_path))

    # 将结果保存到指定路径
    # 将图像从 BGR 顺序转换为 RGB 顺序
    # result_rgb = cv2.cvtColor(result[OutputKeys.OUTPUT_IMG])
    # result_rgb = cv2.cvtColor(result[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)

    # 将转换后的结果保存到指定路径
    cv2.imwrite(output_path, result[OutputKeys.OUTPUT_IMG])


if __name__ == '__main__':
    # 从命令行参数获取文件路径
    template_path = sys.argv[1]
    user_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'result.png'

    # 调用 image_face_fusion 函数
    image_face_fusion(template_path, user_path, output_path)
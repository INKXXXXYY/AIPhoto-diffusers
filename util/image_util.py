import os
import glob
import cv2


def get_first_image(path):
    image_files = glob.glob(os.path.join(path, '*.*'))  # 使用通配符 *.* 匹配任意格式的文件

    if len(image_files) > 0:
        for file_path in sorted(image_files):
            image = cv2.imread(file_path)
            if image is not None:  # 加载成功表示为照片文件
                return file_path

        print("指定路径下没有找到照片")
        return None

    else:
        print("指定路径下没有找到照片")
        return None

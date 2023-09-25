import os

from flask import request
from util.save_photo import save_photos_to_server


def get_all_file(user_id,len_num,files,gender,origin_path,pre_img_path):
    # 创建字典来保存文件数据
    file_dict = {}

    # 提取文件数据，并将不同参数的文件分别存储到字典中
    # 保存照片到服务器并预处理
    # print("-----------------------------------------------------------------")
    # print("將照片保存到服務器")
    # isfemale = files.
    # print()

    for i in range(1, len_num+1):
        parameter_name = f'photo_{i}'
        parameter_famele = f'isfemale{i}'

        file_list = files.getlist(parameter_name)
        isfemale = gender.get(parameter_famele)
        print(isfemale)
        file_dict[parameter_name] = file_list

        save_photos_to_server(user_id,i,file_list,parameter_famele, origin_path, pre_img_path)


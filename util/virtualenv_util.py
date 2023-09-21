import os
import subprocess


def activate_virtual_environment(venv_name,tem_image,user_img,output_path):
    # 构建激活虚拟环境的命令
    # activate_cmd = f"conda activate {venv_name}  && "
    # 切换到另一个虚拟环境（假设虚拟环境路径为 /path/to/new_environment）
    env = {'PATH': '/root/autodl-tmp/miniconda3/envs/modelscope/bin:' + os.environ['PATH']}

    # 调用 Python 文件（假设文件路径为 path/to/file.py）
    python_command = f"python /root/autodl-tmp/aifamily_demo/src/face_fusion.py '{tem_image}' '{user_img}' '{output_path}'"

    # 在子进程中执行命令
    subprocess.run(python_command, env=env, shell=True)


    # try:
    #     # 尝试切换虚拟环境
    #     # os.system(activate_cmd)
    #     subprocess.run(activate_cmd, shell=True)
    #
    #     print(f"成功切换至虚拟环境 {venv_name}")
    #     # os.system("python face_fusion.py '/root/autodl-tmp/lora-scripts-main/train/face/10_face/processed_20230726-185747.jpeg' “/root/autodl-tmp/aifamily_demo/final.png")
    # except Exception as e:
    #     print(f"切换至虚拟环境 {venv_name} 失败: {e}")


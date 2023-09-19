import subprocess
import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import fileinput
import sys


# os.environ["MKL_THREADING_LAYER"] = "GNU"


def train_lora(train_data_dir,output_name):
    # 开始训练lora
    lora_script_path = "/root/autodl-tmp/lora-scripts-main"

    # 切换目录到指定路径
    os.chdir(lora_script_path)

    # 获取当前目录地址并输出
    curr_dir = os.getcwd()
    print(f"当前目录地址: {curr_dir}")

    # 设置要修改的参数和新值
    params = {
        b'train_data_dir': train_data_dir,
        b'output_name': output_name
    }

    # 读取文件内容并替换参数
    with fileinput.FileInput('train.sh', inplace=True,backup='', mode='rb') as file:
        for line in file:
            for param, value in params.items():
                if line.startswith(param):
                    print(f'{param}={value}', file=sys.stdout)
                    break
            else:
                print(line, end='')

    # 执行 bash train.sh 命令
    result = subprocess.run(["bash", "train.sh"], capture_output=True, text=True)
    # result = subprocess.run(["bash", "train.sh"])
    print(result)

    # 输出结果
    print("标准输出：", result.stdout)
    print("错误输出：", result.stderr)
    print("返回码：", result.returncode)


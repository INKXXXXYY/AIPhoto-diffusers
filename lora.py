import subprocess
import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用的 GPU 设备编号

from util.replace_para import replace_params

# os.environ["MKL_THREADING_LAYER"] = "GNU"


def train_lora(train_data_dir,para_num,output_path):
    # 开始训练lora
    lora_script_path = "/root/autodl-tmp/lora-scripts-main"

    # 切换目录到指定路径
    os.chdir(lora_script_path)

    # 获取当前目录地址并输出
    curr_dir = os.getcwd()
    print(f"当前目录地址: {curr_dir}")

    for num in range(para_num):
        # 构建新的 train_data_dir 值
        modified_train_data_dir = os.path.join(train_data_dir, str(num+1))
        print(modified_train_data_dir)

        # params = {
        #     'train_data_dir': f'{modified_train_data_dir}',
        #     # 'output_dir':output_path,
        #     'output_name': f'{str(num)}'
        # }
        #
        # # 替换 train.sh 文件中的参数
        # replace_params('/root/autodl-tmp/lora-scripts-main/train.sh', params)

        # 执行 bash train.sh 命令
        result = subprocess.run(["bash", "/root/autodl-tmp/lora-scripts-main/train.sh",modified_train_data_dir,str(num),output_path], capture_output=True, text=True)
        print(result)

        # 输出结果
        print("标准输出：", result.stdout)
        # print("错误输出：", result.stderr)
        # print("返回码：", result.returncode)
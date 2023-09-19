import fileinput
import sys

def replace_params(file_path, params):
    with fileinput.FileInput(file_path, inplace=True, backup='.bak') as file:
        for line in file:
            for param, value in params.items():
                # value="+value+"
                # print(value)
                if line.startswith(param):
                    line = f'{param}={value}\n'
                    break
            sys.stdout.write(line)

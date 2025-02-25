import json
from copy import deepcopy
import re
import os
 
current_path = os.path.dirname(os.path.realpath(__file__))

print("current_path: ", current_path)

videos_source_file = "/data/cache/live_add_AI_bak_ttted/live"


def file_path_exists(file_path):
    if os.path.exists(file_path):
        print(f"{file_path} 存在！")
    else:
        print(f"{file_path} 不存在！")
        b = os.mkdir(file_path)
        print(f"创建文件夹{file_path} b:{b}")

def traverse_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            abs_file_path = os.path.join(root, file_name)
            # print(abs_file_path)
            file_list.append(abs_file_path)
    
    return file_list


if __name__ == '__main__':
    files_list = traverse_folder(videos_source_file)
    files_list_size = len(files_list)
    print("files_list size: ", files_list_size)
    print("files_list [0:3]: ", files_list[0:3])

    # 打开文件以写入模式
    with open('output.txt', 'w', encoding='utf-8') as file:
        # 遍历列表
        for line in files_list:
            # 写入当前字符串并添加换行符
            file.write(line.replace(f"{videos_source_file}/", "").replace("_AI", "") + '\n')

    print("文件写入完成。")


        
        
    
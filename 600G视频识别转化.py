import json
from copy import deepcopy
import re
import os
 
current_path = os.path.dirname(os.path.realpath(__file__))

print("current_path: ", current_path)

videos_source_file = "/data/cache/live"

out_dir = "/data/cache/live_add_AI"

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

def get_out_file_name(video_path):
    if not video_path.endswith(".mp4"):
        print(f"{video_path} not end with mp4")
        return ""
    
    tmp = video_path.replace(f"{videos_source_file}/", "").split("/")

    print("tmp: ", tmp)

    # check
    if tmp[2].split(".")[1] != "mp4":
        print("get error path: ", video_path)
        return ""

    hash_code = tmp[0]
    date_str = tmp[1]
    file_name = tmp[2].split(".")[0]

    hash_path = os.path.join(out_dir, hash_code)
    file_path_exists(hash_path)

    date_path = os.path.join(hash_path, date_str)
    file_path_exists(date_path)

    return f"{date_path}/{file_name}_AI.mp4"


if __name__ == '__main__':
    file_path_exists(out_dir)


    files_list = traverse_folder(videos_source_file)
    files_list_size = len(files_list)
    print("files_list size: ", files_list_size)
    print("files_list [0:3]: ", files_list[0:3])

    for video_path in files_list:
        out_file_name = get_out_file_name(video_path)
        if out_file_name == "":
            continue
        
        print("out_file_name: ", out_file_name)


        exit()

        
        
    
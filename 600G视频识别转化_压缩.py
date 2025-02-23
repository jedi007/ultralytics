import json
from copy import deepcopy
import re
import os
 
current_path = os.path.dirname(os.path.realpath(__file__))

print("current_path: ", current_path)

videos_source_file = "/data/cache/live_add_AI"

out_dir = "/data/cache/live_add_AI_ffmpeg"

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

    return f"{date_path}/{file_name}.mp4"


import subprocess

def compress_video_bitrate(input_file, output_file, bitrate='1000k'):
    try:
        # 构建 FFmpeg 命令
        command = [
            'ffmpeg',
            '-i', input_file,
            '-b:v', bitrate,  # 设置视频比特率
            output_file
        ]
        # 执行 FFmpeg 命令
        subprocess.run(command, check=True)
        print(f"视频压缩成功，输出文件为 {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"视频压缩失败: {e}")
    except FileNotFoundError:
        print("未找到 FFmpeg 可执行文件，请确保 FFmpeg 已正确安装并添加到系统路径中。")

import time

if __name__ == '__main__':
    file_path_exists(out_dir)


    files_list = traverse_folder(videos_source_file)
    files_list_size = len(files_list)
    print("files_list size: ", files_list_size)
    print("files_list [0:3]: ", files_list[0:3])

    count = 0
    for video_path in files_list:
        try:
            # 获取文件大小
            file_size = os.path.getsize(video_path)
            file_size_MB = file_size/1024/1024
            print(f"文件 {video_path} 的大小是 {file_size_MB} MB")

            if file_size_MB < 0.1:
                continue
        except FileNotFoundError:
            print(f"文件 {video_path} 未找到")
        

        out_file_name = get_out_file_name(video_path)
        if out_file_name == "":
            continue

        if os.path.exists(out_file_name):
            print(f"outfile {out_file_name} exists")
            continue
        
        print(f"begin one file:{out_file_name}")
        time.sleep(0.5)
        compress_video_bitrate(video_path, out_file_name)
        print(f"finish one file:{out_file_name}")

        # 获取文件大小
        out_file_size = os.path.getsize(out_file_name)
        print(f"compress:{out_file_size/file_size}")
        
        count += 1
        print(f"进度: {count}/{files_list_size}")


        
        
    
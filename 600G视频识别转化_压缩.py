import json
from copy import deepcopy
import re
import os

# # 转码到临时文件
# if ffmpeg -hide_banner -loglevel error -nostdin -i "$file" \
#     -c:v libx264 -crf 23 -preset medium \
#     -c:a aac -b:a 128k \
#     -movflags +faststart \
#     -b:v 1000k \
#     -y "$temp_file"; then

# 全局参数
# -hide_banner：
# 作用：隐藏 FFmpeg 启动时显示的版权信息、版本信息等横幅内容。这样可以使命令的输出更加简洁，专注于关键信息。
# -loglevel error：
# 作用：设置日志级别为 error。这意味着 FFmpeg 只会输出错误信息，忽略其他级别的日志（如警告、调试信息等），有助于减少不必要的输出，让你更清晰地看到可能出现的问题。
# -nostdin：
# 作用：禁止 FFmpeg 从标准输入（stdin）读取数据。这可以防止在命令执行过程中意外从标准输入获取数据，避免一些潜在的干扰。
# 输入参数
# -i "$file"：
# 作用：指定输入文件。-i 是输入文件的标志，"$file" 是一个变量，代表实际的输入视频文件的路径和名称。
# 视频编码参数
# -c:v libx264：
# 作用：指定视频编码器为 libx264。libx264 是一个广泛使用的开源 H.264 视频编码器，它可以提供高质量的视频压缩效果，并且在大多数设备和播放器上都有良好的兼容性。
# -crf 23：
# 作用：设置恒定速率因子（Constant Rate Factor，CRF）。CRF 是一种用于控制视频质量的参数，取值范围通常是 0 - 51，数值越小表示视频质量越高，文件大小也越大；数值越大则视频质量越低，文件大小越小。23 是一个比较常用的默认值，在质量和文件大小之间提供了一个较好的平衡。
# -preset medium：
# 作用：指定编码速度和压缩比之间的平衡预设。preset 选项有多个取值，如 ultrafast、superfast、veryfast、faster、fast、medium、slow、slower、veryslow 等。medium 是一个折中的预设，在编码速度和压缩比之间取得了较好的平衡，编码速度不会太慢，同时也能获得相对较高的压缩比。
# -b:v 1000k：
# 作用：设置视频的比特率为 1000kbps（千比特每秒）。比特率决定了视频数据的传输速率，比特率越高，视频质量通常越好，但文件大小也会相应增加。
# 音频编码参数
# -c:a aac：
# 作用：指定音频编码器为 AAC（Advanced Audio Coding）。AAC 是一种广泛使用的音频编码格式，具有较高的音频质量和较小的文件大小，在大多数设备和播放器上都有良好的兼容性。
# -b:a 128k：
# 作用：设置音频的比特率为 128kbps。比特率越高，音频质量越好，但文件大小也会增加。128kbps 是一个常见的音频比特率设置，在大多数情况下可以提供较好的音频质量。
# 输出文件参数
# -movflags +faststart：
# 作用：将视频的元数据（如文件头信息）移动到文件开头。这样可以让视频在网络上更快地开始播放，因为播放器可以在下载了文件开头的元数据后就开始播放视频，而不需要等待整个文件下载完成。
# -y：
# 作用：自动覆盖输出文件。如果输出文件已经存在，使用 -y 选项可以避免 FFmpeg 询问是否覆盖该文件，直接进行覆盖操作。
# "$temp_file"：
# 作用：指定输出文件的路径和名称。"$temp_file" 是一个变量，代表实际的输出视频文件的路径和名称。


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


        
        
    
import os
from os import listdir, getcwd
from os.path import join
from shutil import copyfile
import cv2
import shutil
import ast
import copy

from PIL import Image
from ultralytics import YOLO
import time
import random
 
def traverse_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            abs_file_path = os.path.join(root, file_name)
            # print(abs_file_path)
            file_list.append(abs_file_path)
    
    return file_list

def traverse_folder_filename(folder_path):
    filename_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            filename_list.append(file_name)
    
    return filename_list

def file_path_exists(file_path):
    if os.path.exists(file_path):
        print(f"{file_path} 存在！")
    else:
        print(f"{file_path} 不存在！")
        b = os.mkdir(file_path)
        print(f"创建文件夹{file_path} b:{b}")

def get_box_from_name(image_name:str, width, height):
    words = image_name.split("_")
    # print("words: ", words)

    if len(words) != 5 :
        print(f"image:{image_name} name format error!")
        return "", (0 , 0, 0, 0)
    
    def get_xy(word:str):
        xy = word.split("&")
        if len(xy) != 2:
            print(f"word:{word} format error!")
            return -1,-1
    
        try:
            x = ast.literal_eval(xy[0])
            y = ast.literal_eval(xy[1])
        except (ValueError, SyntaxError):
            print("无法将字符串转换为浮点数")
            return -1,-1
        
        return x,y
    
    x1,y1 = get_xy(words[1])
    x2,y2 = get_xy(words[2])
    x3,y3 = get_xy(words[3])
    x4,y4 = get_xy(words[4])
    
    xarray = [x1,x2,x3,x4]
    yarray = [y1,y2,y3,y4]

    min_x = min(xarray)
    max_x = max(xarray)
    min_y = min(yarray)
    max_y = max(yarray)

    if min_x == -1 or min_y == -1:
        return "", (0 , 0, 0, 0)

    box_cx = (min_x + max_x)/2
    box_cy = (min_y + max_y)/2
    box_w = max_x - min_x
    box_h = max_y - min_y

    cx = box_cx/width
    cy = box_cy/height
    w = box_w/width
    h = box_h/height

    # print("cx, cy, w, h: ", cx, cy, w, h)

    return f"{cx} {cy} {w} {h}" , (min_x, min_y, max_x, max_y)


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"文件 {filename} 删除成功！")
    else:
        print(f"文件 {filename} 不存在。")


if __name__ == "__main__":
    source_dir = R'''/home/hyzh/DATA/car_plate/2/images'''

    file_names = traverse_folder_filename(source_dir)

    print("file_names: ", file_names[0:3])

    error_count = 0
    count = 0
    for file_name in file_names:
        count += 1
        image_path = os.path.join(source_dir, file_name)

        img = cv2.imread(image_path)
        width = img.shape[1]
        height = img.shape[0]

        box_str, box_xyxy = get_box_from_name(file_name[0:-4], width, height)

        if box_str == "":
            error_count += 1
            print(f"find {error_count} error name: {file_name}")
            delete_file(image_path)


        if count % 100 == 0:
            print(f"{count} complete")
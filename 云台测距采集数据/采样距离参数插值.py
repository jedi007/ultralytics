import re
import os

def load_data_from_file(file_path):
    """从文件中读取数据并解析
    每行格式: zoom_step:X ==> abs_zoom:Y ==> Z
    返回: [(abs_zoom, pix), ...] 列表
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 使用正则表达式解析
            match = re.search(r'abs_zoom:(\d+)\s*==>\s*([\d.]+)', line)
            if match:
                abs_zoom = int(match.group(1))
                pix = float(match.group(2))
                data.append((abs_zoom, pix))
    
    return data

def interpolate_data(data, step=10):
    """对数据进行插值，使abs_zoom形成等差数列，步长为step
    插值点的pix值按照线性插值计算
    """
    if not data:
        return []
    
    # 按abs_zoom排序
    data.sort(key=lambda x: x[0])
    
    result = []
    
    for i in range(len(data) - 1):
        abs_zoom1, pix1 = data[i]
        abs_zoom2, pix2 = data[i + 1]
        
        # 添加当前点
        result.append((abs_zoom1, pix1))
        
        # 计算需要插入的点数
        gap = abs_zoom2 - abs_zoom1
        num_insert = (gap // step) - 1
        
        if num_insert > 0:
            # 计算每个插入点的pix值增量
            increment = (pix2 - pix1) / (num_insert + 1)
            
            for j in range(1, num_insert + 1):
                new_abs_zoom = abs_zoom1 + j * step
                new_pix = pix1 + increment * j
                result.append((new_abs_zoom, round(new_pix, 4)))
    
    # 添加最后一个点
    result.append(data[-1])
    
    return result

def save_data_to_file(data, file_path):
    """将数据保存到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("abs_zoom\tpix\n")
        for abs_zoom, pix in data:
            f.write(f"{abs_zoom}\t{pix}\n")

if __name__ == "__main__":
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'zoom_3.995m_1-128.txt')
    
    data = load_data_from_file(input_file)
    print(f"从文件加载了 {len(data)} 条原始数据")
    
    # 插值
    interpolated_data = interpolate_data(data, step=10)
    print(f"插值后共有 {len(interpolated_data)} 条数据")
    
    # 显示部分结果
    print("\n前20条数据:")
    for i, (abs_zoom, pix) in enumerate(interpolated_data[:20]):
        print(f"  abs_zoom={abs_zoom}, pix={pix}")
    
    print("\n后10条数据:")
    for i, (abs_zoom, pix) in enumerate(interpolated_data[-10:]):
        print(f"  abs_zoom={abs_zoom}, pix={pix}")
    
    # 保存到文件
    output_file = os.path.join(script_dir, 'zoom_3.995m_interpolated.txt')
    save_data_to_file(interpolated_data, output_file)
    print(f"\n数据已保存到: {output_file}")

import os
import json
from collections import Counter
from glob import glob

def analyze_jsonl_files(input_pattern, keys_to_analyze=['tag', 'difficulty2filter']):
    # 用于存储所有文件的统计信息
    total_stats = {key: Counter() for key in keys_to_analyze}
    file_stats = {}

    # 遍历所有匹配的文件
    for filepath in glob(input_pattern):
        file_counters = {key: Counter() for key in keys_to_analyze}
        
        # 读取并统计单个文件
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                for key in keys_to_analyze:
                    value = data.get(key)
                    if value is not None:
                        file_counters[key][value] += 1
                        total_stats[key][value] += 1
        
        # 保存单个文件的统计结果
        file_stats[os.path.basename(filepath)] = file_counters
        
        # 打印单个文件的统计信息
        print(f"\n文件 {os.path.basename(filepath)} 的统计信息：")
        for key in keys_to_analyze:
            print(f"{key} 分布：")
            for value, count in sorted(file_counters[key].items()):
                print(f"  {value}: {count}")

    # 打印所有文件的总统计信息
    print("\n所有文件的总统计信息：")
    for key in keys_to_analyze:
        print(f"{key} 总体分布：")
        for value, count in sorted(total_stats[key].items()):
            print(f"  {value}: {count}")

    return total_stats, file_stats

def filter_and_save_jsonl(input_pattern, output_dir, filter_conditions):
    """
    过滤JSONL文件并保存符合条件的数据
    
    参数:
    input_pattern: 输入文件的glob模式
    output_dir: 输出文件夹路径
    filter_conditions: 过滤条件字典，如 {'tag': ['easy', 'medium'], 'difficulty2filter': ['简单']}
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有匹配的文件
    for filepath in glob(input_pattern):
        filtered_data = []
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, f"filtered_{filename}")
        
        # 读取并过滤数据
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                # 检查是否符合所有过滤条件
                match = True
                for key, allowed_values in filter_conditions.items():
                    if key in data:
                        if data[key] not in allowed_values:
                            match = False
                            break
                    else:
                        match = False
                        break
                
                if match:
                    filtered_data.append(data)
        
        # 保存过滤后的数据
        if filtered_data:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for item in filtered_data:
                    json.dump(item, outfile, ensure_ascii=False)
                    outfile.write('\n')
            
            print(f"已保存过滤后的数据到：{output_path}")
            print(f"过滤后的数据条数：{len(filtered_data)}")

# 使用示例
input_pattern = '/map-vepfs/xinrun/KOR-Bench/results/gpqa/filter/Qwen2.5-72B-Instruct_dk-data-toppr-struct-physics-chem-bio-2nd-filter_gpqa-filter.jsonl'
total_stats, file_stats = analyze_jsonl_files(input_pattern, ['quality', 'difficulty'])

output_dir = 'data'
filter_conditions = {
    'quality': [8, 9, 10],
    'difficulty': [8, 9, 10]
}

filter_and_save_jsonl(input_pattern, output_dir, filter_conditions)
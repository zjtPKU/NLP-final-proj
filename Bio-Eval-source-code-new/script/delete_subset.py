import json

def load_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def delete_subset(full_set_path, subset_paths, output_path):
    # 读取全集数据
    full_data = load_jsonl_file(full_set_path)
    
    # 收集所有需要删除的 UUID
    uuids_to_delete = set()
    for subset_path in subset_paths:
        subset_data = load_jsonl_file(subset_path)
        uuids_to_delete.update(item['uuid'] for item in subset_data)
    
    # 过滤掉需要删除的数据
    filtered_data = [item for item in full_data if item['uuid'] not in uuids_to_delete]
    
    # 保存结果
    save_jsonl_file(filtered_data, output_path)
    
    print(f"原始数据数量: {len(full_data)}")
    print(f"删除数据数量: {len(uuids_to_delete)}")
    print(f"剩余数据数量: {len(filtered_data)}")

if __name__ == "__main__":
    # 使用示例
    full_set_path = "data/GPQA-data-20241211-standardized.jsonl"  # 全集文件路径
    subset_paths = ["data/mmludata.jsonl", "data/gpqa_1218data.jsonl", "data/delete.jsonl"]  # 待删除文件路径列表
    output_path = "data/gpqa_check/data_without_delete_20241220.jsonl"  # 输出文件路径
    
    delete_subset(full_set_path, subset_paths, output_path)
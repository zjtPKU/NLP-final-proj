import json
from typing import Dict, Any
from pathlib import Path
import uuid  # 新增：用于生成 UUID
import re
import random
random.seed(42)

def load_jsonl(file_path: str) -> list:
    """加载 jsonl 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: list, file_path: str) -> None:
    """保存为 jsonl 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def standardize_text(text):
    full_to_half = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':', '（': '(', '）': ')',
        '"': '"', '"': '"', ''': "'", ''': "'",
        '【': '[', '】': ']', '『': '{', '』': '}',
        '、': ',', '《': '<', '》': '>', '～': '~',
        '「': '"', '」': '"', '｛': '{', '｝': '}',
        '［': '[', '］': ']', '／': '/', '＼': '\\',
        '｜': '|', '－': '-', '＿': '_', '＋': '+',
        '＝': '=', '＜': '<', '＞': '>', '％': '%',
        '＃': '#', '＆': '&', '＊': '*', '＠': '@',
        '︰': ':',
    }

    # 替换转义字符
    # text = re.sub(r'([^\\])\\([a-mo-sv-z])', r'\1\\\\\2', text)
    text = text.replace('\to', '\\to').replace('\tan', '\\tan').replace('\theta', '\\theta').replace('\text', '\\text').replace('\triangle', '\\triangle').replace('\times', '\\times')

    for full, half in full_to_half.items():
        text = text.replace(full, half)

    return re.sub(r' *\n *', '\n', re.sub(r'[ ]+', ' ', re.sub(r'\n+', '\n', re.sub(r'\t+', '\t', text)))).strip()

def standardize_format(item: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
    """统一数据格式"""
    standardized = {}
    for key in template:
        if key == "options":
            standardized[key] = [standardize_text(option) for option in item.get(key, template[key])]
            random.shuffle(standardized[key])
        elif key == "answer":
            standardized[key] = standardize_text(item.get(key, template[key]))
        elif key == "question":
            standardized[key] = standardize_text(item.get(key, template[key]))
        # elif key == "answer_letter":
        #     standardized[key] = chr(65 + standardized["options"].index(standardized["answer"]))
        else:
            standardized[key] = item.get(key, template[key])
    return standardized

def generate_unique_uuid(existing_uuids: set) -> str:
    """生成不重复的 UUID"""
    while True:
        new_uuid = uuid.uuid4().hex
        if new_uuid not in existing_uuids:
            return new_uuid

def main():
    # 定义标准格式模板
    template = {
        "uuid": "",
        "question": "",
        "options": [],
        "answer": "",
        # "answer_letter": "",
        "difficulty": "",
        "category": "",
        "subcategory": "",
    }

    # 读取主数据集
    main_data = load_jsonl("data/GPQA-data-20241211-standardized.jsonl")
    
    # 读取待合并的数据集
    # TODO: 替换为实际的待合并数据集路径
    merge_data = load_jsonl("results/gpqa/Qwen2.5-72B-Instruct_only_one_correct_at_most_samples_filtered_gen_confusion_options.jsonl")
    
    # 获取主数据集中的所有 UUID
    existing_uuids = {item["uuid"] for item in main_data}
    
    # 为待合并数据生成新的 UUID
    for item in merge_data:
        item["uuid"] = generate_unique_uuid(existing_uuids)
        existing_uuids.add(item["uuid"])
    
    # 统一格式
    standardized_merge_data = [
        standardize_format(item, template) for item in merge_data
    ]
    
    # 合并数据集
    merged_data = main_data + standardized_merge_data
    
    # 保存结果
    output_path = "data/all_data.jsonl"
    save_jsonl(merged_data, output_path)
    print(f"合并完成，共 {len(merged_data)} 条数据")

    # 保存标准化合并数据
    standardized_merge_data_path = "data/toprp-with-confusion-options.jsonl"
    save_jsonl(standardized_merge_data, standardized_merge_data_path)
    print(f"标准化合并数据保存完成，共 {len(standardized_merge_data)} 条数据")

if __name__ == "__main__":
    main()

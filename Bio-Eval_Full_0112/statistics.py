import json
from collections import defaultdict

def count_fields_and_subfields(file_path):
    # 用于统计field_final和subfield_final的数量和分布
    field_counts = defaultdict(int)
    subfield_counts = defaultdict(int)
    field_subfield_map = defaultdict(lambda: defaultdict(int))
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            
            # 提取field_final和subfield_final字段
            field = data.get("field_final", "Unknown")
            subfield = data.get("subfield_final", "Unknown")
            
            # 更新计数
            field_counts[field] += 1
            subfield_counts[subfield] += 1
            field_subfield_map[field][subfield] += 1

    # 打印统计结果
    print("Field Counts:")
    for field, count in field_counts.items():
        print(f"{field}: {count}")

    print("\nSubfield Counts:")
    for subfield, count in subfield_counts.items():
        print(f"{subfield}: {count}")

    print("\nField to Subfield Distribution:")
    for field, subfields in field_subfield_map.items():
        print(f"{field}:")
        for subfield, count in subfields.items():
            print(f"  {subfield}: {count}")

    return field_counts, subfield_counts, field_subfield_map

# 调用函数，传入JSONL文件路径
file_path = '/Users/zhoujt/Desktop/NLP-final-proj/gpqa_final_data_0112/bio.jsonl'  # 替换为实际文件路径
field_counts, subfield_counts, field_subfield_map = count_fields_and_subfields(file_path)

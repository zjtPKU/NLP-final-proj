import json

def extract_specific_entries(input_file, output_file):
    extracted_entries = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if "field_final" in entry and ("医学" in entry["field_final"] or "生物学" in entry["field_final"]):
                extracted_entries.append(entry)

    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in extracted_entries:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"提取完成，共找到 {len(extracted_entries)} 个条目。")

# 输入和输出文件路径
input_file = "/Users/zhoujt/Desktop/NLP-final-proj/gpqa_final_data_0112/gpqa_final_data_0112_standardized.jsonl"
output_file = "bio.jsonl"

# 调用函数
extract_specific_entries(input_file, output_file)

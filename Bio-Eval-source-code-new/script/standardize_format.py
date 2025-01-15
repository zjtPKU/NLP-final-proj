import json
import re
import random

random.seed(42)

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
    # text = re.sub(r'(^\\)\\([a-mo-sv-z])', r'\1\\\\\2', text)
    text = text.replace('\to', '\\to').replace('\tan', '\\tan').replace('\theta', '\\theta').replace('\text', '\\text').replace('\triangle', '\\triangle').replace('\times', '\\times')

    for full, half in full_to_half.items():
        text = text.replace(full, half)

    return re.sub(r' *\n *', '\n', re.sub(r'[ ]+', ' ', re.sub(r'\n+', '\n', re.sub(r'\t+', '\t', text)))).strip()

def standardize_format(input_path, output_path):
    with open(input_path, "r") as f, open(output_path, "w") as f_out:
        for line in f:
            sample = json.loads(line)
            std_sample = template.copy()
            
            std_sample["uuid"] = sample["uuid"]
            std_sample["question"] = standardize_text(sample["question"])
            std_sample["options"] = [standardize_text(option) for option in sample["options"]]
            random.shuffle(std_sample["options"])
            if "answer_letter" in sample:
                std_sample["answer"] = standardize_text(sample[f'choices_{sample["answer_letter"].lower()}'])
            else:
                std_sample["answer"] = sample["answer"]
            # std_sample["answer_letter"] = chr(65 + std_sample["options"].index(std_sample["answer"]))
            std_sample["difficulty"] = sample["difficulty"]
            std_sample["category"] = sample["category"]
            std_sample["subcategory"] = sample["subcategory"]
            
            f_out.write(json.dumps(std_sample, ensure_ascii=False) + "\n")
    print(f"Standardized format saved to {output_path}")

if __name__ == "__main__":

    input_path = "data/GPQA-data-processed-by-lnn-20241211-version2.jsonl"
    output_path = "data/GPQA-data-20241211-standardized.jsonl"
    
    standardize_format(input_path, output_path)

    input_path = "data/gpqa_1218data.jsonl"
    output_path = "data/gpqa_check/gpqa_1218data-standardized.jsonl"
    
    standardize_format(input_path, output_path)

    input_path = "data/mmludata.jsonl"
    output_path = "data/gpqa_check/mmludata-standardized.jsonl"
    
    standardize_format(input_path, output_path)

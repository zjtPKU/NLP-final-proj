import os
import json
import re

def extract_options_from_prompt(prompt, target_options, uuid):
    result_dict = {}
    
    # 遍历每个目标选项
    for option in target_options:
        # 查找选项在prompt中的所有位置
        occurrences = []
        start = 0
        valid_letters = []  # 存储所有符合条件的字母
        
        while True:
            index = prompt.find(option, start)
            if index == -1:
                break
            occurrences.append(index)
            start = index + 1
        
        # 如果选项存在
        if occurrences:
            # 检查每个出现位置的前三个字符和后续字符
            for pos in occurrences:
                if pos >= 3 and pos + len(option) < len(prompt):  # 确保有足够的前导字符和后续字符
                    prefix = prompt[pos-3:pos]
                    current_letter = prefix[0]
                    
                    # 检查格式是否为"X) "，其中X为A-J的字母
                    if not re.match(r'[A-J]\) ', prefix):
                        continue
                        
                    # 检查后面是否为换行符
                    if prompt[pos + len(option)] != '\n':
                        continue
                        
                    # 检查换行符后的内容
                    next_pos = pos + len(option) + 1
                    if next_pos >= len(prompt):
                        # 情况1: 字符串已结束
                        valid_letters.append(current_letter)
                    else:
                        # 情况2: 检查下一个字符是否为更靠后的字母选项
                        next_three_chars = prompt[next_pos:next_pos+3] if next_pos+3 <= len(prompt) else ""
                        if next_three_chars and re.match(r'[A-J]\) ', next_three_chars):
                            next_letter = next_three_chars[0]
                            if ord(next_letter) > ord(current_letter):
                                valid_letters.append(current_letter)
            
            # 如果找到多个匹配的字母，报错
            if len(valid_letters) > 1:
                raise ValueError(f"选项 '{option}' 在文本中多次出现且都符合格式要求，对应字母: {valid_letters}")
            elif len(valid_letters) == 1:
                result_dict[valid_letters[0]] = option
        
    # 检查字母是否连续
    letters = sorted(result_dict.keys())
    # print('letters:', letters)
    # print('result_dict:', result_dict)
    expected_letters = [chr(ord('A') + i) for i in range(len(letters))]
    if letters != expected_letters:
        missing = set(expected_letters) - set(letters)
        # raise ValueError(f"选项字母不连续，缺少: {missing}")
        print(f"uuid: {uuid}, 选项字母不连续，缺少: {missing}")
    
    # 按字母顺序转换为列表
    return [result_dict[letter] for letter in sorted(result_dict.keys())]

def check_options():
    # 读取JSONL文件
    jsonl_path = "data/GPQA-data-processed-by-lnn-20241211-version2.jsonl"
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompt = data['prompt']

            target_options = data['options']
            # print('uuid:', data['uuid'])
            # 从prompt中提取选项
            extracted_options = extract_options_from_prompt(prompt, target_options, data['uuid'])
            
            # 比较两个选项列表是否相等
            if len(extracted_options) != len(target_options):
                print(f"uuid: {data['uuid']}, 选项数量不匹配: 提取到 {len(extracted_options)} 个，目标有 {len(target_options)} 个")
                count += 1
    print(f"共发现 {count} 条数据选项数量不匹配")
            # elif extracted_options != target_options:
            #     print("选项内容不匹配:")
            #     print("提取的选项:", extracted_options)
            #     print("目标选项:", target_options)
            #     print("---")

check_options()
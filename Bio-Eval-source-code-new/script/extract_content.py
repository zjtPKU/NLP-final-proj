import os
import json
import re
import glob
import multiprocessing as mp
import argparse

pattern = re.compile(r'\{[^}]*\}')

def process_file(file_path, keys_to_extract, overwrite=False):
    try:
        # 读取原始文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        updated_lines = []
        parse_fail_count = 0
        total_count = len(lines)

        for line in lines:
            try:
                json_obj = json.loads(line)
                response = json_obj.get('response', '')
                if isinstance(response, str):
                    # 使用正则表达式提取字典内容
                    match = pattern.search(response)
                    if match:   
                        extracted_str = match.group(0)
                        # 去除可能存在的多余逗号
                        extracted_str = re.sub(r',\s*\}', '}', extracted_str)
                        
                        # 修复缺少引号的键和值
                        extracted_str = re.sub(r'(?<=\{)\s*([^"]+?)\s*:', r'"\1":', extracted_str)
                        # extracted_str = re.sub(r':\s*([^",}]+)\s*(?=[,}])', r': "\1"', extracted_str)
                        
                        # 处理没有多余逗号的情况
                        extracted_str = re.sub(r'(?<=\})\s*$', '', extracted_str)
                        
                        extracted_dict = json.loads(extracted_str)

                        # 将提取到的键值对追加到 JSON 对象中
                        for key, value in keys_to_extract.items():
                            json_obj[value] = extracted_dict.get(key)
                    else:
                        # 如果匹配失败，修改 response 为包含错误信息的字典
                        json_obj['response'] = {
                            "error": 'extract_content_failed',
                            "error_response": response
                        }
                        parse_fail_count += 1

                    updated_lines.append(json.dumps(json_obj, ensure_ascii=False) + '\n')
                else:
                    updated_lines.append(line)
                    parse_fail_count += 1
            except json.JSONDecodeError:
                # 解析失败，修改 response 为包含错误信息的字典
                json_obj['response'] = {
                    "error": "extract_content_failed",
                    "error_response": response
                }
                updated_lines.append(json.dumps(json_obj, ensure_ascii=False) + '\n')
                parse_fail_count += 1

        # 计算解析失败的比例
        parse_fail_ratio = parse_fail_count / total_count * 100
        print(f"Processed file: {file_path}, Failed parses: {parse_fail_count}/{total_count} ({parse_fail_ratio:.2f}%)")

        # 将更新后的内容写回文件
        if overwrite:
            output_path = file_path
        else:
            output_path = file_path.replace('.jsonl', '_extracted.jsonl')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Process JSONL files to extract specific keys.')
    parser.add_argument('--file_pattern', type=str, default='/map-vepfs/xinrun/KOR-Bench/results/gpqa/filter/Qwen2.5-72B-Instruct_dk-data-toppr-struct-physics-chem-bio-2nd-filter_gpqa-filter*', help='Glob pattern for files to process')
    parser.add_argument('--keys_to_extract', type=json.loads, default='{"quality": "quality", "difficulty": "difficulty"}', help='Dictionary of keys to extract from JSON with default values')
    # parser.add_argument('--keys_to_extract', type=json.loads, default='{"tag": "tag", "difficulty": "difficulty2filter"}', help='Dictionary of keys to extract from JSON with default values')
    # parser.add_argument('--keys_to_extract', type=json.loads, default='{"subject": "subcategory_level_2"}', help='Dictionary of keys to extract from JSON with default values')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite original files if set')
    args = parser.parse_args()

    # 获取所有匹配的文件路径
    all_files = glob.glob(args.file_pattern)

    # 使用多进程处理所有文件
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(process_file, [(file_path, args.keys_to_extract, args.overwrite) for file_path in all_files])

if __name__ == '__main__':
    main()

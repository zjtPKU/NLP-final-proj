import orjson
import os
from pathlib import Path
from collections import defaultdict

def check_all():
    # 存储所有uuid以检查唯一性
    uuid_dict = defaultdict(list)
    # 预期的key列表
    expected_keys = {'uuid', 'question', 'answer', 'score'}  # 根据实际需求修改
    
    results_dir = Path('results/gpqa')
    for json_file in results_dir.glob('*.jsonl'):
        try:
            with open(json_file, 'r') as f:
                data = orjson.loads(f.read())
            
            # 检查每条记录的key
            for item in data:
                # 检查key是否符合模板
                if not isinstance(item, dict):
                    print(f"错误：{json_file}中存在非字典类型的数据")
                    continue
                    
                item_keys = set(item.keys())
                if item_keys != expected_keys:
                    print(f"警告：{json_file}中的keys不匹配")
                    print(f"缺少的keys: {expected_keys - item_keys}")
                    print(f"多余的keys: {item_keys - expected_keys}")
                
                # 检查uuid唯一性
                if 'uuid' in item:
                    uuid_dict[item['uuid']].append(str(json_file))

    # 输出重复的uuid
    for uuid, files in uuid_dict.items():
        if len(files) > 1:
            print(f"发现重复的UUID {uuid} 在以下文件中：")
            for file in files:
                print(f"  - {file}")

if __name__ == '__main__':
    check_all()
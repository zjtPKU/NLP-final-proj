import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

def read_jsonl_files(file_list):
    data = []
    for file_path in tqdm(file_list, desc='reading file_list'):
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    data.append(json.loads(line))
    return data

def extract_values(data):
    quality = []
    difficulty = []
    quality_count = {i: 0 for i in range(1, 11)}
    difficulty_count = {i: 0 for i in range(1, 11)}
    
    for item in data:
        if 'quality' in item and 'difficulty' in item:
            q = item['quality']
            d = item['difficulty']
            if q is not None and d is not None:
                quality.append(q)
                difficulty.append(d)
                quality_count[q] += 1
                difficulty_count[d] += 1
    
    # 打印每个质量和难度分段的数量
    print("Quality distribution:", quality_count)
    print("Difficulty distribution:", difficulty_count)
    
    return quality, difficulty

def plot_heatmap(quality, difficulty):
    heatmap_data, xedges, yedges = np.histogram2d(difficulty, quality, bins=(10, 10), range=[[1, 10], [1, 10]])
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data, 
        xticklabels=np.arange(1, 11), 
        yticklabels=np.arange(1, 11), 
        cmap='YlGnBu', 
        annot=True, 
        fmt='.0f', 
        annot_kws={"size": 8}
    )
    plt.xlabel('Quality')
    plt.ylabel('Difficulty')
    plt.title('Heatmap of Quality vs Difficulty')
    # plt.show()
    plt.savefig('plot/gpqa/heatmap_qwen_dk-data-toppr-struct-physics-chem-bio-2nd-filter.png')

def main():
    all_files = glob.glob('/map-vepfs/xinrun/KOR-Bench/results/gpqa/filter/Qwen2.5-72B-Instruct_dk-data-toppr-struct-physics-chem-bio-2nd-filter_gpqa-filter.jsonl')
    data = read_jsonl_files(all_files)
    quality, difficulty = extract_values(data)
    plot_heatmap(quality, difficulty)

if __name__ == "__main__":
    main()
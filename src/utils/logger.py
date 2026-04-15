import json
import os

def save_json(data, file_path):
    """保存字典结果为 JSON 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(file_path):
    """读取 JSON 结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
import json
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy类型和布尔值"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (bool, type(None))):
            return obj
        else:
            return super().default(obj)
        
def save_json(data, file_path):
    """保存字典结果为 JSON 文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

def load_json(file_path):
    """读取 JSON 结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
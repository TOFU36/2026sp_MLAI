import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM1D:
    """提取 1D-CNN 注意力热力图 (Phase 4.2 & 3.5)"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册 Hook 以截获特征和梯度
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # 全局平均池化梯度作为权重
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam) # 只保留正向影响
        
        # 插值回原始序列长度
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2], mode='linear').squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # 归一化
        return cam.detach().cpu().numpy()
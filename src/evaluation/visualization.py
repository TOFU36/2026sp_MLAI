import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_dynamics(results_dict, save_path=None):
    """绘制多个模型的训练动力学曲线 (Phase 1 & Phase 2)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for name, hist in results_dict.items():
        axes[0].plot(hist['train_loss'], label=name, marker='o', markersize=4)
        axes[1].plot(hist['val_f1'], label=f"{name} (Best: {max(hist['val_f1']):.4f})", marker='s', markersize=4)

    axes[0].set_title("Training Loss Dynamics")
    axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Weighted CE Loss")
    axes[0].legend()

    axes[1].set_title("Validation F1-Macro")
    axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("F1-Macro Score")
    axes[1].legend()
    
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_robustness_curve(jitters, dl_scores, ml_scores, save_path=None):
    """绘制临床 R 波漂移抗性曲线 (Phase 3.5)"""
    plt.figure(figsize=(8, 5))
    plt.plot(jitters, dl_scores, marker='o', color='purple', linewidth=2, label='1D-CNN (InceptionTime)')
    plt.plot(jitters, ml_scores, marker='s', color='gray', linestyle='--', label='ML Baseline (RF)')
    plt.title("R-Peak Misalignment Robustness (Clinical Scenario)")
    plt.xlabel("Uniform Jitter Steps (+/-)"); plt.ylabel("F1-Macro Score")
    plt.ylim(0, 1.0)
    plt.legend()
    
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_gradcam(signal, cam_heatmap, title="Grad-CAM Attention Heatmap", save_path=None):
    """可视化模型的视线追踪 (Phase 4)"""
    plt.figure(figsize=(12, 4))
    time_steps = np.arange(len(signal))
    plt.plot(time_steps, signal, color='black', linewidth=1.5, label='ECG Signal')
    
    plt.scatter(time_steps, signal, c=cam_heatmap, cmap='jet', alpha=0.8, s=60, zorder=5)
    plt.title(title)
    plt.xlabel("Time Step"); plt.ylabel("Amplitude")
    plt.colorbar(label='Attention Weight')
    plt.legend()
    
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show()
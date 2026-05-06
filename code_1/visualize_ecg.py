import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_ecg_subplots(csv_path, n_points=187, seed=5416, pick="first"):
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    label_col = df.columns[-1]
    
    # MIT-BIH 类别说明
    class_mapping = {
        0: 'Normal (N)',
        1: 'Supraventricular (S)',
        2: 'Ventricular (V)',
        3: 'Fusion (F)',
        4: 'Unknown (Q)'
    }

    classes = sorted(df[label_col].astype(int).unique())
    
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, len(classes), figsize=(20,4))

    for i, c in enumerate(classes):
        sub = df[df[label_col] == c]

        if pick == "random":
            row = sub.iloc[rng.integers(0, len(sub))]
        else:
            row = sub.iloc[0]

        ecg = row.iloc[:n_points].values.astype(float)

        axes[i].plot(ecg)
        axes[i].set_title(f"{class_mapping.get(c, c)}")
        axes[i].set_xlabel("Time step")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# 运行
plot_ecg_subplots("mitbih_train_downsampled_3000.csv")
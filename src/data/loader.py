import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data.dataset import ECGDataset

DEFAULT_TRAIN_PATH = 'data/raw/mitbih_train.csv'
DEFAULT_TEST_PATH = 'data/raw/mitbih_test.csv'


def load_ecg_csv(path):
    """加载并预处理 ECG CSV 文件（无表头、强制数值类型、填充 NaN）"""
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


def load_train_test(train_path=DEFAULT_TRAIN_PATH, test_path=DEFAULT_TEST_PATH,
                    sample_ratio=1.0):
    """加载训练集和测试集。

    Args:
        sample_ratio: 训练集加载比例（分层抽样），1.0 = 全量。测试集始终全量加载。
    """
    train_df = load_ecg_csv(train_path)
    test_df = load_ecg_csv(test_path)
    if sample_ratio < 1.0:
        train_df, _ = train_test_split(
            train_df, train_size=sample_ratio,
            stratify=train_df.iloc[:, -1], random_state=42,
        )
        train_df = train_df.reset_index(drop=True)
        print(f"  [数据] 训练集按 {sample_ratio:.0%} 分层抽样 → {len(train_df)} 行")
    print(f"  [数据] 训练集: {len(train_df)}, 测试集: {len(test_df)}")
    return train_df, test_df


def make_loader(df, feature_type='raw', batch_size=2048, shuffle=False,
                augment=False, max_jitter=0, num_workers=4):
    """创建单个 DataLoader。

    需要自定义参数（如 augment=True, max_jitter=10）时直接调用此函数。
    常规训练流程建议用 create_dataloaders 一次创建全部。
    """
    ds = ECGDataset(df, feature_type=feature_type, augment=augment, max_jitter=max_jitter)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def create_dataloaders(train_df, test_df, feature_type='raw', batch_size=2048,
                       num_workers=4, val_ratio=0.1, random_state=42, augment=False):
    """创建 train / val / test 三个 DataLoader。

    从 train_df 中分层抽样分出 val_ratio 作为验证集（early stopping / 模型选择），
    test_df 仅用于最终评估，训练过程中从未见过。

    Returns:
        (train_loader, val_loader, test_loader)
    """
    y = train_df.iloc[:, -1]
    fit_df, val_df = train_test_split(
        train_df, test_size=val_ratio,
        stratify=y, random_state=random_state,
    )
    fit_df = fit_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"  [分割] 训练: {len(fit_df)}, 验证: {len(val_df)}, 测试: {len(test_df)}")

    train_loader = make_loader(fit_df, feature_type=feature_type, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, augment=augment)
    val_loader = make_loader(val_df, feature_type=feature_type, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    test_loader = make_loader(test_df, feature_type=feature_type, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

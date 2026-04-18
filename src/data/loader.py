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


def make_loader(df, feature_type='raw', batch_size=256, shuffle=False,
                augment=False, max_jitter=0, num_workers=4):
    """创建单个 DataLoader（底层函数）。

    所有 DataLoader 统一走这里创建，确保 num_workers / pin_memory 一致。
    需要自定义参数（如 augment=True, max_jitter=10）时直接调用此函数。

    See also: create_dataloaders —— 基于此函数的便捷包装，一次创建训练+测试两个 loader。
    """
    ds = ECGDataset(df, feature_type=feature_type, augment=augment, max_jitter=max_jitter)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def create_dataloaders(train_df, test_df, feature_type='raw', batch_size=256,
                       num_workers=4):
    """便捷函数：一次性创建训练 DataLoader + 测试 DataLoader。

    内部就是调用两次 make_loader：
      - 训练集: shuffle=True
      - 测试集: shuffle=False
    """
    train_loader = make_loader(train_df, feature_type=feature_type,
                               batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = make_loader(test_df, feature_type=feature_type,
                              batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

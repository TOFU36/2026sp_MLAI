import numpy as np
import pandas as pd


def load_csv(path):
    df = pd.read_csv(path, header=None)
    data = df.values
    X = data[:, :-1].astype(float)
    y = data[:, -1]
    return X, y


def load_csv_robust(path, sep=','):
    """按行读取 CSV，修复类似 '1.234e-02.1' 的粘连字段。

    算法：逐 token 尝试解析为 float；若失败，向左在最后一个 '.' 处拆分，
    将拆出的右侧当作下一个 token（通常为小整数后缀）。重复直到左侧可解析为 float。
    最终把每行的最后一项视为标签。返回 X (2D float ndarray) 和 y (1D ndarray)。
    """
    rows = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            tokens = []
            i = 0
            while i < len(parts):
                tok = parts[i].strip()
                if tok == '':
                    tokens.append('0')
                    i += 1
                    continue
                # try parse
                try:
                    float(tok)
                    tokens.append(tok)
                    i += 1
                except Exception:
                    # try to split at last '.' repeatedly
                    split_ok = False
                    while True:
                        if '.' not in tok:
                            break
                        left, right = tok.rsplit('.', 1)
                        if left == '':
                            break
                        try:
                            float(left)
                            # left is a valid float; push left and put right back as next token
                            tokens.append(left)
                            # insert the right part back into parts at position i+1
                            parts.insert(i + 1, right)
                            split_ok = True
                            break
                        except Exception:
                            tok = left
                            continue
                    if not split_ok:
                        # fallback: try to remove any non-numeric suffix
                        # remove trailing non-digit characters
                        import re
                        m = re.match(r"^([\-+0-9.eE]+)([^0-9]*)$", tok)
                        if m:
                            tokens.append(m.group(1))
                        else:
                            # as last resort, replace with 0
                            tokens.append('0')
                    i += 1
            # last token is label
            if len(tokens) == 0:
                continue
            *vals, lab = tokens
            # convert values to float
            try:
                row = [float(v) for v in vals]
            except Exception:
                # replace unparsable with 0.0
                row = []
                for v in vals:
                    try:
                        row.append(float(v))
                    except Exception:
                        row.append(0.0)
            # parse label
            try:
                yval = float(lab)
                # if label is integer-like, cast to int
                if abs(yval - int(yval)) < 1e-8:
                    yval = int(yval)
            except Exception:
                # try strip non-digits
                import re
                m = re.search(r"(\d+)$", lab)
                if m:
                    yval = int(m.group(1))
                else:
                    yval = lab
            rows.append(row)
            labels.append(yval)
    # pad rows to same length with zeros
    maxlen = max((len(r) for r in rows), default=0)
    X = np.zeros((len(rows), maxlen), dtype=float)
    for i, r in enumerate(rows):
        X[i, :len(r)] = r
    y = np.array(labels)
    return X, y


def detect_effective_length(signal, threshold=1e-8):
    nonzero = np.where(np.abs(signal) > threshold)[0]
    if nonzero.size == 0:
        return 0
    return nonzero[-1] + 1


def trim_trailing_zeros(X, threshold=1e-8):
    lengths = [detect_effective_length(row, threshold) for row in X]
    trimmed = [row[:l] if l > 0 else np.array([]) for row, l in zip(X, lengths)]
    return trimmed, np.array(lengths)


def pad_to_length(list_of_signals, length=None, pad_value=0.0):
    if length is None:
        length = int(np.median([s.size for s in list_of_signals if s.size > 0]))
    out = np.full((len(list_of_signals), length), pad_value, dtype=float)
    for i, s in enumerate(list_of_signals):
        L = min(s.size, length)
        if L > 0:
            out[i, :L] = s[:L]
    return out


def normalize_signals(X, axis=1, eps=1e-8):
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mean) / (std + eps)

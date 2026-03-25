import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq


def time_domain_features(signal):
    if signal.size == 0:
        return np.zeros(8, dtype=float)
    mean = signal.mean()
    var = signal.var()
    skew = float(stats.skew(signal))
    kurt = float(stats.kurtosis(signal))
    maximum = signal.max()
    minimum = signal.min()
    energy = float((signal ** 2).sum())
    zero_crossings = float(((signal[:-1] * signal[1:]) < 0).sum())
    return np.array([mean, var, skew, kurt, maximum, minimum, energy, zero_crossings], dtype=float)


def fft_features(signal, fs=300.0, n_bands=6):
    if signal.size == 0:
        return np.zeros(n_bands + 1, dtype=float)
    N = signal.size
    yf = np.abs(rfft(signal))
    xf = rfftfreq(N, 1.0 / fs)
    total_energy = yf.sum() + 1e-12
    band_edges = np.linspace(0, xf.max(), n_bands + 1)
    band_energy = []
    for i in range(n_bands):
        mask = (xf >= band_edges[i]) & (xf < band_edges[i + 1])
        band_energy.append(yf[mask].sum() / total_energy)
    dominant_freq = xf[yf.argmax()] if yf.size > 0 else 0.0
    return np.concatenate([np.array(band_energy, dtype=float), np.array([dominant_freq], dtype=float)])


def extract_features_list(list_of_signals, fs=300.0):
    feats = []
    for s in list_of_signals:
        t = time_domain_features(s)
        f = fft_features(s, fs=fs)
        feats.append(np.concatenate([t, f]))
    return np.vstack(feats)

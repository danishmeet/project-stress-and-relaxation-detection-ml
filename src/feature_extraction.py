import numpy as np
from scipy.signal import find_peaks

def extract_hrv_features(ecg_signal, fs):
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    peaks, _ = find_peaks(ecg_signal, distance=0.6 * fs)

    if len(peaks) < 10:
        return None

    rr_intervals = np.diff(peaks) / fs
    if len(rr_intervals) < 5:
        return None

    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    mean_hr = 60 / mean_rr

    return {
        "Mean_RR": mean_rr,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "Mean_HR": mean_hr
    }


def extract_resp_features(resp_signal, fs):
    resp_signal = resp_signal - np.mean(resp_signal)
    zero_crossings = np.where(np.diff(np.sign(resp_signal)))[0]

    duration_sec = len(resp_signal) / fs
    breaths = len(zero_crossings) / 2

    if duration_sec == 0:
        return None

    return {
        "Resp_Rate": (breaths / duration_sec) * 60,
        "Resp_Variability": np.std(resp_signal)
    }

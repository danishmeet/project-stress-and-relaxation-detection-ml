import numpy as np

def create_windows(signal, labels, window_size):
    windows = []
    window_labels = []

    for start in range(0, len(signal) - window_size, window_size):
        end = start + window_size
        label_segment = labels[start:end]
        label_segment = label_segment[label_segment != 0]

        if len(label_segment) == 0:
            continue

        label = np.bincount(label_segment).argmax()
        windows.append(signal[start:end])
        window_labels.append(label)

    return windows, window_labels

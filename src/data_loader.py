import pickle
import numpy as np


def load_adanen_stream(file_path):
    with open(file_path, 'rb') as f:
        stream = pickle.load(f)
    return stream


def stream_generator(stream):
    for row in stream:
        features = np.array(row[0]).flatten()
        x = {f'feature_{i}': float(val) for i, val in enumerate(features)}

        y = row[1]
        if hasattr(y, 'item'):
            y = y.item()

        yield x, y
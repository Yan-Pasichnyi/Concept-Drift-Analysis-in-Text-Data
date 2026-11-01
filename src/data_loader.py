import pickle
import numpy as np


def load_adanen_stream(file_path):
    # load pickled stream (list of (features, label))
    with open(file_path, 'rb') as f:
        stream = pickle.load(f)
    return stream


def stream_generator(stream):
    # yield one instance at a time as (features dict, label)
    for row in stream:
        features = np.array(row[0]).flatten()
        # River needs features as a dict
        x = {f'feature_{i}': float(val) for i, val in enumerate(features)}

        y = row[1]
        # numpy label to python number
        if hasattr(y, 'item'):
            y = y.item()

        yield x, y
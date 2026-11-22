import os
import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

CATEGORIES = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x',
    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
    'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
    'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc',
]

# AdaNEN 20NG drift schedule: (start_index, transition_width, topic_to_label)
DRIFT_SCHEDULE = [
    (0,     0,   {'comp': 1, 'rec': 0, 'sci': 1, 'talk': 0}),
    (4000,  50,  {'comp': 1, 'rec': 0, 'sci': 0, 'talk': 0}),
    (4500,  100, {'comp': 0, 'rec': 1, 'sci': 0, 'talk': 0}),
    (5000,  25,  {'comp': 0, 'rec': 0, 'sci': 1, 'talk': 0}),
    (5500,  200, {'comp': 0, 'rec': 0, 'sci': 0, 'talk': 1}),
    (6000,  50,  {'comp': 0, 'rec': 0, 'sci': 1, 'talk': 1}),
    (6500,  25,  {'comp': 0, 'rec': 1, 'sci': 1, 'talk': 0}),
    (7000,  150, {'comp': 1, 'rec': 1, 'sci': 0, 'talk': 0}),
    (7500,  100, {'comp': 1, 'rec': 0, 'sci': 1, 'talk': 0}),
    (8000,  75,  {'comp': 1, 'rec': 0, 'sci': 0, 'talk': 1}),
    (8500,  250, {'comp': 0, 'rec': 1, 'sci': 1, 'talk': 0}),
    (9000,  125, {'comp': 0, 'rec': 1, 'sci': 0, 'talk': 1}),
    (9500,  75,  {'comp': 0, 'rec': 1, 'sci': 1, 'talk': 1}),
    (10000, 50,  {'comp': 1, 'rec': 0, 'sci': 1, 'talk': 1}),
    (10500, 100, {'comp': 1, 'rec': 1, 'sci': 0, 'talk': 1}),
    (11000, 25,  {'comp': 1, 'rec': 1, 'sci': 1, 'talk': 0}),
    (11500, 50,  {'comp': 1, 'rec': 0, 'sci': 1, 'talk': 0}),
]

TRUE_DRIFTS = [4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500,
               8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500]
MODEL_NAME = 'BAAI/bge-small-en-v1.5'


def load_raw_20ng(seed=42):
    dataset = fetch_20newsgroups(
        subset='all',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
    )

    samples = []
    for text, target_idx in zip(dataset.data, dataset.target):
        # remove extra spaces and newlines
        clean_text = ' '.join(text.split())
        if len(clean_text) < 20:
            continue
        topic = dataset.target_names[target_idx].split('.')[0]
        samples.append((clean_text, topic))

    # shuffle documents with fixed seed
    random.Random(seed).shuffle(samples)

    texts = [item[0] for item in samples]
    topics = [item[1] for item in samples]
    return texts, topics


def get_embeddings(cache_path='../data/bge_20ng_cache.npz', seed=42):
    # load from cache if already computed
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data['embeddings'], data['topics'].tolist()

    texts, topics = load_raw_20ng(seed=seed)
    encoder = SentenceTransformer(MODEL_NAME)
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, topics=np.array(topics))
    return embeddings, topics


def get_label(index, topic, mode, rng):
    # find current drift stage
    stage = 0
    for i in range(1, len(DRIFT_SCHEDULE)):
        start_index = DRIFT_SCHEDULE[i][0]
        if index >= start_index:
            stage = i

    start_index, width, new_rules = DRIFT_SCHEDULE[stage]

    # gradual drift: mix old and new rules inside the transition window
    if mode == 'gradual' and stage > 0:
        steps_passed = index - start_index
        if steps_passed < width:
            prob_new = steps_passed / width
            old_rules = DRIFT_SCHEDULE[stage - 1][2]
            if rng.random() < prob_new:
                return new_rules[topic]
            else:
                return old_rules[topic]

    return new_rules[topic]


def stream_generator(mode='abrupt', cache_path='../data/bge_20ng_cache.npz', seed=42):
    embeddings, topics = get_embeddings(cache_path=cache_path, seed=seed)
    rng = random.Random(seed)

    for i in range(len(embeddings)):
        vec = embeddings[i]
        topic = topics[i]

        x = {f'feature_{j}': float(val) for j, val in enumerate(vec)}
        y = get_label(i, topic, mode, rng)
        yield x, y
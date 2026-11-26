import bisect
import random
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_20newsgroups

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

STAGE_STARTS = [stage[0] for stage in DRIFT_SCHEDULE]

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
MIN_TEXT_LEN = 20
SKLEARN_SEED = 42

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / 'data' / 'bge_20ng_canonical.npz'

def _cache_meta():
    # everything that changes the cached embeddings; a mismatch invalidates the cache
    return f"{MODEL_NAME}|{MIN_TEXT_LEN}|{SKLEARN_SEED}|{','.join(CATEGORIES)}"

def load_raw_20ng():
    dataset = fetch_20newsgroups(
        subset='all',
        categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=SKLEARN_SEED,
    )

    texts, topics = [], []
    for text, target_idx in zip(dataset.data, dataset.target):
        # remove extra spaces and newlines
        clean_text = ' '.join(text.split())
        if len(clean_text) < MIN_TEXT_LEN:
            continue
        texts.append(clean_text)
        topics.append(dataset.target_names[target_idx].split('.')[0])
    return texts, topics


def get_canonical_embeddings(cache_path=None):
    cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH

    if cache_path.exists():
        data = np.load(cache_path)  # plain arrays only, no pickle needed
        if 'meta' in data.files and str(data['meta']) == _cache_meta():
            return data['embeddings'], data['topics'].tolist()
        print(f'Cache {cache_path} does not match the current settings, recomputing.')

    from sentence_transformers import SentenceTransformer

    texts, topics = load_raw_20ng()
    encoder = SentenceTransformer(MODEL_NAME)
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        topics=np.array(topics),
        meta=np.array(_cache_meta()),
    )
    return embeddings, topics


def shuffled_order(n, seed):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def get_label(index, topic, mode, rng):
    # find current drift stage
    stage = bisect.bisect_right(STAGE_STARTS, index) - 1
    start_index, width, new_rules = DRIFT_SCHEDULE[stage]

    # gradual drift: mix old and new rules inside the transition window
    if mode == 'gradual' and stage > 0:
        steps_passed = index - start_index
        if steps_passed < width:
            prob_new = steps_passed / width
            old_rules = DRIFT_SCHEDULE[stage - 1][2]
            if rng.random() < prob_new:
                return new_rules[topic]
            return old_rules[topic]

    return new_rules[topic]


def build_labels(topics, mode, seed):
    rng = random.Random(seed)
    return np.array([get_label(i, topic, mode, rng) for i, topic in enumerate(topics)])


def load_stream_arrays(mode='abrupt', seed=42, cache_path=None):
    if mode not in ('abrupt', 'gradual'):
        raise ValueError(f"mode must be 'abrupt' or 'gradual', got {mode!r}")

    embeddings, topics = get_canonical_embeddings(cache_path)
    order = shuffled_order(len(topics), seed)
    embeddings = embeddings[order]
    topics = [topics[i] for i in order]
    labels = build_labels(topics, mode, seed)
    return embeddings, labels, topics


def stream_generator(mode='abrupt', seed=42, cache_path=None):
    embeddings, labels, _ = load_stream_arrays(mode, seed, cache_path)
    keys = [f'feature_{j}' for j in range(embeddings.shape[1])]

    for vec, y in zip(embeddings, labels):
        x = dict(zip(keys, vec.tolist()))
        yield x, int(y)
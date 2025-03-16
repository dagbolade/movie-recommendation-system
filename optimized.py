import pickle
import gzip
import numpy as np
from sklearn.preprocessing import normalize


def compress_models():
    # Load and compress similarity matrix
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)

    # Convert to float32 to reduce memory
    similarity = similarity.astype(np.float32)

    # Normalize and make sparse if possible
    similarity = normalize(similarity, copy=False)

    # Compress and save
    with gzip.open('similarity.pkl.gz', 'wb') as f:
        pickle.dump(similarity, f)

    # Compress other models
    models = ['movies_dict.pkl', 'model2.pkl', 'vectorizer.pkl']
    for model in models:
        with open(f'{model}', 'rb') as f:
            data = pickle.load(f)
        with gzip.open(f'{model}.gz', 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    compress_models()
import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)
    dot = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denominator = norm_a * norm_b
    if denominator == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
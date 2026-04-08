import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.array([]).reshape(0, 0)
        
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    n = len(seqs)
    padded_seqs = np.full((n, max_len), pad_value, dtype=np.float64)
    padded_seqs = np.full((n, max_len), pad_value)
    for i, seq in enumerate(seqs):
        truncated_seq = seq[:max_len]
        padded_seqs[i, :len(truncated_seq)] = truncated_seq
    
    return padded_seqs    
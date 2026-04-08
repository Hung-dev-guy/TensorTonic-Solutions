import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.array([]).reshape(0, 0)

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    n = len(seqs)
    
    # Create an array filled with pad_value
    # We infer dtype from the first sequence if possible, otherwise use default
    if hasattr(seqs[0], 'dtype'):
        dtype = seqs[0].dtype
    else:
        dtype = type(pad_value)
        
    padded_seqs = np.full((n, max_len), pad_value, dtype=dtype)
    
    for i, seq in enumerate(seqs):
        # Copy the sequence into the padded array, up to max_len
        end_idx = min(len(seq), max_len)
        padded_seqs[i, :end_idx] = seq[:end_idx]
        
    return padded_seqs
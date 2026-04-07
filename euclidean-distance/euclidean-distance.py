import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return float(np.linalg.norm(x_arr-y_arr))
    

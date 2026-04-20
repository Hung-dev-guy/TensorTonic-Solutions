import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    N = len(docs)
    
    if N == 0:
        return np.array([])
    
    doc_lens = [len(doc) for doc in docs]
    avgdl = sum(doc_lens) / N
    
    if avgdl == 0:
        avgdl = 1   # 🔥 fix chia 0
    
    # DF
    df = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    
    # IDF
    idf = {}
    for term in df:
        idf[term] = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
    
    scores = []
    
    for doc in docs:
        score = 0
        freq = Counter(doc)
        
        for term in query_tokens:
            if term not in freq:
                continue
            
            tf = freq[term]
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (len(doc) / avgdl))
            
            if denominator != 0:   # 🔥 thêm guard
                score += idf.get(term, 0) * (numerator / denominator)
        
        scores.append(score)
    
    return np.array(scores)
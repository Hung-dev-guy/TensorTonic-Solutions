import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # 1. Tokenize documents (tách từ đơn giản bằng space)
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # 2. Build vocabulary (tập tất cả từ)\
    all_words = []
    for doc in tokenized_docs:
        for word in doc:
            all_words.append(word)
            
    unique_words = set(all_words)
    vocab = sorted(unique_words)
    
    vocab_index = {}
    for i in range(len(vocab)):
        word = vocab[i]
        vocab_index[word] = i
        
    N = len(documents)
    
    # 3. Tính DF (document frequency)
    df = Counter()
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            df[word] += 1
    
    # 4. Tính IDF
    idf = {}
    for word in vocab:
        idf[word] = math.log(N / df[word])
    
    # 5. Tính TF-IDF matrix
    tfidf_matrix = np.zeros((N, len(vocab)))
    
    for i, doc in enumerate(tokenized_docs):
        tf = Counter(doc)
        total_words = len(doc)
        
        for word in doc:
            tf_value = tf[word] / total_words
            tfidf_matrix[i][vocab_index[word]] = tf_value * idf[word]
    
    return tfidf_matrix, vocab
import scipy.sparse
import numpy as np
from sklearn.preprocessing import normalize

#########################Similarity functions#####################


# def cosineSimilarity(mat):
#     "Compute similarity"
#     Ri = normalize(mat.tocsc(), axis=0)
#     return (Ri.T * Ri).tocsc()

def cosineSimilarity(mat):
    "Compute similarity"
    Ri = normalize(mat, axis=0)
    return (Ri.T * Ri)


def jaccardSimilarity(inputmatrix):
    numerator = inputmatrix.T * inputmatrix
    numerator = numerator.tolil()
    numerator.setdiag(0)
    numerator = numerator.tocsr()

    item_purchase_count = np.array(inputmatrix.sum(axis=0).tolist()[0])
    n_items = inputmatrix.shape[1]
    data, ci, indptr = numerator.data, numerator.indices, numerator.indptr
    denom = np.empty(data.shape)
    for i in xrange(n_items):
        data_indices = range(indptr[i], indptr[i + 1])
        item_indices = ci[data_indices]
        denom[data_indices] = item_purchase_count[
            i] + item_purchase_count[item_indices]
    union = (denom - numerator.data)
    numerator.data = numerator.data / union
    return numerator.tocsc()


def ip(mat):
    return mat.T * mat


#########################Get top K Similarity#####################


def keepTopKSimilarity(mat, k):
    if not isinstance(mat, scipy.sparse.csc_matrix):
        mat = mat.tocsc()
        mat.sort_indices()
    m, n = mat.shape
    data, ci, rptr = mat.data, mat.indices, mat.indptr
    for i in range(0, m):
        data_indices = np.arange(rptr[i], rptr[i + 1])
        sorted_indices = np.argsort(data[data_indices])[::-1]
        data[(sorted_indices + rptr[i])[k:]] = 0.0
    mat.eliminate_zeros()
    return mat


def keepTopKSimilarityDense(mat, k):
    m, n = mat.shape
    I, J, V = [], [], []
    for i in range(0, m):
        items = mat[i, :]
        item_indices = np.argsort(items)[-k:][::-1]
        scores = items[item_indices]
        J.extend(item_indices)
        V.extend(scores)
        I.extend([i] * len(scores))
    print len(I), len(J), len(V)
    R = scipy.sparse.coo_matrix(
        (V, (I, J)), shape=(m, n))
    return R.tocsr()

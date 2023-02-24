import numpy as np
from scipy.sparse import coo_matrix


def dense_matrics(edge_index, data = None):
    assert edge_index.ndim ==2
    assert edge_index.shape[0] == 2

    if data is None:
        data = np.ones(u.shape, dtype=np.int8)
    u, v = edge_index
    n = len(np.unique(u))
    m = coo_matrix((data, (u, v)), shape  = (n,n))
    return m.toarray()
import numpy as np
from scipy.sparse import coo_matrix


def dense_edge(edge_index, data = None):
    assert edge_index.ndim ==2
    assert edge_index.shape[0] == 2

    if data is None:
        data = np.ones(edge_index[0].shape, dtype=np.int8)
    else:
        data = data.squeeze() if data.ndim > 1 else data
    edge_index = edge_index.astype(int) if np.issubdtype(edge_index.dtype, np.floating) else edge_index
    u, v = edge_index
    n = np.max(edge_index) + 1
    m = coo_matrix((data, (u, v)), shape  = (n,n))
    return m.toarray()
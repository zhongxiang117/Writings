
import numpy as np


def calc_LDDT(truemap, predmap, R=15, test_set=[0.5, 1, 2, 4]):
    """
    Args:
        truemap | predmap (np.ndarray[N,N]):  distance matrix

    Reference:
        Mariani V, Biasini M, Barbato A, Schwede T.
        lDDT: a local superposition-free score for comparing protein structures and models
        using distance difference tests. Bioinformatics. 2013 Nov 1;29(21):2722-8.
        doi: 10.1093/bioinformatics/btt473.
    """
    # flatten upper triangles
    flat_truemap = truemap[np.triu_indices_from(truemap, k=1)]
    flat_predmap = predmap[np.triu_indices_from(predmap, k=1)]

    rt = flat_truemap < R
    rp = flat_predmap < R
    both = rt & rp
    true_flat_in_R = flat_truemap[both]
    pred_flat_in_R = flat_predmap[both]
    err = np.abs(true_flat_in_R - pred_flat_in_R)

    rn = both.sum()
    preserved_fractions = []
    for t in test_set:
        f = (err < t).sum() / rn
        preserved_fractions.append(f)
    lDDT = np.mean(preserved_fractions)
    return lDDT



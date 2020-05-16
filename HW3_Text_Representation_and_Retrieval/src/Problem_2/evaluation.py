import numpy as np


def read_gt(gt, size):
    y_true = np.zeros(size)
    for el in gt:
        y_true[int(el[1])-1, int(el[0])-1] = 1
    return y_true


def compute_aveP(retrieved, gt):
    """
    computes average precision for a given query
    Parameters
        ----------
        retrieved : numpy array
            vector of similarities scores between the query and the documents
        gt : numpy array,
            ground truth vector for the given query. Ones denote the position of relevant documents

    Returns
    -------
    ap_score : float
        average precision
    """

    num = 0.0
    rel = 0
    for i, r in enumerate(retrieved):
        if gt[r] == 1:
            rel += 1
            num += float(rel)/(i+1)
    if rel > 0:
        return num/rel
    else:
        return 0


def evaluate_retrieval(sim, gt, verbose):
    """
    computes mean average precision
    Parameters
        ----------
        sim : numpy array
            matrix of similarities scores between documents (rows) and queries (columns)
        gt : list,
            ground truth list. Each entry is a tuple, where the first element
            indicates a query and the second a relevant document
        verbose : if average precision score for each query must be printed

    Returns
    -------
    map_score : float
        mean average precision
    """

    y_true = read_gt(gt, sim.shape)
    avps = []
    for q in range(y_true.shape[1]):
        avp = compute_aveP(np.argsort(-sim[:, q]), y_true[:, q])
        if verbose:
            print ('Query:', q+1, 'AveP:', avp)
        avps.append(avp)
    map_score = np.mean(avps)
    print ('MAP:', map_score)
    return map_score

import numpy as np
from scipy import sparse
from collections import deque


def convert_adj_matrix(adj_matrix):
    """
    Convert an adjacency matrix to the Compressed Sparse Row type.
    :param adj_matrix: An adjacency matrix.
    :return: Compressed Sparse Row Matrix
        Adjacency matrix with the expected type.
    """
    if type(adj_matrix) == sparse.csr_matrix:
        return adj_matrix
    elif type(adj_matrix) == np.ndarray:
        return sparse.csr_matrix(adj_matrix)
    else:
        raise TypeError("The argument should be a Numpy Array or a Compressed Sparse Row Matrix.")


def approximate_ppr(adj_matrix, seed_set, alpha=0.85, epsilon=1e-3):

    """
    Compute the approximate Personalized PageRank (PPR) from a set set of seed node.
    This function implements the push method introduced by Andersen et al.
    in "Local graph partitioning using pagerank vectors", FOCS 2006.
    :param adj_matrix: compressed sparse row matrix or numpy array
        Adjacency matrix of the graph
    :param seed_set: list or set of int
        Set of seed nodes.
    :param alpha: float, default 0.85
        1 - alpha corresponds to the probability for the random walk to restarts from the seed set.
    :param epsilon: float, default 1e-3
        Precision parameter for the approximation
    :return: numpy 1D array
        Vector containing the approximate PPR for each node of the graph.
    """

    adj_matrix = convert_adj_matrix(adj_matrix)
    degree = np.array(np.sum(adj_matrix, axis=0))[0]
    n_nodes = adj_matrix.shape[0]

    prob = np.zeros(n_nodes)
    res = np.zeros(n_nodes)
    res[list(seed_set)] = 1. / len(seed_set)

    next_nodes = deque(seed_set)

    while len(next_nodes) > 0:
        node = next_nodes.pop()
        push_val = res[node] - 0.5 * epsilon * degree[node]
        res[node] = 0.5 * epsilon * degree[node]
        prob[node] += (1. - alpha) * push_val
        put_val = alpha * push_val
        for neighbor in adj_matrix[node].indices:
            old_res = res[neighbor]
            res[neighbor] += put_val * adj_matrix[node, neighbor] / degree[node]
            threshold = epsilon * degree[neighbor]
            if res[neighbor] >= threshold > old_res:
                next_nodes.appendleft(neighbor)
    return prob


def conductance_sweep_cut(adj_matrix, score, window=10):
    """
    Return the sweep cut for conductance based on a given score.
    During the sweep process, we detect a local minimum of conductance using a given window.
    The sweep process is described by Andersen et al. in
    "Communities from seed sets", 2006.
    :param adj_matrix: compressed sparse row matrix or numpy array
        Adjacency matrix of the graph.
    :param score: numpy vector
        Score used to order the nodes in the sweep process.
    :param window: int, default 10
        Window parameter used for the detection of a local minimum of conductance.
    :return: set of int
         Set of nodes corresponding to the sweep cut.
    """
    adj_matrix = convert_adj_matrix(adj_matrix)

    n_nodes = adj_matrix.shape[0]
    degree = np.array(np.sum(adj_matrix, axis=0))[0]

    total_volume = np.sum(degree)
   
    sorted_nodes = [node for node in range(n_nodes) if score[node] > 0]
    sorted_nodes = sorted(sorted_nodes, key=lambda node: score[node], reverse=True)
    
    sweep_set = set()
    volume = 0.0
    cut = 0.0
    best_conductance = 1.
    best_sweep_set = {sorted_nodes[0]}
    inc_count = 0      
        
    for node in sorted_nodes:
        volume += degree[node]
     
        for neighbor in adj_matrix[node].indices:
            if neighbor in sweep_set:
                cut -= 1
            else: 
                cut += 1 
     
        sweep_set.add(node)
       
        conductance = cut / min(volume, total_volume - volume)
    
        if conductance < best_conductance:
            best_conductance = conductance
            # Make a copy of the set
            best_sweep_set = set(sweep_set)
            
            inc_count = 0
        else:
            inc_count += 1
            if inc_count >= window:
                break
    return best_sweep_set


def multicom(adj_matrix, seedset, scoring, cut, explored_ratio=0.8, one_community=True):
    """
    :param adj_matrix: compressed sparse row matrix or numpy 2D array
        Adjacency matrix of the graph.
    :param seedset: int (change)
        Id of the seed nodes around which we want to detect communities.
    :param scoring: function
        Function (adj_matrix: numpy 2D array, seed_set: list or set of int) -> score: numpy 1D array.
        Example: approximate_ppr
    :param cut: function
        Function (adj_matrix: numpy 2D array, score: numpy 1D array) -> sweep cut: set of int.
        Example: conductance_sweep_cut
    :param explored_ratio: float, default 0.8
        Parameter used to control the number of new seeds at each step.
    :return:
    seeds: list of int
        Seeds used to detect communities around the initial seed (including this original seed).
    communities: list of set
        Communities detected around the seed node.
    """
   
    scores = []
    communities = list()

    if (one_community):     
        scores = scoring(adj_matrix, seedset)
        community = cut(adj_matrix, scores)
        communities.append(community)
    else:     
        for seed in seedset:
            scores = scoring(adj_matrix, [seed])
            community = cut(adj_matrix, scores)
            communities.append(community)
    return seedset, communities







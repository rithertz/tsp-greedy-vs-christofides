import networkx as nx

def mst_lower_bound(dist_matrix):
    """
    Compute MST cost as a lower bound for TSP.
    """
    n = len(dist_matrix)
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    mst = nx.minimum_spanning_tree(G, weight='weight')
    return mst.size(weight='weight')
import networkx as nx


def christofides_tsp(dist_matrix):
    """
    Christofides approximation algorithm for metric TSP.
    Returns (tour, length).
    """
    n = len(dist_matrix)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0, 0], 0.0

    # Build complete graph.
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=dist_matrix[i, j])

    # 1. Minimum spanning tree.
    mst = nx.minimum_spanning_tree(graph, weight="weight")

    # 2. Vertices with odd degree in the MST.
    odd_vertices = [vertex for vertex in mst.nodes if mst.degree(vertex) % 2 == 1]

    # 3. Minimum weight perfect matching on the odd-degree vertices.
    odd_graph = nx.Graph()
    odd_graph.add_nodes_from(odd_vertices)
    for index, u in enumerate(odd_vertices):
        for v in odd_vertices[index + 1 :]:
            odd_graph.add_edge(u, v, weight=dist_matrix[u, v])

    try:
        matching = nx.min_weight_matching(odd_graph, weight="weight")
    except AttributeError:
        # Fallback for older NetworkX versions.
        for _, _, edge_data in odd_graph.edges(data=True):
            edge_data["weight"] = -edge_data["weight"]
        matching = nx.max_weight_matching(
            odd_graph, weight="weight", maxcardinality=True
        )

    # 4. Combine MST and matching edges into a multigraph.
    multigraph = nx.MultiGraph()
    multigraph.add_nodes_from(range(n))
    for u, v in mst.edges():
        multigraph.add_edge(u, v, weight=dist_matrix[u, v])
    for u, v in matching:
        multigraph.add_edge(u, v, weight=dist_matrix[u, v])

    # 5. Find an Eulerian circuit.
    euler_circuit = list(nx.eulerian_circuit(multigraph, source=0))

    # 6. Shortcut repeated visits to obtain a Hamiltonian tour.
    tour = []
    visited = set()
    for u, _ in euler_circuit:
        if u not in visited:
            tour.append(u)
            visited.add(u)
    tour.append(tour[0])  # return to start

    length = 0.0
    for i in range(len(tour) - 1):
        length += dist_matrix[tour[i], tour[i + 1]]
    return tour, length

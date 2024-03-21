import networkx as nx
import scipy


def build_graph_mst_scipy(MST):
    '''
    Build nx graph from scipy MST
    '''
    if not isinstance(MST, scipy.sparse.coo_matrix):
        MST = MST.tocoo()
    G = nx.Graph()
    n_nodes = len(MST.data) + 1
    G.add_nodes_from(range(n_nodes))
    rows = MST.row
    columns = MST.col
    data = MST.data
    edges = [0]*len(rows)
    for i in range(len(rows)):
        edges[i] = ((rows[i],columns[i]),data[i])
    G.add_edges_from(edges)

    return G


def building_graph_mst_mlpack(MST):
    '''
    Build nx graph from mlpack emst
    '''
    G = nx.Graph()
    n_nodes = len(MST) + 1
    G.add_nodes_from(range(n_nodes))

    edges = [0]*len(MST)
    for i in range(len(MST)):
        edges[i] = ((int(MST[i,0]),int(MST[i,1])),MST[i,2])
    G.add_edges_from(edges)

    return G


import networkx as nx
import scipy


def build_graph_mst(MST):
    '''
    Build nx graph from emst
    '''
    G = nx.Graph()
    n_nodes = len(MST) + 1
    G.add_nodes_from(range(n_nodes))

    edges = [0]*len(MST)
    for i in range(len(MST)):
        edges[i] = (int(MST[i,0]),int(MST[i,1]),MST[i,2])
    G.add_weighted_edges_from(edges)

    return G

def compute_difference_graphs(G1,G2):
    '''
    Compute statistics about two graphs MST graphs G1 and G2
    '''
    n_equal_edges = 0
    w_equal_edges = 0
    for edge in G1.edges.data("weight"):
        if(edge[1] in G2[edge[0]]):
            n_equal_edges+=1
            w_equal_edges += edge[2]
    return n_equal_edges, w_equal_edges
    
        


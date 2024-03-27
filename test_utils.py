import os, errno
import networkx as nx



def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


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
    


def plot_persitance_mst_and_dist(mst_ANN, mst_MLpack,dataset_name):
    import gudhi
    import matplotlib.pyplot as plt
    n_edges = len(mst_ANN)
    sparse_mst_st = gudhi.SimplexTree()
    for i in range(n_edges+1):
        sparse_mst_st.insert([i])
    for i in range(n_edges):
        sparse_mst_st.insert([int(mst_ANN[i,0]),int(mst_ANN[i,1])], filtration = mst_ANN[i,2])

    barcodes = sparse_mst_st.persistence(min_persistence=-1)

    I_mst = sparse_mst_st.persistence_intervals_in_dimension(0)

    mlpack_mst_st = gudhi.SimplexTree()
    for i in range(n_edges+1):
        mlpack_mst_st.insert([i])
    for i in range(n_edges):
        mlpack_mst_st.insert([int(mst_MLpack[i,0]),int(mst_MLpack[i,1])], filtration = mst_MLpack[i,2])

    

    barcodes_mlpack = mlpack_mst_st.persistence(min_persistence=-1)

    I_mlpack = mlpack_mst_st.persistence_intervals_in_dimension(0)

    dist = gudhi.bottleneck_distance(I_mst, I_mlpack)


    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    gudhi.plot_persistence_diagram(barcodes_mlpack,axes = axs[0])
    axs[0].set_title('Persistance - EMST')

    gudhi.plot_persistence_diagram(barcodes, axes = axs[1])

    axs[1].set_title('Persistance - Vamana mst')

    
    fig.suptitle(f'Persistence diagrams of MSTs - {dataset_name} dataset')
    fig.tight_layout()
    plt.show()
    

    return dist

def MST_test_topology(data_path, Index_path,dataset_name, save_msts = False):
    from TopoMap import TopoMap
    from dataset_utils import fvecs_read, Index
    from utils import compute_mst_mlpack, compute_mst_ann
    import numpy as np



    print("--")
    points  = fvecs_read(data_path)
    index = Index(Index_path)
    mst_ANN, ANN_building_time = compute_mst_ann(points,index, compute_time= True)
     
    mst_ANN_nx = build_graph_mst(mst_ANN)
    
    
    
    mst_mlpack, mlpack_building_time =  compute_mst_mlpack(points, compute_time = True)
    
    mst_mlpack_nx = build_graph_mst(mst_mlpack)

    bn_dist = plot_persitance_mst_and_dist(mst_ANN, mst_mlpack,dataset_name)
    n_equal_edges, w_equal_edges =compute_difference_graphs(mst_ANN_nx,mst_mlpack_nx)
    n_edges = len(mst_ANN)
    total_weight = mst_mlpack[:,2].sum()
    approx_weight = mst_ANN[:,2].sum()
    percentual_weight_error = abs(total_weight-approx_weight)/total_weight

    if(save_msts):
        mst_filename = f'./msts/mst_{dataset_name}.npy'
        vamana_mst_filename = f'./msts/Vamana_mst_{dataset_name}.npy'
        silentremove(mst_filename)
        silentremove(vamana_mst_filename)
        np.save(mst_filename,mst_mlpack, allow_pickle=True)
        np.save(vamana_mst_filename,mst_ANN, allow_pickle= True)
        
    

    return [mlpack_building_time,ANN_building_time,bn_dist,n_equal_edges,n_equal_edges/n_edges,w_equal_edges,w_equal_edges/total_weight, percentual_weight_error]




        


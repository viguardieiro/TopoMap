import numpy as np 

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import DisjointSet

import networkx as nx

from utils import get_hull, closest_edge_point, find_angle, Transform, fix_rotation

import matplotlib.pyplot as plt

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from TopoMap import TopoMap

class TopoMapCut(TopoMap):
    def __init__(self, points:np.ndarray,
                 metric='euclidean') -> None:
        self.points = points
        self.n = len(points)
        self.metric = metric

        self.mst = self.get_mst()
        self.sorted_edges = self.get_sorted_edges()

        self.projections = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.components = DisjointSet(list(range(self.n)))
        self.subsets = None
        self.points_component = None
        self.components_points = None

        self.proj_subsets = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.n_components_non_single = []

    def get_component_of_points(self):
        if self.subsets is None:
            self.subsets = self.components.subsets()
        self.points_component = np.zeros(self.n)

        for i in range(self.n):
            for j in range(len(self.subsets)):
                if i in self.subsets[j]:
                    self.points_component[i] = j

        return self.points_component
    
    def get_points_of_components(self):
        if self.subsets is None:
            self.subsets = self.components.subsets()
        self.components_points = []

        for j in range(len(self.subsets)):
            self.components_points.append([])
            for i in self.subsets[j]:
                self.components_points[-1].append(self.points[i,:])
            self.components_points[-1] = np.array(self.components_points[-1])

        return self.components_points
    
    def num_components_non_single(self):
        n_comp = 0
        comps = self.components.subsets()
        for j in range(len(comps)):
            if len(comps[j]) > 1:
                n_comp += 1
        return n_comp
    
    def get_components(self, max_components=-1, 
                       max_comp_non_single=-1,
                       min_dist=-1):
        
        if max_comp_non_single != -1:
            flag_comp = False

        for i in range(len(self.sorted_edges)):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']

            if min_dist!=-1 and d >= min_dist:
                print(f'[INFO] Min distance hit. Distance: {d} | Min_dist: {min_dist}')
                break
            if max_components!=-1 and len(self.components.subsets()) <= max_components:
                print(f'[INFO] Max components hit. # components: {len(self.components.subsets())} | Max_components: {max_components}')
                break
            if max_comp_non_single!=-1:
                if not flag_comp:
                    if len(self.n_components_non_single)>1 and self.n_components_non_single[-1] >= max_comp_non_single:
                        flag_comp = True
                else:
                    if self.n_components_non_single[-1] <= max_comp_non_single:
                        print(f'[INFO] Max components non single hit. # components non single: {self.n_components_non_single[-1]}')
                        break
            
            # Merge components 
            self.components.merge(i_a, i_b)

            self.n_components_non_single.append(self.num_components_non_single())

        if i == len(self.sorted_edges)-1:
            print(f'[INFO] Number of edges hit. Edges processed: {i}')

        self.last_processed_edge = i
        self.subsets = self.components.subsets()
        self.points_component = self.get_component_of_points()
        self.components_points = self.get_points_of_components()

        return self.components
    
    def project_components(self, proj_method='tsne', perplexity=30, rescale=True):
        self.proj_subsets = []

        if self.subsets is None:
            self.subsets = self.components.subsets()

        for j in range(len(self.subsets)):
            self.proj_subsets.append([])

            if proj_method=='tsne' and len(self.subsets[j]) > perplexity:
                proj = TSNE(n_components=2, perplexity=perplexity)
                self.proj_subsets[-1] = proj.fit_transform(self.components_points[j])

            elif proj_method=='umap':
                proj = umap.UMAP(n_components=2)
                self.proj_subsets[-1] = proj.fit_transform(self.components_points[j])

            elif len(self.subsets[j]) >= 2 or (proj_method=='pca' and len(self.subsets[j]) >= 2):
                proj = PCA(n_components=2)
                self.proj_subsets[-1] = proj.fit_transform(self.components_points[j])
                
            else:
                self.proj_subsets[-1] = np.array([[0,0]])

            if rescale and (len(self.subsets[j]) >= 2):
                scale_i = 0
                orig_scale = self.components_points[j][:,scale_i].max() - self.components_points[j][:,scale_i].min()

                if orig_scale > 0:
                    proj_scale = self.proj_subsets[-1][:,scale_i].max() - self.proj_subsets[-1][:,scale_i].min()
                    self.proj_subsets[-1] = (orig_scale/proj_scale)*self.proj_subsets[-1]

                else:
                    scale_i = 1
                    orig_scale = self.components_points[j][:,scale_i].max() - self.components_points[j][:,scale_i].min()
                    if orig_scale > 0:
                        proj_scale = self.proj_subsets[-1][:,scale_i].max() - self.proj_subsets[-1][:,scale_i].min()
                        self.proj_subsets[-1] = (orig_scale/proj_scale)*self.proj_subsets[-1]

            self.projections[list(self.subsets[j]), :] = self.proj_subsets[-1]

        return self.proj_subsets
    
    def join_components(self):
        for i in range(self.last_processed_edge, min(self.n, len(self.sorted_edges))):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']
            
            # Get components the points belong to
            c_a, c_b = self.components.subset(i_a), self.components.subset(i_b)
            proj_c_a, proj_c_b = self.projections[list(c_a)], self.projections[list(c_b)]

            # Rotate the first to be the topmost
            proj_c_a, edge_t = self.rotate_component(proj_c_a, p_a, direction='top')
            proj_c_b, edge_b = self.rotate_component(proj_c_b, p_b, direction='bottom')

            # Rotate the second to be the bottomost
            proj_c_a = self.translate_component(proj_c_a, edge_t, to_point=[0,0])
            proj_c_b = self.translate_component(proj_c_b, edge_b, to_point=[0,d])

            # Merge components 
            self.components.merge(i_a, i_b)

            self.projections[list(c_a), :] = proj_c_a
            self.projections[list(c_b), :] = proj_c_b

        return self.projections

    def run_iter(self, iter=1):
        fig, ax = plt.subplots(1,3, figsize=(12,4))

        for i in range(self.last_processed_edge, self.last_processed_edge+iter):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']
            
            # Get components the points belong to
            c_a, c_b = self.components.subset(i_a), self.components.subset(i_b)
            proj_c_a, proj_c_b = self.projections[list(c_a)], self.projections[list(c_b)]
            i_a_comp, i_b_comp = list(c_a).index(i_a), list(c_b).index(i_b)

            if i==self.last_processed_edge+iter-1:
                print(f'Number of points in A: {len(proj_c_a)}')
                print(f'Number of points in B: {len(proj_c_b)}')
                print(f'Distance: {d}')
                ax[0].scatter(self.projections[:,0], self.projections[:,1], s=5, c='black', 
                              alpha=0.1, linewidths=0)

                ax[0].scatter(proj_c_a[:,0], proj_c_a[:,1], c='red', label='Comp A', s=5, 
                              alpha=0.1, linewidths=0)
                ax[0].scatter(p_a[0], p_a[1], marker='^', c='yellow')

                ax[0].scatter(proj_c_b[:,0], proj_c_b[:,1], c='green', label='Comp B', s=5, 
                              alpha=0.1, linewidths=0)
                ax[0].scatter(p_b[0], p_b[1], marker='^', c='blue')

                ax[0].set_title('Proj before iteration')

            # Rotate the first to be the topmost
            proj_c_a, edge_t = self.rotate_component(proj_c_a, p_a, direction='top')
            # Rotate the second to be the bottomost
            proj_c_b, edge_b = self.rotate_component(proj_c_b, p_b, direction='bottom')

            if i==self.last_processed_edge+iter-1:
                ax[1].scatter(self.projections[:,0], self.projections[:,1], s=5, c='black', 
                              alpha=0.1, linewidths=0)
                
                ax[1].scatter(proj_c_a[:,0], proj_c_a[:,1], c='red', label='Comp A', s=5, 
                              alpha=0.1, linewidths=0)
                ax[1].scatter(proj_c_a[i_a_comp,0], proj_c_a[i_a_comp,1], marker='^', c='yellow')
                ax[1].plot([proj_c_a[edge_t[0],0], proj_c_a[edge_t[1],0]],
                           [proj_c_a[edge_t[0],1], proj_c_a[edge_t[1],1]],
                            color='red', linewidth=1)

                ax[1].scatter(proj_c_b[:,0], proj_c_b[:,1], c='green', label='Comp B', s=5, 
                              alpha=0.1, linewidths=0)
                ax[1].scatter(proj_c_b[i_b_comp,0], proj_c_b[i_b_comp,1], marker='^', c='blue')
                ax[1].plot([proj_c_b[edge_b[0],0], proj_c_b[edge_b[1],0]],
                           [proj_c_b[edge_b[0],1], proj_c_b[edge_b[1],1]],
                            color='blue', linewidth=1)

                ax[1].set_title('Proj after rotation')

            proj_c_a = self.translate_component(proj_c_a, edge_t, to_point=[0,0])
            proj_c_b = self.translate_component(proj_c_b, edge_b, to_point=[0,d])

            if i==self.last_processed_edge+iter-1:
                ax[2].scatter(self.projections[:,0], self.projections[:,1], s=5, c='black', 
                              alpha=0.1, linewidths=0)
                
                ax[2].scatter(proj_c_a[:,0], proj_c_a[:,1], c='red', label='Comp A', s=5, 
                              alpha=0.1, linewidths=0)
                ax[2].scatter(proj_c_a[i_a_comp,0], proj_c_a[i_a_comp,1], marker='^', c='yellow')
                ax[2].plot([proj_c_a[edge_t[0],0], proj_c_a[edge_t[1],0]],
                           [proj_c_a[edge_t[0],1], proj_c_a[edge_t[1],1]],
                            color='red', linewidth=1)

                ax[2].scatter(proj_c_b[:,0], proj_c_b[:,1], c='green', label='Comp B', s=5, 
                              alpha=0.1, linewidths=0)
                ax[2].scatter(proj_c_b[i_b_comp,0], proj_c_b[i_b_comp,1], marker='^', c='blue')
                ax[2].plot([proj_c_b[edge_b[0],0], proj_c_b[edge_b[1],0]],
                           [proj_c_b[edge_b[0],1], proj_c_b[edge_b[1],1]],
                            color='blue', linewidth=1)

                ax[2].set_title('Proj after translation')

            # Merge components 
            self.components.merge(i_a, i_b)

            self.projections[list(c_a), :] = proj_c_a
            self.projections[list(c_b), :] = proj_c_b

        return self.projections, self.components
    
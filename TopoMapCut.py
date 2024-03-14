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
                 metric='euclidean',
                 max_components=-1,
                 max_dist=-1,
                 min_points_comp=2,
                 proj_method='tsne',
                 ignore_outliers=True) -> None:
        self.points = points
        self.n = len(points)
        self.metric = metric

        self.max_components = max_components
        self.max_dist = max_dist
        self.min_points_comp = min_points_comp
        self.proj_method = proj_method
        self.ignore_outliers = ignore_outliers

        self.mst = self.get_mst()
        self.sorted_edges = self.get_sorted_edges()

        self.projections = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.components = DisjointSet(list(range(self.n)))
        self.subsets = None
        self.points_component = None
        self.components_points = None
        self.outlier_comps_i = []
        self.outlier_comps = []

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
                       max_dist=-1):

        for i in range(len(self.sorted_edges)):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']

            if max_dist!=-1 and d >= max_dist:
                print(f'[INFO] Min distance hit. Distance: {d} | Max_dist: {max_dist}')
                break
            if max_components!=-1 and len(self.components.subsets()) <= max_components:
                print(f'[INFO] Max components hit. # components: {len(self.components.subsets())} | Max_components: {max_components}')
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
    
    def identify_outlier_components(self, min_points_comp=2):
        self.outlier_comps = []
        self.outlier_comps_i = []

        for j in range(len(self.subsets)):
            if len(self.subsets[j]) < min_points_comp:
                self.outlier_comps_i.append(j)
                self.outlier_comps.append(self.subsets[j])

        return self.outlier_comps
    
    def project_components(self, 
                           proj_method='tsne', 
                           perplexity=30):
        self.proj_subsets = []

        if self.subsets is None:
            self.subsets = self.components.subsets()

        for j in range(len(self.subsets)):

            if j in self.outlier_comps:
                continue

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

            if c_a in self.outlier_comps or c_b in self.outlier_comps:
                continue

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
    
    def run(self):
        self.get_components(max_components=self.max_components,
                            max_dist=self.max_dist)
        
        if self.ignore_outliers:
            self.identify_outlier_components(min_points_comp=self.min_points_comp)
        else:
            self.outlier_comps = []
            self.outlier_comps_i = []

        self.project_components(proj_method=self.proj_method)
        
        self.join_components()

        return self.projections
    
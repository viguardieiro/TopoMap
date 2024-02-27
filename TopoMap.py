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

class TopoMap():
    def __init__(self, points:np.ndarray) -> None:
        self.points = points
        self.n = len(points)

        self.mst = self.get_mst()
        self.sorted_edges = self.get_sorted_edges()

        self.projections = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.components = DisjointSet(list(range(self.n)))

    def get_mst(self):
        dists = squareform(pdist(self.points))
        self.mst = minimum_spanning_tree(dists).toarray()
        return self.mst
    
    def get_sorted_edges(self):
        G = nx.from_numpy_array(self.mst)
        self.sorted_edges = sorted(G.edges(data=True), 
                                   key=lambda edge: edge[2].get('weight', 1))
        return self.sorted_edges
    
    def rotate_component(self, 
                         component_points:np.ndarray, 
                         ref_point:np.ndarray, 
                         direction='top') -> np.ndarray:
        
        if len(component_points) == 1:
            return component_points, [0,0]
        
        hull = get_hull(component_points)

        closest_edge, edge_i = closest_edge_point(hull, ref_point)
        
        t = Transform()
        t.cos, t.sin = find_angle(closest_edge)
        component_points = t.rotate(component_points)

        component_points = fix_rotation(component_points[edge_i], 
                                        ref_point, 
                                        component_points, 
                                        direction=direction)

        return component_points, edge_i
    
    def translate_component(self, 
                            component_points:np.ndarray, 
                            edge_i:np.ndarray,
                            to_point:list) -> np.ndarray:
        
        if component_points[edge_i[0], 0] <= component_points[edge_i[1], 0]:
            t = Transform(x = to_point[0]-component_points[edge_i[0], 0], 
                          y = to_point[1]-component_points[edge_i[0], 1])
            component_points = t.translate(component_points)

        else:
            t = Transform(x = to_point[0]-component_points[edge_i[1], 0], 
                          y = to_point[1]-component_points[edge_i[1], 1])
            component_points = t.translate(component_points)

        return component_points
    
    def run_iter(self, iter=5):
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        for i in range(iter):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']
            
            # Get components the points belong to
            c_a, c_b = self.components.subset(i_a), self.components.subset(i_b)
            proj_c_a, proj_c_b = self.projections[list(c_a)], self.projections[list(c_b)]

            if i==iter-1:
                print(f'Number of points in A: {len(proj_c_a)}')
                print(f'Number of points in B: {len(proj_c_b)}')
                ax[0].scatter(self.projections[:,0], self.projections[:,1], s=5, c='black')

                ax[0].scatter(proj_c_a[:,0], proj_c_a[:,1], c='red', label='Comp A', s=5)
                for xi, yi, pidi in zip(proj_c_a[:,0],proj_c_a[:,1],list(range(len(proj_c_a)))):
                    ax[0].annotate(str(pidi), xy=(xi,yi), color='red')
                ax[0].scatter(p_a[0], p_a[1], marker='^', c='r')

                ax[0].scatter(proj_c_b[:,0], proj_c_b[:,1], c='green', label='Comp B', s=5)
                for xi, yi, pidi in zip(proj_c_b[:,0],proj_c_b[:,1],list(range(len(proj_c_b)))):
                    ax[0].annotate(str(pidi), xy=(xi,yi), color='green')
                ax[0].scatter(p_b[0], p_b[1], marker='^', c='g')

                ax[0].set_title('Proj before iteration')

            # Rotate the first to be the topmost
            proj_c_a, edge_t = self.rotate_component(proj_c_a, p_a, direction='top')
            proj_c_b, edge_b = self.rotate_component(proj_c_b, p_b, direction='bottom')

            if i==iter-1:
                ax[1].scatter(self.projections[:,0], self.projections[:,1], s=5, c='black')
                ax[1].scatter(proj_c_a[:,0], proj_c_a[:,1], c='red', label='Comp A', s=5)
                for xi, yi, pidi in zip(proj_c_a[:,0],proj_c_a[:,1],list(range(len(proj_c_a)))):
                    ax[1].annotate(str(pidi), xy=(xi,yi), color='red')

                ax[1].scatter(proj_c_b[:,0], proj_c_b[:,1], c='green', label='Comp B', s=5)
                for xi, yi, pidi in zip(proj_c_b[:,0],proj_c_b[:,1],list(range(len(proj_c_b)))):
                    ax[1].annotate(str(pidi), xy=(xi,yi), color='green')

                ax[1].set_title('Proj after rotation')

            # Rotate the second to be the bottomost
            proj_c_a = self.translate_component(proj_c_a, edge_t, to_point=[0,0])
            proj_c_b = self.translate_component(proj_c_b, edge_b, to_point=[0,d])

            if i==iter-1:
                ax[2].scatter(self.projections[:,0], self.projections[:,1], s=5, c='black')
                ax[2].scatter(proj_c_a[:,0], proj_c_a[:,1], c='red', label='Comp A', s=5)
                for xi, yi, pidi in zip(proj_c_a[:,0],proj_c_a[:,1],list(range(len(proj_c_a)))):
                    ax[2].annotate(str(pidi), xy=(xi,yi), color='red')

                ax[2].scatter(proj_c_b[:,0], proj_c_b[:,1], c='green', label='Comp B', s=5)
                for xi, yi, pidi in zip(proj_c_b[:,0],proj_c_b[:,1],list(range(len(proj_c_b)))):
                    ax[2].annotate(str(pidi), xy=(xi,yi), color='green')

                ax[2].set_title('Proj after translation')

            # Merge components 
            self.components.merge(i_a, i_b)

            self.projections[list(c_a), :] = proj_c_a
            self.projections[list(c_b), :] = proj_c_b

        return self.projections, self.components
    
    def run(self):
        for i in range(min(self.n, len(self.sorted_edges))):
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

#####################################################
#####################################################

class TopoMapCut(TopoMap):
    def __init__(self, points:np.ndarray) -> None:
        self.points = points
        self.n = len(points)

        self.mst = self.get_mst()
        self.sorted_edges = self.get_sorted_edges()

        self.projections = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.components = DisjointSet(list(range(self.n)))
        self.subsets = None
        self.points_component = None
        self.components_points = None

        self.proj_subsets = np.zeros(shape=(self.n, 2), dtype=np.float32)

        self.last_processed_edge = -1
    
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
    
    def get_components(self, max_components=-1, 
                       min_dist=-1):
        for i in range(min(self.n, len(self.sorted_edges))):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']

            if min_dist!=-1 and d >= min_dist:
                break
            if max_components!=-1 and len(self.components.subsets()) <= max_components:
                break
            
            # Merge components 
            self.components.merge(i_a, i_b)

        self.last_processed_edge = i
        self.subsets = self.components.subsets()
        self.points_component = self.get_component_of_points()
        self.components_points = self.get_points_of_components()

        return self.components
    
    def project_components(self, proj_method='tsne', perplexity=30):
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

            self.projections[list(self.subsets[j]), :] = self.proj_subsets[-1]

        return self.proj_subsets
    
    def rotate_component(self, 
                         component_points:np.ndarray, 
                         ref_point:np.ndarray, 
                         ref_point_i:int,
                         direction='top') -> np.ndarray:
        
        if len(component_points) == 1:
            return component_points, [0,0]
        
        hull = get_hull(component_points)

        closest_edge, edge_i = closest_edge_point(hull, ref_point)
        
        t = Transform()
        t.cos, t.sin = find_angle(closest_edge)
        component_points = t.rotate(component_points)

        component_points = fix_rotation(component_points[edge_i], 
                                        component_points[ref_point_i], 
                                        component_points, 
                                        direction=direction)

        return component_points, edge_i
    
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
            i_a_comp, i_b_comp = list(c_a).index(i_a), list(c_b).index(i_b)

            # Rotate the first to be the topmost
            proj_c_a, edge_t = self.rotate_component(proj_c_a, p_a, i_a_comp, direction='top')
            proj_c_b, edge_b = self.rotate_component(proj_c_b, p_b, i_b_comp, direction='bottom')

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
            proj_c_a, edge_t = self.rotate_component(proj_c_a, p_a, i_a_comp, direction='top')
            # Rotate the second to be the bottomost
            proj_c_b, edge_b = self.rotate_component(proj_c_b, p_b, i_b_comp, direction='bottom')

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
import numpy as np 

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import DisjointSet

import networkx as nx

from utils import closest_edge_point, find_angle, Transform, fix_rotation

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
        
        hull = ConvexHull(component_points)

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
            t = Transform(x = to_point[0]+component_points[edge_i[1], 0], 
                          y = to_point[1]+component_points[edge_i[1], 1])
            component_points = t.translate(component_points)

        else:
            t = Transform(x = to_point[0]+component_points[edge_i[0], 0], 
                          y = to_point[1]+component_points[edge_i[0], 1])
            component_points = t.translate(component_points)

        return component_points
    
    def run(self):
        for i in range(self.n):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # p_b should be the rightmost point
            if p_b[0] <= p_a[0]:
                i_a, i_b = i_b, i_a
                p_a, p_b = p_b, p_a

            # Distance between points
            d = self.sorted_edges[i][2]['weight']
            
            # Components the points belong to
            c_a, c_b = self.components.subset(i_a), self.components.subset(i_b)
            proj_c_a, proj_c_b = self.projections[list(c_a)], self.projections[list(c_b)]

            proj_c_a, edge_t = self.rotate_component(proj_c_a, p_a, direction='top')
            proj_c_b, edge_b = self.rotate_component(proj_c_b, p_b, direction='bottom')

            proj_c_a = self.translate_component()

            # Merge components 
            self.components.merge(i_a, i_b)

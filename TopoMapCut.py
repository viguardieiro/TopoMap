import numpy as np 

from scipy.cluster.hierarchy import DisjointSet

from utils import Transform

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
    

class TopoMapCutInv(TopoMap):
    def __init__(self, points:np.ndarray,
                 metric='euclidean',
                 max_components=-1,
                 max_dist=-1,
                 proj_method='tsne') -> None:
        self.points = points
        self.n = len(points)
        self.metric = metric

        self.max_components = max_components
        self.max_dist = max_dist
        self.proj_method = proj_method

        self.mst = self.get_mst()
        self.sorted_edges = self.get_sorted_edges()

        self.projections = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.components = DisjointSet(list(range(self.n)))
        self.subsets = None
        self.points_component = None
        self.components_points = None
        self.components_center = None
        self.components_range_top = None
        self.components_range_bottom = None
        self.components_range_left = None
        self.components_range_right = None
        self.components_proj = None

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
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']

            if max_dist!=-1 and d >= max_dist:
                print(f'[INFO] Min distance hit. Distance: {d} | Max_dist: {max_dist}')
                break
            if max_components!=-1 and len(self.components.subsets()) <= max_components:
                print(f'[INFO] Max components hit. # components: {len(self.components.subsets())} | Max_components: {max_components}')
                break

            # Get components the points belong to
            c_a, c_b = self.components.subset(i_a), self.components.subset(i_b)
            proj_c_a, proj_c_b = self.projections[list(c_a)], self.projections[list(c_b)]

            # Rotate the first to be the topmost
            proj_c_a, edge_t = self.rotate_component(proj_c_a, p_a, direction='top')
            # Rotate the second to be the bottomost
            proj_c_b, edge_b = self.rotate_component(proj_c_b, p_b, direction='bottom')

            # Translate components
            proj_c_a = self.translate_component(proj_c_a, edge_t, to_point=[0,0])
            proj_c_b = self.translate_component(proj_c_b, edge_b, to_point=[0,d])
            
            # Merge components 
            self.components.merge(i_a, i_b)

            self.projections[list(c_a), :] = proj_c_a
            self.projections[list(c_b), :] = proj_c_b

        if i == len(self.sorted_edges)-1:
            print(f'[INFO] Number of edges hit. Edges processed: {i}')

        self.last_processed_edge = i
        self.subsets = self.components.subsets()
        self.points_component = self.get_component_of_points()
        self.components_points = self.get_points_of_components()

        return self.components
    
    def get_component_center(self, center_type='average'):
        self.components_center = []

        for j in range(len(self.subsets)):
            if center_type == 'average':
                center = np.average(self.components_points[j], axis=0)
            
            self.components_center.append(center)

        return self.components_center
    
    def get_component_ranges(self):
        self.components_range_top = []
        self.components_range_bottom = []
        self.components_range_left = []
        self.components_range_right = []

        for j in range(len(self.subsets)):
            comp_ids = list(self.subsets[j])

            new_center = np.average(self.projections[comp_ids, :], axis=0)

            range_top = self.projections[comp_ids, 1].max() - new_center[1]
            range_bottom = new_center[1] - self.projections[comp_ids, 1].min()
            range_left = new_center[0] - self.projections[comp_ids, 0].min()
            range_right = self.projections[comp_ids, 0].max() - new_center[0]
            
            self.components_range_top.append(range_top)
            self.components_range_bottom.append(range_bottom)
            self.components_range_left.append(range_left)
            self.components_range_right.append(range_right)

        return self.components_range_top, self.components_range_bottom, self.components_range_left, self.components_range_right
    
    def project_components_center(self, 
                                  proj_method='tsne', 
                                  perplexity=30):
        components_center = np.array(self.components_center)
        self.components_proj = []

        if proj_method=='tsne':
            proj = TSNE(n_components=2, perplexity=perplexity)
            self.components_proj = proj.fit_transform(components_center)

        elif proj_method=='umap':
            proj = umap.UMAP(n_components=2)
            self.components_proj = proj.fit_transform(components_center)

        elif proj_method=='pca':
            proj = PCA(n_components=2)
            self.components_proj = proj.fit_transform(components_center)

        return self.components_proj
    
    def join_components(self):
        
        for j in range(len(self.subsets)):
            # Translate projections of this component to match the center's projection
            comp_ids = list(self.subsets[j])

            proj_center_x = np.mean(self.projections[comp_ids,0])
            proj_center_y = np.mean(self.projections[comp_ids,1])

            comp_center = self.components_center[j]

            t_proj = Transform(x=comp_center[0]-proj_center_x,
                               y=comp_center[1]-proj_center_y)
            self.projections[comp_ids,:] = t_proj.translate(self.projections[comp_ids,:])

            # Translate other centers and projections
            range_top = self.components_range_top[j]
            range_bottom = self.components_range_bottom[j]
            range_left = self.components_range_left[j]
            range_right = self.components_range_right[j]

            t_up = Transform(y=range_top)
            t_down = Transform(y=-range_bottom)
            t_left = Transform(x=-range_left)
            t_right = Transform(x=range_right)

            for k in range(len(self.subsets)):
                if k==j:
                    continue

                comp_k_ids = list(self.subsets[k])

                # k left of j
                if self.components_proj[k][0] < self.components_proj[j][0]:
                    self.projections[comp_k_ids,:] = t_left.translate(self.projections[comp_k_ids,:])
                    self.components_proj[k] = t_left.translate(self.components_proj[k])

                # k right of j
                elif self.components_proj[k][0] >= self.components_proj[j][0]:
                    self.projections[comp_k_ids,:] = t_right.translate(self.projections[comp_k_ids,:])
                    self.components_proj[k] = t_right.translate(self.components_proj[k])

                # k above j
                if self.components_proj[k][1] >= self.components_proj[j][1]:
                    self.projections[comp_k_ids,:] = t_up.translate(self.projections[comp_k_ids,:])
                    self.components_proj[k] = t_up.translate(self.components_proj[k])

                # k below j
                elif self.components_proj[k][1] < self.components_proj[j][1]:
                    self.projections[comp_k_ids,:] = t_down.translate(self.projections[comp_k_ids,:])
                    self.components_proj[k] = t_down.translate(self.components_proj[k])

        return self.projections
    
    def run(self):
        self.get_components(max_components=self.max_components,
                            max_dist=self.max_dist)
        
        self.get_component_center()

        self.project_components_center(proj_method=self.proj_method)

        self.get_component_ranges()
        
        self.join_components()

        return self.projections
    
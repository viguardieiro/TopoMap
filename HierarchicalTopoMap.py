import numpy as np 
from scipy.cluster.hierarchy import DisjointSet

from utils import Transform, get_hull

from TopoMap import TopoMap

class HierarchicalTopoMap(TopoMap):
    def __init__(self, points:np.ndarray,
                 metric='euclidean',
                 max_components=-1,
                 max_dist=-1,
                 min_points_component=2) -> None:
        self.points = points
        self.n = len(points)
        self.metric = metric

        self.max_components = max_components
        self.max_dist = max_dist
        self.min_points_component = min_points_component

        self.mst = self.get_mst()
        self.sorted_edges = self.get_sorted_edges()

        self.projections = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.components = DisjointSet(list(range(self.n)))
        self.subsets = None
        self.components_points = None
        self.components_center = None
        self.components_hull = None
    
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
    
    def get_components_hull(self):
        self.components_hull = []

        for j in range(len(self.subsets)):
            comp_ids = list(self.subsets[j])
            points = self.projections[comp_ids,:]

            if len(comp_ids) == 1:
                self.components_hull.append(points)
            else:
                hull = get_hull(points)
                self.components_hull.append(hull)

        return self.components_hull
    
    def get_components(self, 
                       max_components=-1, 
                       max_dist=-1):

        for i in range(len(self.sorted_edges)):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']

            if max_dist!=-1 and d >= max_dist:
                print(f'[INFO] Min distance hit. Distance: {d} | Max_dist: {max_dist}')
                self.next_dist = d
                break
            if max_components!=-1 and len(self.components.subsets()) <= max_components:
                print(f'[INFO] Max components hit. # components: {len(self.components.subsets())} | Max_components: {max_components}')
                self.next_dist = d
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
            self.next_dist = d

        self.last_processed_edge = i
        self.subsets = self.components.subsets()
        self.components_points = self.get_points_of_components()

        return self.components

    def get_components_center(self):
        self.components_center = []

        for j in range(len(self.subsets)):
            comp_ids = list(self.subsets[j])
            points = self.projections[comp_ids,:]

            center_x = (points[:,0].max() - points[:,0].min())/2
            center_y = (points[:,1].max() - points[:,1].min())/2
            center = np.array([center_x, center_y])
            self.components_center.append(center)

        return self.components_center
    
    def get_components_density(self):
        self.components_density = []

        for j in range(len(self.subsets)):
            comp_ids = list(self.subsets[j])
            points = self.projections[comp_ids,:]

            if points.shape[0] < self.min_points_component:
                self.components_density.append(0)
                continue

            side_x = points[:,0].max() - points[:,0].min()
            side_y = points[:,1].max() - points[:,1].min()
            if side_x == 0 or side_y == 1:
                density = 0
            else:
                density = len(points)/(side_x*side_y)
            self.components_density.append(density)

        return self.components_density
    
    def scale_component(self, component_id, alpha):
        comp_ids = list(self.subsets[component_id])
        points = self.projections[comp_ids,:]
        
        t_center = Transform(x=-self.components_center[component_id][0],
                             y=-self.components_center[component_id][1])
        self.projections[comp_ids,:] = t_center.translate(self.projections[comp_ids,:])
        
        
        area = (points[:,0].max() - points[:,0].min())*(points[:,1].max() - points[:,1].min())
        print(f' scaling - initial area: {area:.3f}...', end='')

        t_scale = Transform(scalar=alpha)
        self.projections[comp_ids,:] = t_scale.scale(self.projections[comp_ids,:])

        t_center = Transform(x=self.components_center[component_id][0],
                             y=self.components_center[component_id][1])
        self.projections[comp_ids,:] = t_center.translate(self.projections[comp_ids,:])

        points = self.projections[comp_ids,:]
        area = (points[:,0].max() - points[:,0].min())*(points[:,1].max() - points[:,1].min())
        print(f'done - final area: {area:.3f}.')

        return self.projections
    
    def scale_components(self):
        self.max_density = 1/(self.next_dist**2)
        self.components_alpha = []

        print(f'Max_density: {self.max_density:.3f}')

        for j in range(len(self.subsets)):
            if (len(self.subsets[j]) < self.min_points_component or
                self.components_density[j]==0):
                self.components_alpha.append(1)
                continue

            alpha = np.sqrt(len(self.subsets[j])*self.components_density[j]/self.max_density)
            self.components_alpha.append(alpha)

            print(f'Scalling component {j} - Density: {self.components_density[j]:.3f} - Alpha: {alpha:.3f}...', end='')

            # Now scale the component by alpha
            self.scale_component(j, alpha)

        return self.projections
    
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
    
    def run(self):
        self.get_components(max_components=self.max_components,
                            max_dist=self.max_dist)
        
        self.get_components_center()

        self.get_components_density()
        
        self.scale_components()

        self.join_components()

        return self.projections
    
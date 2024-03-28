import pandas as pd
import numpy as np 
from scipy.cluster.hierarchy import DisjointSet

from utils import Transform, get_hull

from TopoMap import TopoMap
from TopoTree import TopoTree

class HierarchicalTopoMap(TopoMap):
    def __init__(self, points:np.ndarray,
                 metric='euclidean',
                 index_path = '',
                 drop_zeros = False, 
                 approach = 'mlpack', 
                 load_mst = False, 
                 mst_path = '',
                 min_points_component=2,
                 max_edge_length=-1,
                 components_to_scale=[],
                 max_scalar=20,
                 selection_method='layers_down',
                 layers_down=3,
                 n_components=8,
                 min_persistence_selection=0.1,
                 min_size_selection=0,
                 max_persistence_selection=0.1,
                 max_size_selection=1,
                 verbose=True
                 ) -> None:
        """Generate HierarchicalTopoMap projection of the input data points.

        Parameters
        ----------
        
        points : numpy array
            The high dimensional data points to be projected.

        metric : {'euclidean'}, default='euclidean'
            Distance metric to compute the MST. Check out TopoMap to see other options.

        min_points_component : int, default=2
            Minumum number of points for a component to have an assigned box.

        max_edge_length : float, default=-1 (ignore restriction)
            Optional parameter to control the maximum length for an edge to be processed.

        components_to_scale : list, default=[]
            List of component ids to be scaled. If it is empty, the method get_component_to_scale 
            will be used to select the components.

        max_scalar : float, default=20
            Maximum scalar alpha to scale a component.

        selection_method : {'layers_down', 'n_components', 'min_persistence', 'min_size', \
                            'max_persistence', 'max_size'}, default='layers_down'
            Method for selecting the components to be scale. Only used if components_to_scale is empty.

        layers_down : int, default=3
            Number of layers to go down from the hierarchical tree node to get the components to scale.
            Only used if components_to_scale is empty and selection_method is 'layers_down'.

        n_components : int, default=8
            Number of components to select for scaling. The selection starts on the root of the tree
            and moves towards the leafs until we have selected n_components. If we reach a component
            wihtout childs, it is added to the list of selected components. Otherwise, we continue
            processing the component's childs until the end criteria is met.
            Only used if components_to_scale is empty and selection_method is 'n_components'.

        min_persistence_selection : float, default=0.1
            Minimum persistence a component must have to be selected. The selection starts on the root 
            of the tree and moves towards the leafs while the components have persistence above this value. 
            If we reach a component wihtout childs, it is added to the list of selected components. 
            If all childs of this component have persistence less then this value, the component
            is added to the list of selected components. Otherwise, we continue processing the component's 
            childs until there are no more components to explore.
            Only used if components_to_scale is empty and selection_method is 'min_persistence'.

        min_size_selection : float, default=0.1
            Minimum number of points a component must have to be selected. The selection starts on the root 
            of the tree and moves towards the leafs while the components have size above this value. 
            If we reach a component wihtout childs, it is added to the list of selected components. 
            If all childs of this component have size smaller then this value, the component is added to the 
            list of selected components. Otherwise, we continue processing the component's childs until 
            there are no more components to explore.
            Only used if components_to_scale is empty and selection_method is 'min_size'.

        max_persistence_selection : float, default=0.1
            Maximum persistence a component must have to be selected. The selection starts on the root 
            of the tree and moves towards the leafs until the components have persistent below this value. 
            If the component persistence below this value, it is added to the list of selected components. 
            Otherwise, we continue exploring the component's childs.
            Only used if components_to_scale is empty and selection_method is 'max_persistence'.

        max_size_selection : float, default=1
            Maximum number of points a component must have to be selected. The selection starts on the root 
            of the tree and moves towards the leafs until the components size is below this value. 
            If the component size below this value, it is added to the list of selected components. 
            Otherwise, we continue exploring the component's childs.
            Only used if components_to_scale is empty and selection_method is 'max_size'.

        verbose : bool, default=True
            Option to print information about the execution.
        """
        self.verbose = verbose

        self.points = points
        self.n = len(points)
        self.metric = metric

        self.approach = approach
        self.load_mst  = load_mst
        self.mst_path = mst_path
        self.drop_zeros = drop_zeros
        self.index_path = index_path

        self.min_points_component = min_points_component
        self.max_edge_length = max_edge_length
        self.components_to_scale = components_to_scale
        self.max_scalar = max_scalar

        # Options for get_component_to_scale
        self.selection_method = selection_method
        self.layers_down = layers_down
        self.n_components = n_components
        self.min_persistence_selection = min_persistence_selection
        self.min_size_selection = min_size_selection
        self.max_persistence_selection = max_persistence_selection
        self.max_size_selection = max_size_selection

        self.index = None
        self.mst = None
        self.sorted_edges = None

        self.projections = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.components = DisjointSet(list(range(self.n)))
        self.points_component = np.array([None for i in range(self.n)])
        self.components_info = []
        self.components_density = []
        self.subsets = None
        self.components_points = None
        self.components_hull = None
        self.points_scaled = np.zeros(shape=(self.n), dtype=bool)

        self.goal_density = -1
        if max_edge_length!=-1:
            self.goal_density = 1/(np.pi*max_edge_length**2)
        else:
            if not self.sorted_edges is None:
                biggest_edge = self.sorted_edges[-1][2]
                self.goal_density = 1/(np.pi*biggest_edge**2)

    def get_component_to_scale(self, 
                               df_comp=None, 
                               selection_method='layers_down',
                               layers_down=3,
                               n_components=8,
                               min_persistence=0.1,
                               min_size=0,
                               max_persistence=0.1,
                               max_size=0):
        if df_comp is None:
            topotree = TopoTree(self.points, 
                                min_box_size=self.min_points_component)
            topotree.mst = self.mst
            topotree.sorted_edges = self.sorted_edges
            comp_info = topotree.run()

            df_comp = pd.DataFrame.from_dict(comp_info)

        childs_map = {}
        for i in range(df_comp.shape[0]):
            childs_component = df_comp[df_comp['parent']==i].id.to_list()
            childs_map[i] = childs_component

        selected_comps = []

        if selection_method == 'layers_down':       
            parents = [df_comp.shape[0]-1]
            
            for i in range(layers_down):
                parents_aux = parents.copy()
                for parent in parents:
                    parents_aux.remove(parent)
                    if len(childs_map[parent]) > 0:
                        parents_aux.extend(childs_map[parent])
                    else:
                        selected_comps.append(parent)
                parents = parents_aux

        elif selection_method == 'n_components':
            parents = [df_comp.shape[0]-1]

            while len(selected_comps) < n_components:
                parents_aux = parents.copy()
                for parent in parents:
                    parents_aux.remove(parent)
                    if len(childs_map[parent]) > 0:
                        parents_aux.extend(childs_map[parent])
                    else:
                        selected_comps.append(parent)
                parents = parents_aux
                if len(parents) == 0:
                    break

        elif selection_method == 'min_persistence' or selection_method == 'min_size':
            if selection_method == 'min_persistence':
                feature = 'died_at'
                min_value = min_persistence
            else:
                feature = 'size'
                min_value = min_size

            parents = [df_comp.shape[0]-1]

            while len(parents) > 0:
                parents_aux = parents.copy()
                for parent in parents:
                    parents_aux.remove(parent)
                    if len(childs_map[parent]) > 0:
                        added_child = False
                        for child in childs_map[parent]:
                            if df_comp.loc[child,feature] > min_value:
                                parents_aux.append(child)
                                added_child = True
                        if not added_child:
                            selected_comps.append(parent)
                    else:
                        selected_comps.append(parent)
                parents = parents_aux

        elif selection_method == 'max_persistence' or selection_method == 'max_size':
            if selection_method == 'max_persistence':
                feature = 'died_at'
                max_value = max_persistence
            else:
                feature = 'size'
                max_value = max_size

            parents = [df_comp.shape[0]-1]

            while len(parents) > 0:
                parents_aux = parents.copy()
                for parent in parents:
                    parents_aux.remove(parent)
                    if df_comp.loc[parent,feature] < max_value:
                        selected_comps.append(parent)
                    else:
                        parents_aux.extend(childs_map[parent])
                parents = parents_aux
        
        else:
            print("[ERROR] Selection method not recognized. Options: 'layers_down', 'n_components', 'min_persistence', 'min_size', 'max_persistence', 'max_size'")
            return False
        
        self.components_to_scale = selected_comps
        return selected_comps


    def get_component_density(self, component_id):
        comp_ids = self.components_info[component_id]['points']
        points = self.projections[comp_ids,:]

        side_x = points[:,0].max() - points[:,0].min()
        side_y = points[:,1].max() - points[:,1].min()
        if side_x == 0 or side_y == 1:
            density = 0
        else:
            density = len(points)/(side_x*side_y)
        self.components_density.append(density)

        self.components_info[component_id]['density'] = density

        return density
    
    def scale_component(self, component_id):
        density = self.get_component_density(component_id)
        m = self.components_info[component_id]['size']
        alpha = np.sqrt(np.sqrt(m)*density/self.goal_density)

        alpha = min([alpha, self.max_scalar])

        comp_ids = self.components_info[component_id]['points']
        self.points_scaled[comp_ids] = True

        if self.verbose:
            print(f'Scalling component {component_id} - Scale: {alpha}', end='')
            points = self.projections[comp_ids,:]
            area = (points[:,0].max() - points[:,0].min())*(points[:,1].max() - points[:,1].min())
            print(f' scaling - initial area: {area:.3f}...', end='')

        t_scale = Transform(scalar=alpha)
        self.projections[comp_ids,:] = t_scale.scale(self.projections[comp_ids,:])

        if self.verbose:
            points = self.projections[comp_ids,:]
            area = (points[:,0].max() - points[:,0].min())*(points[:,1].max() - points[:,1].min())
            print(f' final area: {area:.3f}.')

        return alpha

    def merge_components_boxes(self, c_a, c_b, i_a, i_b, d):
        # If a has a box
        if not self.points_component[i_a] is None:

            # If b does not have a box -> Join b to a's box
            if self.points_component[i_b] is None:
                a_box_id = self.points_component[i_a]
                self.points_component[list(c_b)] = a_box_id
                self.components_info[a_box_id]['points'].extend(list(c_b))
                self.components_info[a_box_id]['size'] += len(c_b)
                self.components_info[a_box_id]['persistence'] = d
                self.components_info[a_box_id]['children'] += 1

            # If b has a box -> Create parent box
            else:
                parent_box_id = len(self.components_info)
                self.components_info.append({})
                self.components_info[parent_box_id]['id'] = parent_box_id
                self.components_info[parent_box_id]['points'] = list(c_a.union(c_b))
                self.components_info[parent_box_id]['size'] = len(c_a)+len(c_b)
                self.components_info[parent_box_id]['persistence'] = d
                self.components_info[parent_box_id]['created_at'] = d
                self.components_info[parent_box_id]['children'] = 2

                a_box_id = self.points_component[i_a]
                self.components_info[a_box_id]['parent'] = parent_box_id
                self.components_info[a_box_id]['died_at'] = d
                self.components_info[a_box_id]['persistence'] = d-self.components_info[a_box_id]['created_at']

                b_box_id = self.points_component[i_b]
                self.components_info[b_box_id]['parent'] = parent_box_id
                self.components_info[b_box_id]['died_at'] = d
                self.components_info[b_box_id]['persistence'] = d-self.components_info[b_box_id]['created_at']
                
                self.points_component[list(c_a)] = parent_box_id
                self.points_component[list(c_b)] = parent_box_id

                # Scale a and b
                if a_box_id in self.components_to_scale:
                    alpha_a = self.scale_component(a_box_id)
                    self.components_info[a_box_id]['alpha'] = alpha_a
                if b_box_id in self.components_to_scale:
                    alpha_b = self.scale_component(b_box_id)
                    self.components_info[b_box_id]['alpha'] = alpha_b

        # If a does not have a box
        else:
            # If b has a box -> Join a to b's box
            if not self.points_component[i_b] is None:
                b_box_id = self.points_component[i_b]
                self.points_component[list(c_a)] = b_box_id
                self.components_info[b_box_id]['points'].extend(list(c_a))
                self.components_info[b_box_id]['size'] += len(c_a)
                self.components_info[b_box_id]['persistence'] = d
                self.components_info[b_box_id]['children'] += 1

            # If none has box
            else:
                # If a and b merged is bigger than min -> Create box
                if len(c_a)+len(c_b) >= self.min_points_component:
                    new_box_id = len(self.components_info)
                    self.components_info.append({})
                    self.components_info[new_box_id]['id'] = new_box_id
                    self.components_info[new_box_id]['points'] = list(c_a.union(c_b))
                    self.components_info[new_box_id]['size'] = len(c_a)+len(c_b)
                    self.components_info[new_box_id]['persistence'] = d
                    self.components_info[new_box_id]['created_at'] = d
                    self.components_info[new_box_id]['children'] = 0

                    self.points_component[list(c_a)] = new_box_id
                    self.points_component[list(c_b)] = new_box_id
    
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
    
    def get_components(self):
        for i in range(min(self.n, len(self.sorted_edges))):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]

            # Distance between points
            d = self.sorted_edges[i][2]

            if self.max_edge_length!=-1 and d > self.max_edge_length:
                if self.verbose:
                    print(f'[INFO] Max edge length hit. Distance: {d} | max_edge_length: {self.max_edge_length}')
                self.next_dist = d
                break

            # Get components the points belong to
            c_a, c_b = self.components.subset(i_a), self.components.subset(i_b)

            # Merge boxes and scale if necessary
            self.merge_components_boxes(c_a, c_b, i_a, i_b, d)

            # Get points from edge and components
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]
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
            if self.verbose:
                print(f'[INFO] Number of edges hit. Edges processed: {i}')
            self.next_dist = d

        self.last_processed_edge = i
        self.subsets = self.components.subsets()
        self.components_points = self.get_points_of_components()
        self.get_points_of_components()

        return self.components

    def get_scaled_hulls(self):
        for c in self.components_to_scale:
            c_points_id = self.components_info[c]['points']
            c_hull = get_hull(self.projections[list(c_points_id),:])
            self.components_info[c]['hull'] = c_hull

        return self.components_info

    def run(self):
        if self.index is None:
            self.index = self._compute_index(self.index_path)

        if self.mst is None:
            self.mst = self._compute_mst()

        if self.sorted_edges is None:
            self.sorted_edges = self._compute_ordered_edges()

        if self.goal_density==-1:
            biggest_edge = self.sorted_edges[-1][2]
            self.goal_density = 1/(np.pi*biggest_edge**2)

        if len(self.components_to_scale) == 0:
            self.components_to_scale = self.get_component_to_scale(selection_method=self.selection_method,
                                                                   layers_down=self.layers_down,
                                                                   min_persistence=self.min_persistence_selection,
                                                                   min_size=self.min_size_selection,
                                                                   max_persistence=self.max_persistence_selection,
                                                                   max_size=self.max_size_selection
                                                                   )

        self.get_components()

        self.get_scaled_hulls()

        return self.projections
    
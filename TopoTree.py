import numpy as np 
from scipy.cluster.hierarchy import DisjointSet

from TopoMap import TopoMap

class TopoTree(TopoMap):
    def __init__(self, points:np.ndarray,
                 metric='euclidean',
                 min_box_size=10,
                 ) -> None:
        self.points = points
        self.n = len(points)
        self.metric = metric

        self.min_box_size = min_box_size

        self.mst = self.get_mst()
        self.sorted_edges = self.get_sorted_edges()

        self.components = DisjointSet(list(range(self.n)))
        self.points_component = np.array([None for i in range(self.n)])
        self.components_info = []
    
    def merge_components_boxes(self, c_a, c_b, i_a, i_b, d, i):
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
                    self.components_info[parent_box_id]['created_at'] = i
                    self.components_info[parent_box_id]['children'] = 2

                    a_box_id = self.points_component[i_a]
                    b_box_id = self.points_component[i_b]
                    self.components_info[a_box_id]['parent'] = parent_box_id
                    self.components_info[b_box_id]['parent'] = parent_box_id

                    self.points_component[list(c_a)] = parent_box_id
                    self.points_component[list(c_b)] = parent_box_id
                    
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
                    if len(c_a)+len(c_b) >= self.min_box_size:
                        new_box_id = len(self.components_info)
                        self.components_info.append({})
                        self.components_info[new_box_id]['id'] = new_box_id
                        self.components_info[new_box_id]['points'] = list(c_a.union(c_b))
                        self.components_info[new_box_id]['size'] = len(c_a)+len(c_b)
                        self.components_info[new_box_id]['persistence'] = d
                        self.components_info[new_box_id]['created_at'] = i
                        self.components_info[new_box_id]['children'] = 0

                        self.points_component[list(c_a)] = new_box_id
                        self.points_component[list(c_b)] = new_box_id
    
    def get_components(self):

        for i in range(len(self.sorted_edges)):
            # Get points from the edge
            i_a, i_b = self.sorted_edges[i][0], self.sorted_edges[i][1]

            # Distance between points
            d = self.sorted_edges[i][2]['weight']

            # Get components the points belong to
            c_a, c_b = self.components.subset(i_a), self.components.subset(i_b)

            # Merge components 
            self.merge_components_boxes(c_a, c_b, i_a, i_b, d, i)
            self.components.merge(i_a, i_b)

        return self.components_info
    
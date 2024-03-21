import numpy as np 
import mlpack

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import DisjointSet

import networkx as nx

from utils import get_hull, closest_edge_point, find_angle, Transform, fix_rotation, check_aligned_points,distance_segment_point

import matplotlib.pyplot as plt


class TopoMapMlPack():
    def __init__(self, points:np.ndarray,
                 metric='euclidean') -> None:
        self.points = points
        self.n = len(points)
        self.metric = metric

        self.mst = self.get_mst()
        self.sorted_edges = self.get_sorted_edges()

        self.projections = np.zeros(shape=(self.n, 2), dtype=np.float32)
        self.components = DisjointSet(list(range(self.n)))

    def get_mst(self):
        d = mlpack.emst(input_ = self.points)
        d = d["output"]
        d = np.array([[int(d[i,0]),int(d[i,1]),d[i,2]] for i in range(len(d))],dtype='O')
        self.mst = d
        return self.mst
    
    def get_sorted_edges(self):
        self.sorted_edges = self.mst[self.mst[:, 2].argsort()]
        return self.sorted_edges
    
    def rotate_component(self, 
                         component_points:np.ndarray, 
                         ref_point:np.ndarray, 
                         direction='top') -> np.ndarray:
        #Create Copy of input_elements and after put it back
                
        unique_points = np.unique(component_points, axis=0)

        
        original_repeated_points = component_points.copy()
        # repeated_componente_points = component_points.copy()
        
        component_points = unique_points.copy()
        if len(unique_points) == 1:
            return original_repeated_points, [0,0]
        
        # if(check_aligned_points(component_points)):
        #     #O(n)
        #     #All points aligned
        #     hull = get_hull(component_points,aligned_points = True)
        # else:
        try:
            hull = get_hull(component_points)
        except Exception as e:
            error_message = str(e)
            error_type = error_message[0:6]
            print(error_type)
            hull = get_hull(component_points,aligned_points=True)

        # try:
        #     repeted_hull = get_hull(repeated_componente_points)
        # except Exception as e:
        #     error_message = str(e)
        #     error_type = error_message[0:6]
        #     # print(error_type)
        #     repeted_hull = get_hull(repeated_componente_points,aligned_points=True)
        


        

        closest_edge, edge_i = closest_edge_point(hull, ref_point)
       

        # repeated_closest_edge, repeated_edge_i = closest_edge_point(repeted_hull, ref_point)
        # # print("using repeated:", repeated_closest_edge, distance_segment_point(repeated_closest_edge,ref_point))
        # # print("-----------")
        # if(not np.array_equal(closest_edge,repeated_closest_edge)  ):
        #      #checking if change the order solve
        #      if ( not ( np.array_equal(closest_edge[0], repeated_closest_edge[1] ) and np.array_equal(closest_edge[1], repeated_closest_edge[0])) ):
        #         #checking if they have the same distance, what can happen
        #         if(distance_segment_point(closest_edge,ref_point ) !=  distance_segment_point(repeated_closest_edge,ref_point)):
        #             print("not equal")
        #             print(repeated_componente_points)
        #             print("closest edge to", ref_point, ":",closest_edge,distance_segment_point(closest_edge,ref_point )) 
        #             print("using repeated:", repeated_closest_edge, distance_segment_point(repeated_closest_edge,ref_point))
        
        
        
        t = Transform()
        t.cos, t.sin = find_angle(closest_edge)
        component_points = t.rotate(component_points)

        component_points = fix_rotation(component_points[edge_i], 
                                        component_points, 
                                        direction=direction)
        
        if(len(unique_points) == len(original_repeated_points)):
            #Case where there are no repeated points
            return component_points, edge_i
        
        #Case where there are repeated points
        #Apply the same transformation to all points 
        transformed_repeated_points = [0]*len(original_repeated_points)
        map_original_transformed = {str(k):v for k, v in zip(unique_points, component_points)}
        # for i in range(len(unique_points)):
        #     #O(n)
        #     key = str(unique_points[i])
        #     value = component_points[i]
        #     if(key in map_original_transformed.keys()):
        #         continue
        #     else:
        #         map_original_transformed[key] = value
        
        
        for i in range(len(original_repeated_points)):
            #O(n)
            key = str(original_repeated_points[i])
            transformed_repeated_points[i] = map_original_transformed[key]
        transformed_repeated_points = np.array(transformed_repeated_points)

        #transforming the edge to previous correspondence
        b_edge = edge_i[0]
        e_edge =edge_i[1]
        point_b_edge = unique_points[b_edge,:]
        point_e_edge = unique_points[e_edge,:]
        edge = [0,0]
        for i in range(len(original_repeated_points)):
            point = original_repeated_points[i]
            if(np.array_equal(point_b_edge,point)):
                edge[0] = i
                break

        for i in range(len(original_repeated_points)):
            point = original_repeated_points[i]
            if(np.array_equal(point_e_edge,point)):
                edge[1] = i
                break

        
        
        
        

        return transformed_repeated_points, edge
    
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
            i_a, i_b = int(self.sorted_edges[i][0]), int(self.sorted_edges[i][1])
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # Distance between points
            d = self.sorted_edges[i][2]
            
            # Get components the points belong to
            c_a, c_b = self.components.subset(i_a), self.components.subset(i_b)
            proj_c_a, proj_c_b = self.projections[list(c_a)], self.projections[list(c_b)]
            i_a_comp, i_b_comp = list(c_a).index(i_a), list(c_b).index(i_b)

            if i==iter-1:
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

            if i==iter-1:
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

            if i==iter-1:
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
    
    def run(self):
        for i in range(min(self.n, len(self.sorted_edges))):
            

            # Get points from the edge
            i_a, i_b = int(self.sorted_edges[i][0]), int(self.sorted_edges[i][1])
            # print(i_a,i_b)
            # print("-------")
            p_a, p_b = self.projections[i_a,:], self.projections[i_b,:]

            # print("projections")
            # print(p_a,p_b)

            # Distance between points
            d = self.sorted_edges[i][2]
            
            
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
            # print("Pos translacao")
            # print(proj_c_a,proj_c_b)
            # Merge components 
            self.components.merge(i_a, i_b)

            self.projections[list(c_a), :] = proj_c_a
            self.projections[list(c_b), :] = proj_c_b

        return self.projections
    
import pickle
import numpy as np
import datetime
from pathlib import Path
from Vamanautils import insert_ordered, almost_in
from distance_pair import distance_pair
from sortedcontainers import SortedList

class VamanaIndex:
    """Vamana graph algorithm implementation. Every element in each graph is a
    2-tuple containing the vector and a list of unidirectional connections
    within the graph.
    """

    def __init__(self, L: int = 40, a: float = 1.2, R: int = 40):
        super().__init__()
        self._L = L
        self._a = a
        self._R = R
        self._start = None  # index of starting vector
        self._index = {} #dictionary with key idx and value = (vec, out_neighbors)
        self._size = None


    def one_pass(self,a = None):
        permutation = np.random.permutation(self._size)
        for n in permutation:
            node = self._index[n]
            
            (_, V) = self.search(node[0], nq=1)
            # print(node)
            self._robust_prune(n, node, V,a = a)
            # print(node)
            
            for inb in node[1]:
                nbr = self._index[inb]
                if len(nbr[1].union({n})) > self._R:
                    
                    self._robust_prune(inb, nbr, nbr[1].union({n}))
                    
                else:
                    nbr[1].add(n)
        
    def create(self, dataset,save_intermediate = False):
        '''
        Create a Vamana Index on the dataset. Set save_intermediate  = True to save the intermediate states of the graph during the algorithm
        '''
        

        self._R = min(self._R, len(dataset))
        starting_time = datetime.datetime.now()
        # intialize graph with dataset

        # set starting location as medoid vector
        dist = float("inf")
        medoid = np.median(dataset, axis=0)
        self._size = len(dataset)

        for (n, vec) in enumerate(dataset):
            d = np.linalg.norm(medoid - vec)
            if d < dist:
                dist = d
                self._start = n
            self._index[n]= (vec, set())

       
        # randomize out-connections for each node
        #idxs vary from 0 to n-2, so when we sum 1, the max is still n-1 
        for n in range(self._size):
            node = self._index[n]
            idxs = np.random.choice(self._size - 1, replace=False, size=(self._R,))
            idxs[idxs >= n] += 1  # ensure no node points to itself
            node[1].update(idxs)

        self.one_pass(a=1)
        self.one_pass()

        
        

            


    def search(self,query, nq: int = 10):
        """Greedy search.
        """
        assert (self._L >= nq,f'expected L >= k, but received L = {self._L} and k = {nq}')


        
        query_distance = np.linalg.norm(self._index[self._start][0] - query)
        
        nns = SortedList() #nns is a SortedList or pairs(index,distance) ordered by distance
        nns.add(distance_pair(self._start,query_distance))
        
        visit = set()  # set of visited nodes
        

        # find top-k nearest neighbors
        while set([dp.index for dp in nns]) - visit:             #Check if have nns that were not visited. TO DO: speed up that part.
            
            for i in range(len(nns)):                 #Improve that. nns is a list of neighbors ordered by start distance. I want to find the smallest element of nns that is not in visit. This approach takes O(nns * visit)
                if nns[i].index not in visit :  
                    nn = nns[i].index
                    
                    break
            #We only define nn inside the loop, but since the while condition is still satisfied, I know that the if condition will be satisfied at least one time.
            
            for idx in self._index[nn][1]:
                #Union of nns and N_out(nn)
                
                if idx in [dp.index for dp in nns]: #Here is other point of improvement. Using SortedLists has this drawback of have to create a list of index to search. Maybe I can do a for loop and check instead of creating a new list to use "in"
                    
                    continue
                    
                d = np.linalg.norm(self._index[idx][0] - query)               
                
                nns.add(distance_pair(idx,d))               
            
            visit.add(nn)

            # retain up to search list size elements.
            if(len(nns)>self._L):
                nns = SortedList(nns[:self._L])
        
        # print(nns)

        return (nns[:nq], visit)


    def _robust_prune(self, index_node:int ,node: tuple[np.ndarray, set[int]], candid: set[int], a = None):
        #TO DO: Update the data structure to record the index of the index of the current node, not only of the neighbors
        if( a is None):
            a = self._a
        
        candid.update(node[1])

        if index_node in candid:
            candid.remove(index_node)


        #Removing out neighbors of index_node
        node[1].clear()


        while candid:
            (min_d, nn) = (float("inf"), None)

            # find the closest element/vector to input node
            for k in candid:
                p = self._index[k][0]
                d = np.linalg.norm(node[0] - p)
                if d < min_d:
                    (min_d, nn) = (d, k)
            node[1].add(nn)

            # set at most R out-neighbors for the selected node
            if len(node[1]) == self._R:
                break

            # future iterations must obey distance threshold
            delete_set = set()
            for idx in candid:
                if a * np.linalg.norm(self._index[nn][0]- self._index[idx][0]) <= np.linalg.norm(node[0] - self._index[idx][0] ):
                    delete_set.add(idx)
            candid = candid - delete_set

    def get_index(self):
        return self._index

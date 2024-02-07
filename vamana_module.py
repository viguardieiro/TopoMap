# from heapq import heapify, heappop, heappush
# import bisect
import pickle
import numpy as np
from Vamanautils import insert_ordered, almost_in

class VamanaIndex:
    """Vamana graph algorithm implementation. Every element in each graph is a
    2-tuple containing the vector and a list of unidirectional connections
    within the graph.
    """

    def __init__(self, L: int = 10, a: float = 1.5, R: int = 10):
        super().__init__()
        self._L = L
        self._a = a
        self._R = 10
        self._start = None  # index of starting vector
        self._index = {} #dictionary with key idx and value = (vec, out_neighbors)
        self._size = None


    def create(self, dataset):
        

        self._R = min(self._R, len(dataset))

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
        #idxs vary from 0 to n-2, so when we sum 1, the max is still n-1.
        for n in range(self._size):
            node = self._index[n]
            idxs = np.random.choice(self._size - 1, replace=False, size=(self._R,))
            idxs[idxs >= n] += 1  # ensure no node points to itself
            node[1].update(idxs)
        

        # random permutation + sequential graph update
        #TO DO: Here there is no random permutation, only doing that in the same order.
        for n in range(self._size):
            node = self._index[n]
            
            (_, V) = self.search(node[0], nq=1)
            # print(node)
            self._robust_prune(node, V)
            # print(node)
            
            for inb in node[1]:
                nbr = self._index[inb]
                if len(nbr[1].union({n})) > self._R:
                    
                    self._robust_prune(nbr, nbr[1].union({n}))
                    
                else:
                    nbr[1].add(n)
            # with open(f'index_iter_{n}.pkl', 'wb') as f:
            #     pickle.dump(self._index,f)


    def search(self,query, nq: int = 10):
        """Greedy search.
        """
       


        
        query_distance = np.linalg.norm(self._index[self._start][0] - query)
        nns = np.array([[query_distance, self._start]]) #nns is a list of pairs of the form (distance, index)
        visit = set()  # set of visited nodes
        

        # find top-k nearest neighbors
        while set(nns[:,1]) - visit:             #Check if have nns that were not visited. TO DO: speed up that part.
           
            for neighbor in nns:                 #Improve that. nns is a list of neighbors ordered by start distance. I want to find the smallest element of nns that is not in visit. This approach takes O(nns * visit)
                if not almost_in(visit, neighbor[1]) :
                    nn = neighbor[1]
                    
                    break
            #We only define nn inside the loop, but since the while condition is still satisfied, I know that the if condition will be satisfied at least one time.
            
            for idx in self._index[nn][1]:

                d = np.linalg.norm(self._index[idx][0] - query)
                  
                if almost_in(nns[:,1],idx):
                    
                    continue            
                
                nns = insert_ordered(nns,[d,idx])
                
                # heappush(nns, (d, idx))
            
            visit.add(nn)

            # retain up to search list size elements.
            if(len(nns)>self._L):
                nns = nns[:self._L]
        
        # print(nns)

        return (nns[:nq], visit)


    def _robust_prune(self,node: tuple[np.ndarray, set[int]], candid: set[int], a = None):
        #TO DO: Update the data structure to record the index of the index of the current node, not only of the neighbors
        if( a is None):
            a = self._a
        
        candid.update(node[1])
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

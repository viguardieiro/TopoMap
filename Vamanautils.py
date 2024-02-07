import numpy as np 



def insert_ordered(ordered_array: list[list[float, int]], item:list[float,int]):
    '''function to insert item in an ordered array.
    Here we are ordering by the second column of the array.
    Result: return a new ordered_array with the item added and ordered
    '''

    #trivial way but not good

    ordered_array = np.concatenate((ordered_array,np.array([item])),axis = 0)
    ordered_array = ordered_array[ordered_array[:, 0].argsort()]



    # if nns[-1,0] < d:
    #                 nns = np.concatenate((nns,[[d,idx]]),axis = 0)
                    
    #             else :
    #                 sanity_check = False
    #                 for i in range(nns.shape[0]):
    #                     if nns[i,0] > d: #First index that is greater than d. I have to append in this position to preserve the order. 
    #                         index = i
    #                         sanity_check = True
    #                         break
    #                 if not sanity_check:
    #                     print(nns.shape)
    #                     print(sanity_check)
    #                     print("weird case:", nns, d)
    #                 # print(nns[:index,:],[[d,idx]], nns[index:,:])
    #                 # I don't understand why, but I'm having duplicated itens on nns

    #                 nns = np.concatenate((nns[:index,:],[[d,idx]], nns[index:,:]), axis = 0) #5x faster than np.insert 
                    

    return ordered_array


def almost_in(array:list[float], item:any, eps = 0.001)->bool:
    for item in array:
        if(abs(item - item)) < eps:
            return True
    else:
        return False


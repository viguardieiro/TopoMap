import numpy as np 



def insert_ordered(ordered_array: list[list[float, int]], item:list[float,int]):
    '''function to insert item in an ordered array.
    Here we are ordering by the second column of the array.
    Result: return a new ordered_array with the item added and ordered
    '''

    #trivial way but not good

    ordered_array = np.concatenate((ordered_array,np.array([item])),axis = 0)
    ordered_array = ordered_array[ordered_array[:, 0].argsort()]

    return ordered_array


def almost_in(array:list[float], item:any, eps = 0.001)->bool:
    for item in array:
        if(abs(item - item)) < eps:
            return True
    else:
        return False


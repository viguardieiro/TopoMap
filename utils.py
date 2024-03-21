import numpy as np 
from numpy.linalg import norm
from scipy.spatial import ConvexHull

class AuxHull():
    def __init__(self, points, simplices):
        self.points = points
        self.simplices = simplices

def get_hull(points:np.ndarray,aligned_points = False):
    if len(points) == 1:
        hull = AuxHull(points=points,
                       simplices=np.array([[0,0]]))
    elif len(points) == 2:
        hull = AuxHull(points=points,
                       simplices=np.array([[0,1], [1,0]]))
    elif aligned_points:
        print("aligned points")
        print( points)
        simplices = [[i,(i+1)%len(points)] for i in range(len(points))]
        hull = AuxHull(points = points,
                       simplices = np.array(simplices))
    else:
        hull = ConvexHull(points)
        
    return hull

def distance_segment_point(segment, p):
    a, b = segment
    
    if np.array_equal(a,p) or np.array_equal(b, p):
        return 0
    if np.array_equal(a,b):
        return norm(a-p)
    else:
        proj_p_ab = np.dot((p-a), (b-a))
        norm_proj_p_ab = proj_p_ab / (norm(b-a)**2)

        if norm_proj_p_ab <= 0:
            closest_point = a
        elif norm_proj_p_ab>= 1:
            closest_point = b
        else:
            closest_point = a + norm_proj_p_ab*(b-a)

        return norm(closest_point-p)

def closest_edge_point(hull, 
                       ref_point:np.ndarray):
    '''
    Find edge of convex hull that is closer to the reference point.
    Returns the points of the edge and its indexes.
    '''
    closest_i = -1
    closest_dist = np.inf

    for i in range(len(hull.simplices)):
        a, b = hull.points[hull.simplices[i]]
        d = distance_segment_point((a, b), ref_point)
        
        if d < closest_dist:
            closest_dist = d
            closest_i = i

    closest_edge = hull.points[hull.simplices[closest_i]]

    return closest_edge, hull.simplices[closest_i]

def find_angle(segment:np.ndarray):
    '''
    Find angle to rotate component so that segment is paralel to the vertical axis
    '''
    p1, p2 = segment
    # Use rightmost vertice as reference for angle
    if p1[0] < p2[0]:
        vec = p2 - p1
    else:
        vec = p1 - p2

    hip = norm(vec)
    vec[0] /= hip
    vec[1] /= hip

    cos = vec[0]
    sin = np.sqrt(1 - cos**2)
    if vec[1] >= 0:
        sin = -1*sin

    return cos, sin

class Transform():
    def __init__(self, x=0, y=0, sin=0, cos=1) -> None:
        self.x = x
        self.y = y
        self.sin = sin
        self.cos = cos

    def translate(self, points:np.ndarray) -> np.ndarray:
        trans_points = np.zeros(shape=(len(points),2))
        trans_points = points + np.array([self.x, self.y])
        return trans_points

    def rotate(self, points:np.ndarray) -> np.ndarray:
        rot_points = np.zeros(shape=(len(points),2))
        rot_points[:,0] = points[:,0] * self.cos - points[:,1] * self.sin
        rot_points[:,1] = points[:,0] * self.sin + points[:,1] * self.cos
        return rot_points

    def transform(self, points:np.ndarray) -> np.ndarray:
        transform_points = np.zeros(shape=(len(points),2))
        transform_points = self.translate(points)
        transform_points = self.rotate(transform_points)
        return transform_points
    
def fix_rotation(segment:np.ndarray, 
                 points:np.ndarray, 
                 direction='top') -> np.ndarray:
    '''
    Rotates points so that the segment is topmost of bottomost edge
    '''
    new_points = points.copy()

    if direction == 'top':
        max_y = points[:,1].max()
        if max_y > segment[:,1].max():
            # Rotate 180 degrees
            t = Transform(sin=0, cos=-1)
            new_points = t.rotate(new_points)

    elif direction == 'bottom':
        min_y = points[:,1].min()
        if min_y < segment[:,1].min():
            # Rotate 180 degrees
            t = Transform(sin=0, cos=-1)
            new_points = t.rotate(new_points)

    return new_points

def check_3_aligned_points(p0,p1,p2,tol=1e-12):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < tol


def check_aligned_points(points:np.ndarray):
    '''
    Check if all points are alligned 
    '''
    if(len(points)<3):
        return True

    p0 = points[0,:]
    p1 = points[1,:]
    p2 = points[2,:]
    all_colinear = check_3_aligned_points(p0,p1,p2)
    if(not all_colinear):
        return False
    else:
        for i in range(len(points) - 2):
            p0 = points[i,:]
            p1 = points[i+1,:]
            p2 = points[i+2,:]
            aux_colinear = check_3_aligned_points(p0,p1,p2)
            if(not aux_colinear):
                return False
        return True


    



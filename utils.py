import numpy as np 
from numpy.linalg import norm

def distance_segment_point(segment, p):
    a, b = segment
    
    if all(a == p) or all(b == p):
        return 0
    if all(a == b):
        return norm(a-p)
    if np.arccos(np.dot((p-a) / norm(p-a), (b-a) / norm(b-a))) > np.pi / 2:
        return norm(p-a)
    if np.arccos(np.dot((p-b) / norm(p-b), (a-b) / norm(a-b))) > np.pi / 2:
        return norm(p-b)
    return norm(np.cross(a-b, a-p))/norm(b-a)

def closest_edge_point(hull, ref_point):
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

def find_angle(segment):
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
        trans_points = points - np.array([self.x, self.y])
        return trans_points

    def rotate(self, points):
        rot_points = np.zeros(shape=(len(points),2))
        rot_points[:,0] = points[:,0] * self.cos - points[:,1] * self.sin
        rot_points[:,1] = points[:,0] * self.sin + points[:,1] * self.cos
        return rot_points

    def transform(self, points:np.ndarray) -> np.ndarray:
        transform_points = np.zeros(shape=(len(points),2))
        transform_points = self.translate(points)
        transform_points = self.rotate(transform_points)
        return transform_points
    
def fix_rotation(segment, ref_point, points, direction='top'):
    '''
    Rotates points so that the segment is topmost of bottomost edge
    '''
    new_points = points.copy()

    if direction == 'top':
        if ref_point[1] >= segment[:,1].min():
            # Rotate 180 degrees
            t = Transform(sin=0, cos=-1)
            new_points = t.rotate(new_points)

    elif direction == 'bottom':
        if ref_point[1] <= segment[:,1].min():
            # Rotate 180 degrees
            t = Transform(sin=0, cos=-1)
            new_points = t.rotate(new_points)

    return new_points
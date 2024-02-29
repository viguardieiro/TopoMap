class distance_pair(object):

    def __init__(self, index, distance):
        self.index = index
        self.distance = distance

    def __eq__(self,other,tol = 0.001):
    	return self.index == other.index and abs(self.distance - other.distance) < tol

    def __lt__(self, other):
        return self.distance < other.distance

    def __repr__(self):
    	return f'<distance_pair index:{self.index} distance:{self.distance}>'

    def __str__(self):
    	return f'(index:{self.index},distance:{self.distance})'





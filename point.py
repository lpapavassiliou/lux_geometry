import numpy as np

class Point3D(np.ndarray):
    def __new__(cls, input_array):
        # Convert input to a numpy array
        obj = np.asarray(input_array, dtype=float).view(cls)
        # Check if input has exactly 3 elements
        if obj.shape != (3,):
            raise ValueError("Input must be an iterable with 3 elements")
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # Additional attributes could be added here

    @property
    def x(self):
        return self[0]
    
    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]
    
    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]
    
    @z.setter
    def z(self, value):
        self[2] = value

    def to_list(self):
        return self.tolist()
    
    def to_tuple(self):
        return tuple(self)
    
    def to_ndarray(self):
        return np.array(self)

    def __repr__(self):
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"
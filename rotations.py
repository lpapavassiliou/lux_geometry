import numpy as np
from math import isclose

class RotationMatrix(np.ndarray):
    def __new__(cls, matrix=None):
        if matrix is None:
            obj = np.eye(3).view(cls)
        if matrix.shape != (3, 3):
            raise ValueError("Matrix must be a 3x3 matrix.")
        if not isclose(abs(np.linalg.det(matrix)), 1, rel_tol=1e-5, abs_tol=0.00001) :
            raise ValueError("Matrix must be an orthogonal matrix.")
        obj = np.asarray(matrix).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __repr__(self):
        return f"RotationMatrix({super().__repr__()})"

    @staticmethod
    def elementary(axis, angle_rad):
        assert axis in ['x', 'y', 'z']

        if axis == "x":
            R = np.array([[1,       0,                    0        ], 
                          [0, np.cos(angle_rad), -np.sin(angle_rad)], 
                          [0, np.sin(angle_rad), np.cos(angle_rad)]])
        if axis == "y":
            R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)], 
                          [0,                 1,       0           ], 
                          [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
        if axis== "z":
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                          [np.sin(angle_rad), np.cos(angle_rad), 0],
                          [0,                          0,        1]])
        
        return RotationMatrix(R)
    

class HomogeneusMatrix(np.ndarray):
    def __new__(cls, R : RotationMatrix = None, t : np.ndarray = None):
        if R is not None and t is not None:
            RotationMatrix(R) # checks that R is a valid rotation matrix
            obj = np.vstack([np.hstack([   R,     t.reshape(-1,1)]),
                             np.array([[0, 0, 0,  1]])])
            obj.view(cls)
        else:
            obj = np.eye(4).view(cls)

        obj = np.asarray(obj).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __repr__(self):
        return f"TransformationMatrix({super().__repr__()})"
    
    def __matmul__(self, other):
        if isinstance(other, HomogeneusMatrix):
            result = np.matmul(self.view(np.ndarray), other.view(np.ndarray))
            return HomogeneusMatrix(result[:3, :3], result[:3, 3])
        else:
            return np.matmul(self.view(np.ndarray), other)

    def inverse(self):
        R_old = self.get_R()
        t_old = self.get_t()
        R_new = R_old.T
        t_new = - R_old.T @ t_old
        self = HomogeneusMatrix(R_new, t_new)

    def get_R(self):
        return self[0:3, 0:3].view(np.ndarray)

    def get_t(self):
        return self[0:3, 3].view(np.ndarray)
    
    def set_R(self, R_new):
        self[0:3, 0:3] = R_new
    
    def set_t(self, t_new):
        self[0:3, 3] = t_new
    
    def transform(self, vector):
        return homogeneus_to_vector(self @ vector_to_homogeneus(vector))
        
def homogeneus_to_vector(h):
    assert h.shape == (4,)
    return h[:3]

def vector_to_homogeneus(v):
    assert v.shape == (3,)
    return np.concatenate((v, [1]))
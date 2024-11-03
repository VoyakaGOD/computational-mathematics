class Matrix:
    pass
class Vector:
    pass

def argmax(iterable) -> int:
    max_index = 0
    max_item = iterable[0]
    for index, item in enumerate(iterable):
        if item > max_item:
            max_index = index
            max_item = item
    return max_index

class Matrix:
    def __init__(self, data):
        self.data = data
        self.m = len(data)          #rows
        self.n = len(data[0])       #cols

    def shape(self) -> tuple[int, int]:
        return self.m, self.n
    
    def T(self) -> Matrix:
        return Matrix([[self.data[j][i] for j in range(self.m)] for i in range(self.n)])

    def __str__(self) -> str:
        max_item_size = 0
        for row in self.data:
            for item in row:
                max_item_size = max(max_item_size, len(str(item)))
        def min_str(item):
            string = str(item)
            delta = max_item_size - len(string)
            left = delta // 2
            return (" " * left) + str(item) + (" " * (delta - left))
        return '\n'.join(["|| " + (" ".join(map(min_str, row))) + " ||" for row in self.data])

    def __repr__(self) -> str:
        return f"Matrix({self.m}x{self.n})"

    def __add__(self, other : Matrix) -> Matrix:
        if isinstance(other, Matrix) and self.shape() == other.shape():
            return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.n)] for i in range(self.m)])
        else:
            raise ValueError("matrix add/sub error")
        
    def __mul__(self, other : Matrix | float) -> Matrix:
        if isinstance(other, Matrix):
            if self.n != other.m:
                raise ValueError("bad dimensions")
            return Matrix([[sum(self.data[i][k] * other.data[k][j] for k in range(self.n)) 
                      for j in range(other.n)] for i in range(self.m)])
        elif isinstance(other, (int, float)):
            return Matrix([[self.data[i][j] * other for j in range(self.n)] for i in range(self.m)])
        else:
            raise ValueError("unsupported operand for matrix multiplication")
        
    def __rmul__(self, other : Matrix | float) -> Matrix:
        return self * other
        
    def __sub__(self, other : Matrix) -> Matrix:
        return self + other * (-1)
    
    def __neg__(self) -> Matrix:
        return self * (-1)
    
    def __matmul__(self, other : Matrix) -> Matrix:
        if isinstance(other, Matrix):
            if self.shape() != other.shape():
                raise ValueError("bad shapes in Hadamard product")
            return Matrix([[self[i, j] * other[i, j] for j in range(self.n)] for i in range(self.m)])
        else:
            raise ValueError("unsupported operand for Hadamard product")
        
    def __rmatmul__(self, other : Matrix) -> Matrix:
        return self @ other
    
    def __truediv__(self, other : float) -> Matrix:
        return self * (1 / other)

    def __eq__(self, other : Matrix) -> bool:
        if not isinstance(other, Matrix):
            return False
        return self.data == other.data
    
    def is_square(self) -> bool:
        return self.m == self.n
    
    def minor(self, row, col) -> Matrix:
        return Matrix([self.data[i][:col] + self.data[i][col + 1:] for i in range(self.m) if i != row])
    
    def det(self) -> float:
        if not self.is_square():
            raise ValueError("attempt to get determinant of not square matrix")
        n = self.n
        data = [[item for item in row] for row in self.data]
        det = 1
        for j in range(n - 1):
            pivot = argmax([abs(data[k][j]) for k in range(j, n)]) + j
            if abs(data[pivot][j]) < 1e-15:
                return 0 #singular matrix
            data[pivot], data[j] = data[j], data[pivot]
            det *= (1 if pivot == j else -1)
            for i in range(j + 1, n):
                factor = data[i][j] / data[j][j]
                for k in range(j, n):
                    data[i][k] -= factor * data[j][k]
        for i in range(n):
            det *= data[i][i]
        return det

    def tr(self) -> float:
        if not self.is_square():
            raise ValueError("attempt to get trace of not square matrix")
        trace = 0
        for i in range(self.n):
            trace += self.data[i][i]
        return trace
    
    def copy(self) -> Matrix:
        return Matrix([[item for item in row] for row in self.data])
    
    def __getitem__(self, index : int | tuple[int, int]) -> float:
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            return self.data[row][col]
        elif isinstance(index, int):
            if self.n != 1:
                if self.m != 1:
                    raise IndexError("this indexation method is used for row or column Vectors")
                return self.data[0][index]
            return self.data[index][0]
        else:
            raise IndexError("invalid matrix index")
    
    def __setitem__(self, index : int | tuple[int, int], value : float) -> None:
        if isinstance(index, tuple) and len(index) == 2:
            row, col = index
            self.data[row][col] = value
        elif isinstance(index, int):
            if self.n != 1:
                if self.m != 1:
                    raise IndexError("this indexation method is used for row or column Vectors")
                self.data[0][index] = value
            else:
                self.data[index][0] = value
        else:
            raise IndexError("invalid matrix index")
        
    def expand(self, f : Vector) -> Matrix:
        return Matrix([row + [f[i]] for i, row in enumerate(self.data)])
    
    def zeros(m : int, n : int) -> Matrix:
        return Matrix([[0 for j in range(n)] for i in range(m)])

class Vector(Matrix):
    def __init__(self, *args):
        data = args
        if(len(args) == 1) and hasattr(args[0], '__iter__'):
            data = args[0]
        super().__init__([[item] for item in data])

    def zeros(size : int):
        return Vector([0 for i in range(size)])

def det(A : Matrix):
    return A.det()

def tr(A : Matrix):
    return A.tr()

def get_max_norm(A : Matrix):
    return max([max([abs(A[i, j]) for i in range(A.m)]) for j in range(A.n)])

def get_Manhattan_norm(A : Matrix):
    return sum([sum([abs(A[i, j]) for i in range(A.m)]) for j in range(A.n)])

def get_Euclidian_norm(A : Matrix):
    return sum([sum([A[i, j]*A[i, j] for i in range(A.m)]) for j in range(A.n)]) ** 0.5

def get_induced_max_norm(A : Matrix):
    return max([sum([abs(A[i, j]) for j in range(A.n)]) for i in range(A.m)])

def get_induced_Manhattan_norm(A : Matrix):
    return max([sum([abs(A[i, j]) for i in range(A.m)]) for j in range(A.n)])

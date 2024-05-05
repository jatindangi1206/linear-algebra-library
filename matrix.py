class MatrixError(Exception):
    pass

class Matrix:
    def __init__(self, m, n, entries):
        if len(entries) != m * n:
            raise MatrixError("The number of entries does not match the dimensions of the matrix.")
        self.m = m
        self.n = n
        self.matrix = [entries[i*n:(i+1)*n] for i in range(m)]

    def row_length(self):
        return self.m

    def column_length(self):
        return self.n

    def is_square(self):
        return self.m == self.n

    def display(self):
        for row in self.matrix:
            print(row)

    def _swap_rows(self, row1, row2):
        self.matrix[row1], self.matrix[row2] = self.matrix[row2], self.matrix[row1]

    def _multiply_row(self, row, multiplier):
        self.matrix[row] = [element * multiplier for element in self.matrix[row]]

    def _add_rows(self, source_row, target_row, multiplier):
        self.matrix[target_row] = [
            target + multiplier*source for target, source in zip(self.matrix[target_row], self.matrix[source_row])
        ]

    def rref(self):
        lead = 0
        rowCount = self.m
        columnCount = self.n
        for r in range(rowCount):
            if lead >= columnCount:
                return
            i = r
            while self.matrix[i][lead] == 0:
                i += 1
                if i == rowCount:
                    i = r
                    lead += 1
                    if columnCount == lead:
                        return
            self._swap_rows(i, r)
            lv = self.matrix[r][lead]
            self._multiply_row(r, 1/lv)
            for i in range(rowCount):
                if i != r:
                    lv = self.matrix[i][lead]
                    self._add_rows(r, i, -lv)
            lead += 1

    def rank(self):
        tempMatrix = Matrix(self.m, self.n, [item for sublist in self.matrix for item in sublist])
        tempMatrix.rref()
        return sum(1 for row in tempMatrix.matrix if any(row))

    def nullity(self):
        return self.n - self.rank()

    def invertible(self):
        return self.is_square() and self.rank() == self.m

    def inverse(self):
        if not self.invertible():
            raise MatrixError("The matrix is not invertible.")

        augmented = [row + [0]*self.n for row in self.matrix]
        for i in range(self.n):
            augmented[i][self.n+i] = 1

        augMatrix = Matrix(self.m, 2*self.n, [item for sublist in augmented for item in sublist])
        augMatrix.rref()

        inverse_matrix = [row[self.n:] for row in augMatrix.matrix]
        return Matrix(self.m, self.n, [item for sublist in inverse_matrix for item in sublist])

    def __add__(self, other):
        if self.m != other.m or self.n != other.n:
            raise MatrixError("Matrix dimensions must match for addition.")
        result_entries = [self.matrix[i][j] + other.matrix[i][j] for i in range(self.m) for j in range(self.n)]
        return Matrix(self.m, self.n, result_entries)

    def __mul__(self, other):
        if self.n != other.m:
            raise MatrixError("The number of columns in the first matrix must equal the number of rows in the second.")
        result_matrix = [[sum(self.matrix[i][k] * other.matrix[k][j] for k in range(self.n)) for j in range(other.n)] for i in range(self.m)]
        return Matrix(self.m, other.n, [item for sublist in result_matrix for item in sublist])
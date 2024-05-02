# Characteristic Equation and Eigenvalues/Eigenvectors


def dot_product(v1, v2):
    """
    Calculate the dot product of two vectors.
    """
    return sum(x*y for x, y in zip(v1, v2))

def matrix_multiply(A, B):
    """
    Multiply two matrices A and B.
    """
    return [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def transpose(matrix):
    """
    Calculate the transpose of a matrix.
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def gaussian_elimination(matrix):
    """
    Perform Gaussian elimination on a matrix.

    Gaussian elimination is a method for solving systems of linear equations by
    transforming the augmented matrix into row echelon form.

    Reference:
    Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press.
    """
    n = len(matrix)
    for i in range(n):
        pivot = matrix[i][i]
        for j in range(i+1, n):
            factor = matrix[j][i] / pivot
            for k in range(i, n+1):
                matrix[j][k] -= factor * matrix[i][k]
    return matrix

def back_substitution(matrix):
    """
    Perform back substitution on a matrix in row echelon form.
    """
    n = len(matrix)
    solution = [0] * n
    for i in range(n-1, -1, -1):
        solution[i] = (matrix[i][-1] - sum(matrix[i][j] * solution[j] for j in range(i+1, n))) / matrix[i][i]
    return solution

def characteristic_equation(matrix):
    """
    Compute the characteristic equation of a matrix.

    The characteristic equation of a square matrix A is det(A - λI) = 0, where λ represents
    the eigenvalues and I is the identity matrix.

    Reference:
    Strang, G. (2016). Introduction to linear algebra. Wellesley-Cambridge Press.
    """
    n = len(matrix)
    identity = [[float(i==j) for j in range(n)] for i in range(n)]
    characteristic_matrix = [[matrix[i][j] - identity[i][j] for j in range(n)] + [0] for i in range(n)]
    characteristic_matrix[-1][-1] = 1
    return gaussian_elimination(characteristic_matrix)

def eigenvalues(char_eq):
    """
    Compute the eigenvalues from the characteristic equation.
    """
    return [row[-1] for row in char_eq[:-1]]

def eigenvectors(matrix, eigenvalue):
    """
    Compute the eigenvectors for a given eigenvalue.
    """
    n = len(matrix)
    identity = [[float(i==j) for j in range(n)] for i in range(n)]
    eigenvector_matrix = [[matrix[i][j] - eigenvalue * identity[i][j] for j in range(n)] + [0] for i in range(n)]
    eigenvector_matrix[-1] = [1] * (n+1)
    eliminated_matrix = gaussian_elimination(eigenvector_matrix)
    return back_substitution(eliminated_matrix)

def is_diagonalizable(eigenvalues, eigenvectors):
    """
    Check if the matrix is diagonalizable.

    A matrix is diagonalizable if it has a full set of linearly independent eigenvectors.
    """
    return len(set(eigenvalues)) == len(eigenvalues) and all(eigenvectors)

def main():
    n = int(input("Enter the size of the square matrix (n): "))
    print("Enter the matrix entries:")
    matrix = [[float(input(f"Enter element [{i+1},{j+1}]: ")) for j in range(n)] for i in range(n)]

    char_eq = characteristic_equation(matrix)
    eigenvalues_list = eigenvalues(char_eq)

    print("\nCharacteristic Equation:")
    for row in char_eq:
        print([round(x, 4) for x in row])

    print("\nEigenvalues:")
    print([round(x, 4) for x in eigenvalues_list])

    eigenvectors_list = [eigenvectors(matrix, eigenvalue) for eigenvalue in eigenvalues_list]

    print("\nEigenvectors:")
    for eigenvector in eigenvectors_list:
        print([round(x, 4) for x in eigenvector])

    if is_diagonalizable(eigenvalues_list, eigenvectors_list):
        print("\nThe matrix is diagonalizable.")
        change_of_basis = transpose(eigenvectors_list)
        print("\nChange of Basis Matrix:")
        for row in change_of_basis:
            print([round(x, 4) for x in row])
    else:
        print("\nThe matrix is not diagonalizable.")
        print("Eigenvalues with algebraic multiplicity > geometric multiplicity:")
        for eigenvalue, eigenvector in zip(eigenvalues_list, eigenvectors_list):
            if eigenvector[0] == 0:
                print(round(eigenvalue, 4))

if __name__ == "__main__":
    main()
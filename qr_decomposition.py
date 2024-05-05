# QR Decomposition


def dot_product(v1, v2):
    """
    Calculate the dot product of two vectors.
    """
    return sum(x*y for x, y in zip(v1, v2))

def norm(v):
    """
    Calculate the Euclidean norm of a vector.
    """
    return dot_product(v, v) ** 0.5

def projection(v1, v2):
    """
    Calculate the projection of v1 onto v2.
    """
    return [x * (dot_product(v1, v2) / dot_product(v2, v2)) for x in v2]

def gram_schmidt(matrix):
    """
    Perform Gram-Schmidt orthogonalization on a matrix.

    The Gram-Schmidt orthogonalization process is used to convert a set of linearly
    independent vectors into an orthonormal set of vectors spanning the same subspace.

    Reference:
    Trefethen, L. N., & Bau III, D. (1997). Numerical linear algebra. SIAM.
    """
    Q = []
    for v in matrix:
        w = v
        for u in Q:
            w = [x - y for x, y in zip(w, projection(v, u))]
        Q.append([x / norm(w) for x in w])
    return Q

def qr_decomposition(matrix):
    """
    Perform QR decomposition on a matrix.

    The QR decomposition factorizes a matrix A into a product A = QR, where Q is an
    orthogonal matrix and R is an upper triangular matrix.

    Reference:
    Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press.
    """
    Q = gram_schmidt(matrix)
    R = [[dot_product(q, v) for v in matrix] for q in Q]
    return Q, R

def main():
    m = int(input("Enter the number of rows (m): "))
    n = int(input("Enter the number of columns (n): "))
    print("Enter the matrix entries:")
    matrix = [[float(input(f"Enter element [{i+1},{j+1}]: ")) for j in range(n)] for i in range(m)]

    Q, R = qr_decomposition(matrix)

    print("\nMatrix Q:")
    for row in Q:
        print([round(x, 4) for x in row])

    print("\nMatrix R:")
    for row in R:
        print([round(x, 4) for x in row])

if __name__ == "__main__":
    main()
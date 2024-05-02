import math

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def vector_norm(v):
    return math.sqrt(dot_product(v, v))

def matrix_multiply(A, B):
    return [[dot_product(row, col) for col in zip(*B)] for row in A]

def transpose(A):
    return list(map(list, zip(*A)))

def vector_multiply(v, k):
    return [x * k for x in v]

def vector_add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]

def vector_subtract(v1, v2):
    return [x - y for x, y in zip(v1, v2)]

def householder_reflection(A, i):
    x = [row[i] for row in A[i:]]
    e = [0] * len(x)
    e[0] = vector_norm(x)
    u = vector_add(x, vector_multiply(e, -math.copysign(1, x[0])))
    v = vector_multiply(u, 1 / vector_norm(u))
    P = [[float(i == j) for j in range(len(A))] for i in range(len(A))]
    for i in range(len(v)):
        for j in range(len(v)):
            P[i+len(A)-len(v)][j+len(A)-len(v)] -= 2 * v[i] * v[j]
    return P

def bidiagonalize(A):
    m, n = len(A), len(A[0])
    U = [[float(i == j) for j in range(m)] for i in range(m)]
    V = [[float(i == j) for j in range(n)] for i in range(n)]
    for i in range(min(m, n)):
        P = householder_reflection(A, i)
        A = matrix_multiply(P, A)
        U = matrix_multiply(U, transpose(P))
        if i < n - 2:
            P = householder_reflection(transpose(A), i)
            A = matrix_multiply(A, transpose(P))
            V = matrix_multiply(V, transpose(P))
    return U, A, V

def svd(A, max_iterations=100, tolerance=1e-6):
    m, n = len(A), len(A[0])
    U, B, V = bidiagonalize(A)
    for _ in range(max_iterations):
        if all(abs(B[i][j]) < tolerance for i in range(m) for j in range(n) if i != j):
            break
        Q, R = qr_decomposition(transpose(B), tolerance)
        B = transpose(R)
        V = matrix_multiply(V, Q)
        Q, R = qr_decomposition(B, tolerance)
        B = R
        U = matrix_multiply(U, Q)
    S = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(min(m, n)):
        S[i][i] = B[i][i]
    return U, S, transpose(V)

def qr_decomposition(A, tolerance=1e-6):
    m, n = len(A), len(A[0])
    Q = [[float(i == j) for j in range(m)] for i in range(m)]
    R = [row[:] for row in A]
    for i in range(n):
        v = [R[j][i] for j in range(i, m)]
        norm_v = vector_norm(v)
        if abs(norm_v) < tolerance:
            continue
        v = vector_multiply(v, 1 / norm_v)
        for j in range(i, m):
            R[j][i] -= 2 * dot_product(v, [R[j][k] for k in range(i, n)]) * v[j-i]
            for k in range(i+1, n):
                R[j][k] -= 2 * v[j-i] * dot_product(v, [R[l][k] for l in range(i, m)])
        for j in range(m):
            for k in range(i, m):
                Q[j][k] -= 2 * dot_product(v, [Q[j][l] for l in range(i, m)]) * v[k-i]
    return Q, R

def main():
    m = int(input("Enter the number of rows (m): "))
    n = int(input("Enter the number of columns (n): "))
    print(f"Enter {m * n} matrix entries:")
    A = [[float(input()) for _ in range(n)] for _ in range(m)]
    U, S, V = svd(A)
    print("U =")
    for row in U:
        print(row)
    print("S =")
    for row in S:
        print(row)
    print("V =")
    for row in V:
        print(row)

if __name__ == "__main__":
    main()
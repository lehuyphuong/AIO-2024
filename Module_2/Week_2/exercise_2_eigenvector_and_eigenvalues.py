import numpy as np
import sympy as sp


def compute_eigenvalues_eigenvectors(matrix):

    # Convert the input matrix to a numpy array
    A = np.array(matrix)

    # Calculate eigen_values and eigen_vectors using numpy
    eigen_values, eigen_vectors = np.linalg.eig(A)

    return eigen_values, eigen_vectors


if __name__ == "__main__":
    A = [[0.9, 0.2],
         [0.1, 0.8]]

    result = compute_eigenvalues_eigenvectors(A)
    print(result)

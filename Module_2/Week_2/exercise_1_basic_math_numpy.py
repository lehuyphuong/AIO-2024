import numpy as np


def compute_vector_length(vector):
    return np.sqrt(np.sum(vector * vector))


def compute_dot_product(vector1, vector2):
    return np.sum(vector1 * vector2)


def matrix_multi_vector(matrix, vector):
    return np.dot(matrix, vector)


def matrix_multi_matrix(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


def inverse_matrix(matrix):

    a, b = matrix[0]
    c, d = matrix[1]
    det_matrix = a * d - b * c

    if det_matrix == 0:
        raise ValueError("det matrix is equal to zero, unable to compute")

    new_mat = [[d / det_matrix, -b / det_matrix],
               [-c / det_matrix, a / det_matrix]]

    return new_mat


if __name__ == "__main__":
    vector = np.array([-2, 4, 9, 21])
    compute_vector_length_result = compute_vector_length(vector)
    print(compute_vector_length_result)

    v1 = np.array([0, 1, -1, 2])
    v2 = np.array([2, 5, 1, 0])
    compute_dot_product_result = compute_dot_product(v1, v2)
    print(compute_dot_product_result)

    m = np.array([[-1, 1, 1], [0, -4, 9]])
    v = np.array([0, 2, 1])
    matrix_multi_vector_result = matrix_multi_vector(m, v)
    print(matrix_multi_vector_result)

    m1 = np.array([[0, 1, 2], [2, -3, 1]])
    m2 = np.array([[1, -3], [6, 1], [0, -1]])
    matrix_multi_matrix_result = matrix_multi_matrix(m1, m2)
    print(matrix_multi_matrix_result)

    m1 = np.array([[-2, 6],
                   [8, -4]])
    inverse_matrix_result = inverse_matrix(m1)
    print(inverse_matrix_result)

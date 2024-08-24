import numpy as np


def compute_vector_length(vector):
    return np.sqrt(np.sum(vector * vector))


def compute_cosine(v1, v2):

    assert len(v1) == len(v2), " shape between arrays are not equal"
    sum_of_product = 0

    for i in range(len(v1)):
        sum_of_product += v1[i]*v2[i]

    len_v1 = compute_vector_length(v1)
    len_v2 = compute_vector_length(v2)

    cos_sim = sum_of_product/(len_v1*len_v2)
    return cos_sim


if __name__ == "__main__":
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    result = compute_cosine(x, y)
    print(result)

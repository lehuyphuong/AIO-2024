import numpy as np


def maxx(x, y):
    if x >= y:
        return x
    else:
        return y


if __name__ == "__main__":
    # Exercise 1: Initialize array from 0 to 9
    print("\n Exercise 1: Initialize array from 0 to 9")
    arr = np.arange(0, 10, 1)
    print(arr)
    # -> Answer: A

    # Exercise 2: Initialize bollean matrix 3x3 all value True
    print("\n Exercise 2: Initialize bollean matrix 3x3 all value True")
    arr = np.ones((3, 3)) > 0
    print(arr)

    arr = np.ones((3, 3), dtype=bool)
    print(arr)

    arr = np.full((3, 3), fill_value=True, dtype=bool)
    print(arr)
    # -> Answer: D

    # Exercise 3: What is the answer for below code?
    print("\n Exercise 3: What is the answer for below code?")
    arr = np.arange(0, 10)
    print(arr[arr % 2 == 1])
    # -> Answer: A

    # Excercise 4: What is the answer for below code?
    print("\n Excercise 4: What is the answer for below code?")
    arr = np.arange(0, 10)
    arr[arr % 2 == 1] = -1
    print(arr)
    # -> Answer: B

    # Exercise 5: What is the answer for below code?
    print("\n Excercise 5: What is the answer for below code?")
    arr = np.arange(10)
    arr_2d = arr.reshape(2, -1)
    print(arr_2d)
    # -> Answer: B

    # Exercise 6: What is the answer for below code?
    print("\n Excercise 6: What is the answer for below code?")
    arr1 = np.arange(10).reshape(2, -1)
    arr2 = np.repeat(1, 10).reshape(2, -1)
    c = np.concatenate([arr1, arr2], axis=0)
    print(c)
    # -> Answer: A

    # Exercise 7: What is the answer for below code?
    print("\n Excercise 7: What is the answer for below code?")
    arr1 = np.arange(10).reshape(2, -1)
    arr2 = np.repeat(1, 10).reshape(2, -1)
    c = np.concatenate([arr1, arr2], axis=1)
    print(c)
    # -> Answer: C

    # Exercise 8: What is the answer for below code?
    print("\n Excercise 8: What is the answer for below code?")
    arr = np.array([1, 2, 3])
    print(np.repeat(arr, 3))
    print(np.tile(arr, 3))
    # -> Answer: A

    # Exercise 9: What is the answer for below code?
    print("\n Excercise 9: What is the answer for below code?")
    a = np.array([2, 6, 1, 9, 10, 3, 27])
    index = np.nonzero((a >= 5) & (a <= 10))
    print(a[index])
    # -> Answer: C

    # Exercise 10: What is the answer for below code?
    print("\n Excercise 10: What is the answer for below code?")
    a = np.array([5, 7, 9, 8, 6, 4, 5])
    b = np.array([6, 3, 4, 8, 9, 7, 1])
    pair_max = np.vectorize(maxx, otypes=[float])
    print(pair_max(a, b))
    # -> Answer: D

    # Exercise 11: What is the answer for below code?
    print("\n Excercise 11: What is the answer for below code?")
    a = np.array([5, 7, 9, 8, 6, 4, 5])
    b = np.array([6, 3, 4, 8, 9, 7, 1])
    print(np.where(a < b, b, a))
    # -> Answer: A

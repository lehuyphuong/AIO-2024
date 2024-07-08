def is_float(prompt):
    while True:
        try:
            input_user = float(input(prompt))
            return input_user
        except ValueError:
            print("input must be a float number, please try again!")


def is_positive_int(prompt):
    while True:
        try:
            input_user = int(input(prompt))
            if input_user > 0:
                return input_user
        except ValueError:
            print("input must be a positve integer number, please try again!")


def cal_factorial(x):
    """
    x: int - number to calculate factorial
    """
    if x == 0:
        return 1
    else:
        return x * cal_factorial(x - 1)


def approx_sin(x, n):
    """
    x: float - radiant number
    n: int - number of repeats
    """
    result = 0
    for i in range(0, n):
        result += (((-1) ** i) * (x ** (2 * i + 1))) / (
            cal_factorial(2 * i + 1))

    return result


def approx_cos(x, n):
    """
    x: float - randiant number
    n: int - number of repeats
    """
    result = 0
    for i in range(0, n):
        result += (((-1) ** i) * (x ** (2 * i))) / (cal_factorial(2 * i))

    return result


def approx_sinh(x, n):
    """
    x: float - randiant number
    n: int - number of repeats
    """
    result = 0
    for i in range(0, n):
        result += (x ** (2 * i + 1))/(cal_factorial(2 * i + 1))

    return result


def approx_cosh(x, n):
    """
    x: float - randiant number
    n: int - number of repeats
    """
    result = 0
    for i in range(0, n):
        result += (x ** (2 * i)) / (cal_factorial(2 * i))

    return result


if __name__ == "__main__":

    # According to request 4, x: radiant, n: positive integer number
    x = is_float("Enter a radian number: ")
    n = is_positive_int("Enter a positive integer number: ")

    print("Estimate value sin({}) = {}".format(x, approx_sin(x, n)))
    print("Estimate value cos({}) = {}".format(x, approx_cos(x, n)))
    print("Estimate value sinh({}) = {}".format(x, approx_sinh(x, n)))
    print("Estimate value cosh({}) = {}".format(x, approx_cosh(x, n)))

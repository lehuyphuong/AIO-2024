def is_float(prompt):
    while True:
        try:
            input_user = float(input(prompt))
            return input_user
        except ValueError:
            print("input must be a float number, please try again")


def is_positive_int(prompt):
    while True:
        try:
            input_user = int(input(prompt))
            if input_user > 0:
                return input_user
        except ValueError:
            print("input must be a positve integer number, please try again")


def cal_mean_diff_n_root_error(y, y_hat, n, p):
    """
    y: float - predicted value
    y_hat: float - target value
    n: int - level of root
    p: int - level of loss
    """
    result = (y ** (1 / n) - y_hat ** (1 / n)) ** p

    return result


if __name__ == "__main__":
    """
    y: float - predicted value
    y_hat: float - target value
    n: int - level of root
    p: int - level of loss
    """
    y = is_float("Enter a predicted value: ")
    y_hat = is_float("Enter a target value: ")

    n = is_positive_int("Enter n(th) root: ")
    p = is_positive_int("Enter p(th) loss: ")

    result = cal_mean_diff_n_root_error(y, y_hat, n, p)
    print("Value of MD_nRE in this scope would be: {}".format(result))

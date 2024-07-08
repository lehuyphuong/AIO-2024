import math
import random


def is_number(n):
    try:
        int(n)
    except ValueError:
        return False
    return True


def cal_mean_abs_error(n):
    """
    This function is created for calculating mean absolute error
    """

    # Force input to make sure n sorely belong to integer type
    n = int(n)

    # random.uniform help generate float number
    predict = random.uniform(0, 10)
    target = random.uniform(0, 10)

    sum = 0

    for _ in range(0, n):
        sum += abs(predict - target)

    loss = sum / n
    return n, predict, target, loss


def cal_mean_squa_error(n):
    """
    This function is created for calculating mean squared error
    """

    # Force input to make sure n sorely belong to integer type
    n = int(n)

    # random.uniform help generate float number
    predict = random.uniform(0, 10)
    target = random.uniform(0, 10)

    sum = 0

    for _ in range(0, n):
        sum += math.pow(predict - target, 2)

    loss = sum / n
    return n, predict, target, loss


def cal_root_mean_squa_err(n):
    """
    This function is created for calculating root mean squared error
    """

    # Force input to make sure n sorely belong to integer type
    n = int(n)

    # random.uniform help generate float number
    predict = random.uniform(0, 10)
    target = random.uniform(0, 10)

    sum = 0

    for _ in range(0, n):
        sum += math.pow(predict - target, 2)

    loss = math.sqrt(sum / n)
    return n, predict, target, loss


if __name__ == "__main__":
    num_of_samples = input("Enter number of samples: ")

    if is_number(num_of_samples):
        type_loss_func = input("Input loss function name(MAE|MSE|RMSE): ")

        if type_loss_func == "MAE" or "mae":
            sample, pred, target, loss = cal_mean_abs_error(num_of_samples)
            print("loss name: MAE, sample: {}, pred: {}, target: {}, loss: {}"
                  .format(sample, pred, target, loss))
        elif type_loss_func == "MSE" or "mse":
            sample, pred, target, loss = cal_mean_squa_error(num_of_samples)
            print("loss name: MSE, sample: {}, pred: {}, target: {}, loss: {}"
                  .format(sample, pred, target, loss))
        elif type_loss_func == "RMSE" or "rmse":
            sample, pred, target, loss = cal_root_mean_squa_err(num_of_samples)
            print("loss name: RMSE, sample: {}, pred: {}, target: {}, loss: {}"
                  .format(sample, pred, target, loss))
        else:
            print("Activation " + type_loss_func + " is not supported")

    else:
        print("Invalid input")

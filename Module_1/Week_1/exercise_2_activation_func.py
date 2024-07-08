import math


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def sigmoid(x):
    x = float(x)
    sigmoid_value = 1 / (1 + math.exp(-x))
    return sigmoid_value


def relu(x):
    x = float(x)
    if x <= 0:
        return 0
    else:
        return x


def elu(x, alpha):
    x = float(x)
    if x <= 0:
        return alpha * (math.exp(x) - 1)
    else:
        return x


if __name__ == "__main__":
    input_user = input("Enter a number: ")
    if is_number(input_user):
        type_activation_func = input("Activation function (sigmoid|relu|elu):")

        if type_activation_func == "sigmoid":
            print("Sigmoid: f({}) = {}".format(input_user,
                                               sigmoid(input_user)))
        elif type_activation_func == "relu":
            print("Relu: f({}) = {}".format(input_user,
                                            relu(input_user)))
        elif type_activation_func == "elu":
            alpha = float(input("Enter the value of alpha: "))
            print("Elu: f({}) = {}".format(input_user,
                                           elu(input_user, alpha)))
        else:
            print("Activation " + type_activation_func + " is not supported")
    else:
        print("Invalid input")

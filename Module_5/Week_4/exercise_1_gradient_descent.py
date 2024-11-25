import numpy as np


def df_w(W):
    """ Compute gradient for dw1 and dw2 w.r.t w1 and w3 accordingly 
    Argument:
    W -- np.array [w1, w2]
    Return:
    dW -- np.array [dw1, dw2]
    """

    dW = np.array([0.1*W[0]*2, 2*W[1]*2])

    return dW


def sgd(W, dw, lr):
    """ Use gradient descent to update w1 and w2 
    Arguments:
    W -- np.array:[w1, w2]
    dW -- np.array: [dw1, dw2]
    lr -- float: learning rate
    Returns:
    W -- np.array: [w1, w2]
    """
    W = W - lr*dw
    return W


def train_p1(optimizer, lr, epochs):
    """ Find global minimum of a function based on optimizer algorithm 
    Arguments:
    optimizer : optimization function
    lr -- float: learning rate
    epoch -- int: number of iteration for finding minimum value
    Returns:
    results -- list: list w1 and w2 updated every epoch
    """
    # Initialize point:
    W = np.array([-5, -2], dtype=np.float32)
    results = [W]

    for _ in range(epochs):
        dw = df_w(W)

        W = optimizer(W, dw, lr)

        results.append(W)

    return results


if __name__ == "__main__":
    result = np.array(train_p1(sgd, 0.4, 30), dtype=np.float32)
    print(result)

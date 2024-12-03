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


def sgd_and_momentum(W, dw, lr, momentum, previous_change):
    """ Use gradient descent to update w1 and w2 
    Arguments:
    W -- np.array:[w1, w2]
    dW -- np.array: [dw1, dw2]
    lr -- float: learning rate
    momentum -- float: momentum value
    preivous change -- float: latest change (for creating momentum)
    Returns:
    W -- np.array: [w1, w2]
    current_change -- float: update current change 
    """
    current_change = momentum*previous_change + (1-momentum)*dw

    W = W - lr*current_change
    return W, current_change


def train_p2(optimizer, lr, epochs, momentum=0.9):
    """ Find global minimum of a function based on optimizer algorithm 
    Arguments:
    optimizer : optimization function
    lr -- float: learning rate
    epoch -- int: number of iteration for finding minimum value
    momentum -- float: momentum value (default : 0.9)
    Returns:
    results -- list: list w1 and w2 updated every epoch
    """
    # Initialize point:
    previous_change = 0.0
    W = np.array([-5, -2], dtype=np.float32)
    results = [W]

    for _ in range(epochs):
        dw = df_w(W)

        W, current_change = optimizer(W, dw, lr, momentum, previous_change)

        previous_change = current_change
        results.append(W)

    return results


if __name__ == "__main__":
    result = np.array(train_p2(sgd_and_momentum, 0.6,
                      30, 0.5), dtype=np.float32)
    print(result)

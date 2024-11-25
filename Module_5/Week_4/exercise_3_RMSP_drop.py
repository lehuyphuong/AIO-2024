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


def RMSP_drop(W, dw, lr, decay_rate, previous_change):
    """ Use gradient descent to update w1 and w2 
    Arguments:
    W -- np.array:[w1, w2]
    dW -- np.array: [dw1, dw2]
    lr -- float: learning rate
    decay_rate -- float: decay_rate value
    preivous change -- float: latest change (for creating momentum)
    Returns:
    W -- np.array: [w1, w2]
    current_change -- float: update current change 
    """
    current_change = decay_rate*previous_change + (1-decay_rate)*(dw*dw)

    # 1e-6 as constant value to avoid zero-division
    W = W - lr*(dw/np.sqrt(current_change) + 1e-6)
    return W, current_change


def train_p3(optimizer, lr, epochs, decay_rate=0.9):
    """ Find global minimum of a function based on optimizer algorithm 
    Arguments:
    optimizer : optimization function
    lr -- float: learning rate
    epoch -- int: number of iteration for finding minimum value
    decay_rate -- float: decay_rate value (default : 0.9)
    Returns:
    results -- list: list w1 and w2 updated every epoch
    """
    # Initialize point:
    previous_change = 0.0
    W = np.array([-5, -2], dtype=np.float32)
    results = [W]

    for _ in range(epochs):
        dw = df_w(W)

        W, current_change = optimizer(W, dw, lr, decay_rate, previous_change)

        previous_change = current_change
        results.append(W)

    return results


if __name__ == "__main__":
    result = np.array(train_p3(RMSP_drop, 0.3,
                      30, 0.9), dtype=np.float32)
    print(result)

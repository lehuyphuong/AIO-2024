import numpy as np

t = 0


def df_w(W):
    """ Compute gradient for dw1 and dw2 w.r.t w1 and w3 accordingly 
    Argument:
    W -- np.array [w1, w2]
    Return:
    dW -- np.array [dw1, dw2]
    """

    dW = np.array([0.1*W[0]*2, 2*W[1]*2])

    return dW


def adam(W, dw, lr, decay_rate_1, decay_rate_2, previous_v, previous_s):
    """ Use gradient descent to update w1 and w2 
    Arguments:
    W -- np.array:[w1, w2]
    dW -- np.array: [dw1, dw2]
    lr -- float: learning rate
    decay_rate_1 -- float: decay_rate_1 value
    decay_rate_2 -- float: decay_rate_2 value
    previous_v -- float: exponentially decaying average of past gradients (for creating momentum)
    previous_s -- float: exponentially decaying average of past squared gradients
    Returns:
    W -- np.array: [w1, w2]
    current_v -- float: update current change 
    current_s -- float: update current change 
    """
    global t
    current_v = decay_rate_1*previous_v + (1-decay_rate_1)*dw
    current_s = decay_rate_2*previous_s + (1-decay_rate_2)*(dw*dw)

    v_bias_correction = current_v/(1-decay_rate_1**t)
    s_bias_correction = current_s/(1-decay_rate_2**t)

    # 1e-6 as constant value to avoid zero-division
    W = W - lr*(v_bias_correction/np.sqrt(s_bias_correction) + 1e-6)
    return W, current_v, current_s


def train_p4(optimizer, lr, epochs, decay_rate_1=0.999, decay_rate_2=0.999):
    """ Find global minimum of a function based on optimizer algorithm 
    Arguments:
    optimizer : optimization function
    lr -- float: learning rate
    epoch -- int: number of iteration for finding minimum value
    decay_rate_1 -- float: decay_rate_1 value (default : 0.9)
    decay_rate_2 -- float: decay_rate_2 value (default : 0.9)
    Returns:
    results -- list: list w1 and w2 updated every epoch
    """
    global t
    t = 1
    # Initialize point:
    previous_v = 0.0
    previous_s = 0.0
    W = np.array([-5, -2], dtype=np.float32)
    results = [W]

    for epoch in range(epochs):
        t = epoch + 1
        dw = df_w(W)

        W, current_v, current_s = optimizer(
            W, dw, lr, decay_rate_1, decay_rate_2, previous_v, previous_s)

        previous_v = current_v
        previous_s = current_s
        results.append(W)

    return results


if __name__ == "__main__":
    result = np.array(train_p4(adam, 0.2,
                      30), dtype=np.float32)
    print(result)

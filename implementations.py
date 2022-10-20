import numpy as np

def compute_loss(y, tx, w):

    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, M)
        w: numpy array of shape=(M,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx@w
    n = y.shape[0]
    
    return e.T@e/(2*n)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    n = y.shape[0]

    return -tx.T @ e / n


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    return w, loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    e = y - tx @ w
    n = y.shape[0]

    return -tx.T @ e / n


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: implement stochastic gradient descent.
        # ***************************************************
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient
    return w, loss


def least_squares(y, tx):
    """Calculate the optimal vector w from the least squares regression, using the normal equations

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, M)

    Returns:
        w: the optimal model parameters resulting from the least squares regression
        loss: the MSE loss for the model parameters w.r. to y and tx
    """
    # Normal equations : w* = inverse(tx.T @ tx) @ tx.T @ y -> (tx.T @ tx) @ w* = tx.T @ y
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)  # Aw = b

    loss = compute_loss(y, tx, w)

    return (w, loss)

def least_squares(y, tx):
    """Calculate the optimal vector w from the least squares regression, using the normal equations
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, M)
        
    Returns:
        w: the optimal model parameters resulting from the least squares regression
        loss: the MSE loss for the model parameters w.r. to y and tx
    """
    # Normal equations : w* = inverse(tx.T @ tx) @ tx.T @ y -> (tx.T @ tx) @ w* = tx.T @ y
    A = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(A, b) # Aw = b
    
    loss = compute_loss(y, tx, w)
    
    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """Compute ridge regression using normal equations
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss : the MSE loss for the model parameters w.r. to y and tx
    """
    # Normal equations : w*_ridge = inverse(tx.T @ tx + lambda*I) @ tx.T @ y -> (tx.T @ tx + lambda_prime*I) @ w* = tx.T @ y
    n = y.shape[0]
    d = tx.shape[1]
    lambda_prime = 2*n*lambda_
    
    A = tx.T@tx + lambda_prime*np.identity(d)
    b = tx.T@y
    w = np.linalg.solve(A, b)
    
    loss = compute_loss(y, tx, w) + lambda_*np.sum(w**2)
    
    return (w, loss)
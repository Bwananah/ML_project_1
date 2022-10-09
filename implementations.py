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
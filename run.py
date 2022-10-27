import numpy as np
import matplotlib.pyplot as plt
import csv

DATA_PATH = "../data/"
N = 32 # number of columns in data
ratio = 0.9 # ratio of data to use for training
seed = 1 # random seed
d = 10 # degree of polynomial feature expansion, found by running the model with different d, and choosing the best validation loss


# -- LOAD DATA
def load_one_column(filename, column, data_type='float'):
    """load one column from data"""
    data = np.loadtxt(DATA_PATH + filename, delimiter=",", skiprows=1, unpack=True, usecols=[column], dtype=data_type)
    return data

def load_features(filename):
    """load data features"""
    data = np.loadtxt(DATA_PATH + filename, delimiter=",", skiprows=1, unpack=True, usecols=np.arange(2, N))
    return data

def build_model_data(y, x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    fun = lambda e : 0 if e == 'b' else 1 # convert b to 0 and s to 1
    y = [fun(e) for e in y]
    tx = np.c_[np.ones(num_samples), x.T]
    return np.array(y), tx

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


# -- FEATURE EXPANSION

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
        
    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    n = x.shape[0]
    res = np.zeros((n, degree+1))
    
    for i in range(n):
        for j in range(degree+1):
            res[i, j] = x[i]**j
            
    return res

def feature_expansion(tx):
    """ Add polynomial feature expansion of degree d for EVERY feature to tx"""
    poly = build_poly(tx[:, 0], d)
    for i in range(1, N-1):
        poly = np.hstack((poly, build_poly(tx[:, i], d)[:, 1:])) # ommit first colum (ones), so we don't have multiple 1 columns
        
    return poly


# -- SPLIT DATA

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
        
    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    
    n = y.shape[0]
    nb_tr = int(np.floor(n * ratio))
    
    indices = np.random.permutation(np.arange(n))
    
    x_tr = x[indices[:nb_tr]]
    x_te = x[indices[nb_tr:]]
    y_tr = y[indices[:nb_tr]]
    y_te = y[indices[nb_tr:]]
    
    return x_tr, x_te, y_tr, y_te


# -- ML METHODS

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


# -- CREATE SUBMISSION

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


# -- SCRIPT

# build training model
training_predictions = load_one_column("train.csv", 1, 'str')
training_features = load_features("train.csv")
training_y, training_tx = build_model_data(training_predictions, training_features)
training_tx_std = standardize(training_tx)[0]

# build model for predictions
test_ids = load_one_column("test.csv", 0)
test_features = load_features("test.csv")
_, test_tx = build_model_data(test_ids, test_features)
test_tx_std = standardize(test_tx)[0]

# Feature expansion
training_tx_poly = feature_expansion(training_tx_std)
test_tx_poly = feature_expansion(test_tx_std)

# Split training data into training and validation
tx_tr, tx_val, y_tr, y_val = split_data(training_tx_poly, training_y, ratio, seed)

# Get best lambda
lambdas = np.linspace(0.003, 0.005, 100) # bounds found by trying over and over and plotting the results
results = [compute_loss(y_val, tx_val, ridge_regression(y_tr, tx_tr, e)[0]) for e in lambdas]
best_lambda = lambdas[np.argmin(results)]

# Get best w
best_w_ridge, loss_train = ridge_regression(y_tr, tx_tr, best_lambda) # use all data now that we found best

# Check validation loss (we used this to fine tune our parameters)
validation_loss = compute_loss(y_val, tx_val, best_w_ridge)

# Now we know we have the best parameters -> train model on all the available data (no more training vs validation)
w, loss = ridge_regression(training_y, training_tx_poly, best_lambda)

# Create submission file
y_pred = [-1 if e < 0.5 else 1 for e in test_tx_poly@w]
create_csv_submission(test_ids, y_pred, "submit_ridge.csv")




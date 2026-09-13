"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    n = X.shape[0]
    rng = np.random.default_rng(seed=seed)
    shuffle_index = rng.permutation(n)

    X_shuffled = X[shuffle_index]
    Y_shuffled = y[shuffle_index]

    return X_shuffled, Y_shuffled

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    
    n_sample = X.shape[0]

    n_train = int(n_sample * train_frac)
    n_val = int(n_sample * val_frac)
    val_end = n_train + n_val

    X_train, y_train = X[:n_train], y[:n_train]
    
    X_val, y_val = X[n_train:val_end], y[n_train:val_end]
    
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    std = np.where(std == 0, 1, std)

    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    return (X - mean) / std

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.

    X_standardized = (X - mean) / std

    n_samples = X.shape[0]
    bias = np.ones((n_samples, 1), dtype=float)

    return np.hstack([bias, X_standardized])

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    
    return np.mean((y_true - y_pred) ** 2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    return  X.T @ (y_pred - y_true) * 2 / X.shape[0]

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    
    A = X.T @ X
    B = X.T @ y

    return np.linalg.solve(A, B)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0, 0.01, n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    y_pred = predict_linear(X, weights)
    grad = mse_gradient(X, y, y_pred)

    weights -= lr * grad

    return weights

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats

    y_pred_train = predict_linear(X_train, weights)
    y_pred_val = predict_linear(X_val, weights)

    train_loss = mse_loss(y_train, y_pred_train)
    val_loss = mse_loss(y_val, y_pred_val)
    return train_loss, val_loss

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    wait += 1

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        best_weights = weights.copy()


    is_stop = (wait == patience)

    return best_val_loss, wait, best_weights, is_stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    
    weights = initialize_weights(n_features = n_features, seed = seed)
    best_weights = weights.copy()
    best_val_loss = np.inf
    wait = 0
    train_losses, val_losses = [], []
    stopped = False

    return {
        'weights': weights,
        'best_weights': best_weights,
        'best_val_loss': best_val_loss,
        'wait': wait,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'stopped': stopped
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...
    
    if state['stopped']:
        return state
    
    weights = state['weights']
    best_weights = state['best_weights']
    best_val_loss = state['best_val_loss']
    wait = state['wait']
    train_losses = state['train_losses']
    val_losses = state['val_losses']
    
    weights = gd_step(X_train, y_train, weights, lr)
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, weights)
    state['train_losses'].append(train_loss)
    state['val_losses'].append(val_loss)

    best_val_loss, wait, best_weights, stopped = update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience)


    state["weights"] = weights
    state["best_weights"] = best_weights
    state["best_val_loss"] = best_val_loss
    state["wait"] = wait
    state["stopped"] = stopped

    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    # TODO: Train weights with full-batch GD for up to epochs, with early stopping.
    
    state = init_training_state(X_train.shape[1], seed)

    for epoch in range(epochs):
        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        if state['stopped']:
            break
    
    return state['best_weights'], state['train_losses'], state['val_losses']

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: Compute the mean absolute error between true targets and predictions
    
    return np.mean(np.abs(y_true - y_pred))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # TODO: Return the root mean squared error between y_true and y_pred.

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute the coefficient of determination R^2.
    
    y_hat = np.mean(y_true)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_hat) ** 2)

    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / (ss_tot + 1e-8)

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, and R^2 into one metrics dictionary for test-set reporting.
    
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred)
    }

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement


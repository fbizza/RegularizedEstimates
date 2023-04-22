from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

NUM_EXPERIMENTS = 100
N_SAMPLES = 100
N_FEATURES = 50

alpha_lasso = 0.1
alpha_ridge = 0.1

# To store the results
mse_lasso = np.zeros(NUM_EXPERIMENTS)
mse_ridge = np.zeros(NUM_EXPERIMENTS)

for i in range(NUM_EXPERIMENTS):

    # Data generation
    X, y, coef = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, coef=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lasso = Lasso(alpha=alpha_lasso)
    lasso.fit(X_train, y_train)

    ridge = Ridge(alpha=alpha_ridge)
    ridge.fit(X_train, y_train)

    # Evaluation
    mse_lasso[i] = mean_squared_error(y_test, lasso.predict(X_test))
    mse_ridge[i] = mean_squared_error(y_test, ridge.predict(X_test))


avg_mse_lasso = np.mean(mse_lasso)
avg_mse_ridge = np.mean(mse_ridge)

# Print the results
print("Lasso: Average MSE = {:.4f}".format(avg_mse_lasso))
print("Ridge: Average MSE = {:.4f}".format(avg_mse_ridge))

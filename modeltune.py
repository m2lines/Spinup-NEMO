import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ExpSineSquared,RationalQuadratic
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.spatial.distance import cosine
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from scipy.linalg import orthogonal_procrustes

# from forecast import Predictions,load_ts


## Files
data_path = '/home/sg2147/nc_files/nc_files/'


### Read Actual Principle Components
actual = np.load(data_path+'so.npz',allow_pickle=True)
actual_ts = actual['ts']
print(actual['ts'].shape)


# #  ### Read predicted principle components   
# # pred = np.load(data_path+'hat_so.npy')
# # pred = pred[:20]  #20 years of prediction
# # print(pred.shape)


n_steps = 5  # Use past 5 steps to predict next step

X_train = []
y_train = []

for i in range(len(actual_ts) - n_steps):
    X_train.append(actual_ts[i:i + n_steps])   # Use past `n_steps`
    y_train.append(actual_ts[i + n_steps])     # Predict the next step

# Convert to numpy arrays
X_train = np.array(X_train)   # Shape: (samples, n_steps, n_features)
y_train = np.array(y_train)   # Shape: (samples, n_features)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# Split into train-test subsets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, shuffle=False)
print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)

X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Shape: (n_samples, n_steps * n_features)
X_test_flat = X_test.reshape(X_test.shape[0], -1)    # Shape: (n_samples, n_steps * n_features)
print("X_train_flat shape:", X_train_flat.shape)
print("X_test_flat shape:", X_test_flat.shape)





class CustomGP(BaseEstimator):
    def __init__(self, irregularities_kernel=None, noise_kernel=None, long_term_trend_kernel=None, alpha=1e-10, n_restarts_optimizer=5):
        self.irregularities_kernel = irregularities_kernel
        self.noise_kernel = noise_kernel
        self.long_term_trend_kernel = long_term_trend_kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

        # Combine kernels (if defined)
        if self.irregularities_kernel and self.noise_kernel and self.long_term_trend_kernel:
            self.kernel = self.irregularities_kernel + self.noise_kernel + self.long_term_trend_kernel
        else:
            self.kernel = None
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=42
        )

    def fit(self, X, y):
        self.gp.fit(X, y)
        return self

    def predict(self, X):
        return self.gp.predict(X, return_std=True)[0]






param_grid = {
    'irregularities_kernel': [
        1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0),
        10.0 * ExpSineSquared(length_scale=5.0, periodicity=5.0),
        0.5 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    ],
    'noise_kernel': [
        1.0 * WhiteKernel(noise_level=1.0),
        0.1 * WhiteKernel(noise_level=0.1)
    ],
    'long_term_trend_kernel': [
        0.1 * DotProduct(sigma_0=0.0),
        0.5 * DotProduct(sigma_0=1.0),
        1.0 * RBF(length_scale=1.0)
    ],
    'alpha': [1e-10, 1e-5, 1e-2],
    'n_restarts_optimizer': [5, 10]
}




# Define Grid Search
grid_search = GridSearchCV(
    estimator=CustomGP(
        irregularities_kernel=param_grid['irregularities_kernel'][0],
        noise_kernel=param_grid['noise_kernel'][0],
        long_term_trend_kernel=param_grid['long_term_trend_kernel'][0],
        alpha=1e-10,
        n_restarts_optimizer=5,
    ),
    param_grid=param_grid,
    cv=3,  # Cross-validation folds
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)




# Fit Grid Search
grid_search.fit(X_train_flat, y_train)

# Get Best Model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Predict with the best model
y_pred = best_model.predict(X_test_flat)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Best Model RMSE:", rmse)

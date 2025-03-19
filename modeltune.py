import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ExpSineSquared
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.spatial.distance import cosine
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



train_data, test_data = train_test_split(actual_ts, test_size=0.3, random_state=42)
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")



# Define base kernel
kernel = (
    1.0 * RBF(length_scale=1.0) +
    WhiteKernel(noise_level=1.0) +
    DotProduct(sigma_0=1.0)
)

# Create base model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

param_grid = {
    'kernel__k1__length_scale': [0.1, 1.0, 10.0],
    'kernel__k2__noise_level': [0.1, 1.0, 10.0],
    'kernel__k3__sigma_0': [0.1, 1.0, 10.0],
    'alpha': [1e-10, 1e-5, 1e-2],
}

grid_search = GridSearchCV(
    estimator=gp,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Use RMSE as evaluation
    cv=None,  # No cross-validation
    n_jobs=-1  # Use all processors
)
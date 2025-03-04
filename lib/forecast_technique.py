from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
    DotProduct,
)  # , Matérn


def init_technique():
    """
    Define Gaussian Process regressor with specified kernel. We use :
        - a long term trend kernel that contains a Dot Product with sigma_0 = 0, for the linear behaviour.
        - an irregularities_kernel for periodic patterns CHANGER 5/45 1/len(data)?
        - a noise_kernel
    We also set a n_restarts_optimizer to optimize hyperparameters

    Returns:
        GaussianProcessRegressor: The Gaussian Process regressor.
    """
    long_term_trend_kernel = 0.1 * DotProduct(
        sigma_0=0.0
    )  # + 0.5*RBF(length_scale=1/2)# +
    irregularities_kernel = (
        10 * ExpSineSquared(length_scale=5 / 45, periodicity=5 / 45)
    )  # 0.5**2*RationalQuadratic(length_scale=5.0, alpha=1.0) + 10 * ExpSineSquared(length_scale=5.0)
    noise_kernel = 2 * WhiteKernel(
        noise_level=1
    )  # 0.1**2*RBF(length_scale=0.01) + 2*WhiteKernel(noise_level=1)
    kernel = irregularities_kernel + noise_kernel + long_term_trend_kernel
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=8, random_state=42
    )


# All the tings a user may want to do with the data [Normalization/Data transformation, Feature Engineering, Dimonsionality reduction, Forecasting]
# Define the Dimentionality Technique class and the Forecast Technique class somewhere in seperate files in the project
# These should take in input in a standard form and return the output in a standard form
# The content/techniques used and implementation is not important, just that the input and output is in a standard form
# The classes should be able to be used in the following way:
# - They can be imported and called in the forecast.py file
# - The user only needs to change the two files to change the technique used and given that the input and output is in a standard form
#   the user should not need to change the forecast.py file
# If multiple techniques are used, the user should be able to choose which technique to use in the forecast.py file by way of changing a variable
# This should later be done by way of a config file

# Specifically the defining of the GP should not be done in the Prediction contructor

# First refactor the forecast.py file to use the classes while still using PCA and the GP


def apply_forecast(y_train, x_train, x_pred):
    # mean, std, y_train, y_test, x_train, x_pred = *dataain, x_pred = data

    gp = init_technique()

    gp.fit(x_train, y_train)
    y_hat, y_hat_std = gp.predict(x_pred, return_std=True)

    return y_hat, y_hat_std


def rmseOfPCA(self, n):
    reconstruction = self.reconstruct(n)
    rmse_values = self.rmseValues(reconstruction) * 2 * self.desc["std"]
    rmse_map = self.rmseMap(reconstruction) * 2 * self.desc["std"]
    return reconstruction, rmse_values, rmse_map

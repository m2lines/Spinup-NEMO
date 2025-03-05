from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
    DotProduct,
)  # , Matérn
import numpy as np
import pandas as pd


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


def apply_forecast(y_train, x_train, x_pred):
    # mean, std, y_train, y_test, x_train, x_pred = *dataain, x_pred = data

    gp = init_technique()
    forecaster = ForecasterRecursive(
        regressor=gp,
        lags=29,
        window_features=RollingFeatures(stats=["mean"], window_sizes=10),
    )
    forecaster.fit(pd.Series(y_train))
    y_hat, y_hat_std = forecaster.predict(steps=30), np.asarray([1, 1, 1, 1, 1, 1])

    # gp.fit(x_train, y_train)
    # y_hat, y_hat_std = gp.predict(x_pred, return_std=True)

    return y_hat, y_hat_std


def rmseOfPCA(self, n):
    reconstruction = self.reconstruct(n)
    rmse_values = self.rmseValues(reconstruction) * 2 * self.desc["std"]
    rmse_map = self.rmseMap(reconstruction) * 2 * self.desc["std"]
    return reconstruction, rmse_values, rmse_map

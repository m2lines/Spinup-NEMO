import os
import copy
import random
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn.decomposition import PCA
import joblib
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
    DotProduct,
)  # , Matérn
import pickle
import warnings

warnings.filterwarnings("ignore")


# file    #Select the file where the prepared simu was saved
# var     #Select the var you want to forecast
def load_ts(file, var):
    """
    Load time series data from the file where are saved the prepared simulations.
    This function is used the get the prepared data info in order to instantiate a prediction class

    Parameters:
        file (str): The path to the file containing the time series data.
        var (str) : The variable to be loaded.

    Returns:
        tuple: A tuple containing:
            - df (DataFrame): DataFrame containing the time series data.
            - dico (dict): A dictionary containing all informations on the simu (pca, mean, std, time_dim)...
    """
    dico = np.load(file + f"/{var}.npz", allow_pickle=True)
    dico = {key: dico[key] for key in dico.keys()}
    df = pd.DataFrame(
        dico["ts"], columns=[f"{var}-{i + 1}" for i in range(np.shape(dico["ts"])[1])]
    )
    with open(file + f"/pca_{var}", "rb") as file:
        dico["pca"] = pickle.load(file)
    return df, dico


##################################
##                              ##
##   LOAD AND PREPARE A SIMU    ##
##                              ##
##################################


class Simulation:
    """!!!Modified version where we apply a script to get yearly average for the simu before!!!"""

    """
    A class for loading and preparing simulation data.

    Attributes:
        path (str)                    : The path to the simulation data.
        term (str)                    : The term for the simulation.
        files (list)                  : List of files related to the simulation.
        start (int)                   : The start index for data slicing.
        end (int)                     : The end index for data slicing.
        ye (bool)                     : Flag indicating whether to use ye or not.
        comp (float)                  : The comp value for the simulation.
        len (int)                     : The length of the simulation.
        desc (dict)                   : A dictionary containing descriptive statistics of the simulation data.
        time_dim (str)                : The name of the time dimension.
        y_size (int)                  : The size of the y dimension.
        x_size (int)                  : The size of the x dimension.
        z_size (int or None)          : The size of the z dimension, if available.
        shape (tuple)                 : The shape of the simulation data.
        simulation (xarray.DataArray) : The loaded simulation data
    """

    def __init__(
        self, path, term, start=0, end=None, comp=0.9, ye=True, ssca=False
    ):  # choose jobs 3 if 2D else 1
        """
        Initialize Simulation with specified parameters.

        Parameters:
            path (str)             : The path to the simulation data.
            term (str)             : The term for the simulation.
            start (int, optional)  : The start index for data slicing. Defaults to 0.
            end (int, optional)    : The end index for data slicing. Defaults to None.
            comp (float, optional) : The comp value for the simulation. Defaults to 0.9. Percentage of explained variance.
            ye (bool, optional)    : Flag indicating whether to use ye or not. Defaults to True.
            ssca (bool, optional)  : Flag indicating whether ssca is used. Defaults to False.  #Not used in this class
        """
        self.path = path
        self.term = term
        self.files = Simulation.getData(path, term)
        self.start = start
        self.end = end
        self.ye = ye
        self.comp = comp
        self.len = 0
        self.desc = {}
        self.getAttributes()  # self time_dim, y_size, x_size,
        self.getSimu()  # self simu , desc {"mean","std","min","max"}

    #### Load files and dimensions info ###

    def getData(path, term):
        """
        Get the files related to the simulation in the right directory

        Parameters:
            path (str): The path to the simulation data.
            term (str): The term for the simulation.
                        zos    -> sea surface height (also ssh) - (t,y,z)
                        so     -> salinity - (t,z,y,x)
                        thetao -> temperature - (t,z,y,x)

        Returns:
            list: List of files related to the simulation.
        """
        grid = []
        for file in sorted(os.listdir(path)):
            if term[1] in file:  # add param!=""
                grid.append(path + "/" + file)
        return grid

    def getAttributes(self):
        """
        Get attributes of the simulation data.
        """
        array = xr.open_dataset(
            self.files[-1], decode_times=False, chunks={"time": 200, "x": 120}
        )
        self.time_dim = list(array.dims)[
            0
        ]  # Seems that index at 0 is 'nav_lat' TODO: Investigate this
        self.y_size = array.sizes["y"]
        self.x_size = array.sizes["x"]
        if "deptht" in array[self.term[0]].dims:
            self.z_size = array.sizes["deptht"]
            self.shape = (self.z_size, self.y_size, self.x_size)
        elif "olevel" in array[self.term[0]].dims:
            self.z_size = array.sizes["olevel"]
            self.shape = (self.z_size, self.y_size, self.x_size)
        else:
            self.z_size = None
            self.shape = (self.y_size, self.x_size)
        # self.getSSCA(array)

    #### Load simulation ###

    def getSimu(self):
        """
        Load simulation data.
        """
        # array = list(Parallel(jobs)(delayed(self.loadFile)(file) for file in self.files))
        array = [self.loadFile(file) for file in self.files if self.len < self.end]
        array = xr.concat(array, self.time_dim)
        self.desc = {
            "mean": np.nanmean(array),
            "std": np.nanstd(array),
            "min": np.nanmin(array),
            "max": np.nanmax(array),
        }
        self.simulation = array
        del array

    def loadFile(self, file):
        """
        Load simulation data from a file. Stop when the imported simulation date is superior to the attirbute end.
        This is why we cannot use parallelisation to import simulations.

        Parameters:
            file (str): The path to the file containing the simulation data.

        Returns:
            (xarray.DataArray) : The loaded simulation data.
        """
        array = xr.open_dataset(
            file, decode_times=False, chunks={"time": 200, "x": 120}
        )
        array = array[self.term[0]]
        self.len = self.len + array.sizes[self.time_dim]
        # print(self.len)
        # if self.ye:
        #    #array = array.coarsen({self.time_dim: 12}).mean()   #TO CHANGE WITH TIME DIM
        # if self.len + array.sizes[self.time_dim] > self.end:
        #    array = array[0:self.end-self.len]
        #    self.len = self.len + array.sizes[self.time_dim]
        # else:
        #    self.len = self.len + array.sizes[self.time_dim]
        return array.load()

    #########################
    #  prepare simulation   #
    #########################

    def prepare(self, stand=True):
        """
        Prepare the simulation data selecting indices from start to end, updating length and obtaining statistics,
        standardizing if specified.

        Parameters:
            stand (bool, optional): Flag indicating whether to standardize the simulation data. Defaults to True.
        """
        if self.end is not None:
            self.simulation = self.simulation[self.start : self.end]
        else:
            self.simulation = self.simulation[self.start :]
        self.len = np.shape(self.simulation)[0]
        # self.removeClosedSeas()
        self.desc.update(
            {
                "mean": np.nanmean(self.simulation),
                "std": np.nanstd(self.simulation),
                "min": np.nanmin(self.simulation),
                "max": np.nanmax(self.simulation),
            }
        )
        if stand:
            self.standardize()
        self.simulation = self.simulation.values

    def getSSCA(self, array):
        """
        Extract the seasonality data from the simulation. Not used : we import yearly data

        Parameters:
            array (xarray.Dataset): The last dataset containing simulation data in the simulation file.
        """
        array = array[self.term[0]].values
        n = np.shape(array)[0] // 12 * 12
        array = array[-n:]
        ssca = np.array(array).reshape(
            (n // 12, 12) + self.shape
        )  # np.array(array[self.term])
        ssca = np.mean(ssca, axis=0)
        ssca_extended = np.tile(ssca, (n // 12, 1, 1))
        self.desc["ssca"] = sscad
        if self.ye == False:
            self.simulation = array - ssca_extended

    def removeClosedSeas(self):
        """
        Remove closed seas from the simulation data. Not used : we don't have the specific mask to fill with predictions
        """
        array = self.simulation
        y_range = [
            slice(240, 266),
            slice(235, 276),
            slice(160, 201),
        ]  # mer noir, grands lacs, lac victoria
        x_range = [slice(195, 213), slice(330, 351), slice(310, 325)]
        for y, x in zip(y_range, x_range):
            array = array.where(
                (array["x"] < x.start)
                | (array["x"] >= x.stop)
                | (array["y"] < y.start)
                | (array["y"] >= y.stop),
                drop=True,
            )
        self.simulation = array

    def standardize(self):
        """
        Standardize the simulation data.
        """
        self.simulation = (self.simulation - self.desc["mean"]) / (2 * self.desc["std"])

    ##################
    #  Compute PCA   #
    ##################

    def applyPCA(self):
        """
        Apply Principal Component Analysis (PCA) to the simulation data.
        """
        array = self.simulation.reshape(self.len, -1)
        self.bool_mask = np.asarray(np.isfinite(array[0, :]), dtype=bool)
        array_masked = array[:, self.bool_mask]
        pca = PCA(self.comp, whiten=False)
        self.components = pca.fit_transform(array_masked)
        self.pca = pca

    def getPC(self, n):
        """
        Get principal component map for the specified component.

        Parameters:
            n (int) : component used for reconstruction.

        Returns:
            (numpy.ndarray): The principal component map.
        """
        # map_ = np.zeros((np.product(self.shape)), dtype=float)
        map_ = np.zeros((np.prod(self.shape)), dtype=float)
        map_[~self.bool_mask] = np.nan
        map_[self.bool_mask] = self.pca.components_[n]
        map_ = map_.reshape(self.shape)
        map_ = 2 * map_ * self.desc["std"] + self.desc["mean"]
        return map_

    ############################### NOT USED IN THE MAIN.PY ###############################
    def rmseOfPCA(self, n):
        reconstruction = self.reconstruct(n)
        rmse_values = self.rmseValues(reconstruction) * 2 * self.desc["std"]
        rmse_map = self.rmseMap(reconstruction) * 2 * self.desc["std"]
        return reconstruction, rmse_values, rmse_map

    def rmseValues(self, reconstruction):
        truth = self.simulation  # * 2 * self.desc["std"] + self.desc["mean"]
        rec = reconstruction  # * 2 * self.desc["std"] + self.desc["mean"]
        if len(np.shape(truth)) == 3:
            n = np.count_nonzero(~np.isnan(truth[0]))
            rmse_values = np.sqrt(np.nansum((truth - rec) ** 2, axis=(1, 2)) / n)
        else:
            n = np.count_nonzero(~np.isnan(self.simulation[0]), axis=(1, 2))
            rmse_values = np.nansum((truth - rec) ** 2, axis=(2, 3))
            for i in range(len(n)):
                rmse_values[:, i] = rmse_values[:, i] / n[i]
            rmse_values = np.sqrt(rmse_values)
        return rmse_values

    def rmseMap(self, reconstruction):
        t = self.len
        truth = self.simulation
        reconstruction = reconstruction
        return np.sqrt(np.sum((self.simulation[:] - reconstruction) ** 2, axis=0) / t)

    def reconstruct(self, n):
        """
        Reconstruct data using a specified number of principal components.

        Parameters:
            n (int) : The number of components used for reconstruction.

        Returns:
            (numpy.array) : The reconstructed data.
        """
        rec = []
        # int_mask =   # Convert the boolean mask to int mask once
        self.int_mask = self.bool_mask.astype(np.int32).reshape(
            self.shape
        )  # Reshape to match the shape of map_
        for t in range(len(self.components)):
            map_ = np.zeros(self.shape, dtype=np.float32)
            arr = np.array(
                list(self.components[t, :n]) + [0] * (len(self.pca.components_) - n)
            )
            map_[self.int_mask == 1] = self.pca.inverse_transform(arr)
            map_[self.int_mask == 0] = np.nan
            rec.append(map_)
        return np.array(rec)

    #######################################################################################################

    ##################
    #   Save in db   #
    ##################

    def makeDico(self):
        """
        Create a dictionary containing simulation data, descriptive statistics, and other relevant information.

        Returns:
            (dict) : A dictionary containing simulation data and information.
        """
        dico = dict()
        dico["ts"] = self.components.tolist()
        dico["mask"] = self.bool_mask
        dico["desc"] = self.desc
        dico["cut"] = self.start
        dico["x_size"] = self.x_size
        dico["y_size"] = self.y_size
        if self.z_size is not None:
            dico["z_size"] = self.z_size
        dico["shape"] = self.shape
        return dico

    def save(self, file, term):
        """
        Save the simulation data and information to files.

        Parameters:
            file (str): The path to the directory where the files will be saved.
            term (str): The term used in the filenames.
        """
        simu_dico = self.makeDico()
        if not os.path.exists(file):  # save infos
            os.makedirs(file)
        with open(f"{file}/{term}/pca_{term}", "wb") as f:
            pickle.dump(self.pca, f)
        np.savez(f"{file}/{term}/{term}", **simu_dico)


#################################
##                             ##
##   Forecast & reconstruct    ##
##                             ##
#################################


class Predictions:
    """
    Class for forecasting and reconstructing time series data using Gaussian Processes.

     Attributes:
            var (str)                     : The variable name.
            data (DataFrame)              : The time series data.
            info (dict)                   : Additional information.
            gp (GaussianProcessRegressor) : The Gaussian Process regressor.
            w (int)                       : Width for moving average and metrics calculation.
    """

    def __init__(self, var, data=None, info=None, gp=None, w=12):
        """
        Initialize the Predictions object.

        Parameters:
            var (str)                     : The variable name.
            data (DataFrame)              : The time series data.
            info (dict)                   : Additional information.
            gp (GaussianProcessRegressor) : The Gaussian Process regressor.
            w (int)                       : Width for moving average and metrics calculation.
        """
        self.var = var
        self.gp = Predictions.defineGP() if gp is None else gp
        self.w = w
        self.data = data
        self.info = info
        self.info["desc"] = self.info["desc"].item()
        self.len_ = len(self.data)

    def __len__(self):
        return len(self.data)

    ##################
    #    Forecast    #
    ##################

    @staticmethod
    def defineGP():
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

    def Forecast(self, train_len, steps, jobs=1):
        """
        Parallel forecast of time series data/eofs using an independent GP for each time series

        Parameters:
            train_len (int)      : Length of the training data.
            steps (int)          : Number of steps to forecast.
            jobs (int, optional) : Number of parallel jobs to run. Defaults to 1.

        Returns:
            A tuple containing:
                - y_hats (DataFrame)     : Forecasted values.
                - y_stds (DataFrame)     : Standard deviations of the forecasts.
                - metrics (list of dict) : One dict of metrics by forecast
        """
        r = Parallel(n_jobs=jobs)(
            delayed(self.forecast_ts)(c, train_len, steps)
            for c in range(1, self.data.shape[1] + 1)
        )
        y_hats = pd.DataFrame(
            np.array([r[i][0] for i in range(len(r))]).T, columns=self.data.columns
        )
        y_stds = pd.DataFrame(
            np.array([r[i][1] for i in range(len(r))]).T, columns=self.data.columns
        )
        metrics = [r[i][2] for i in range(len(r))]
        return y_hats, y_stds, metrics

    def forecast_ts(self, n, train_len, steps=0):
        """
        Forecast of one time series, function called in parallel in Forecast

        Parameters:
            n (int)               : Variable index.
            train_len (int)       : Length of the training data.
            steps (int, optional) : Number of steps to forecast. Defaults to 0.

        Returns:
            A tuple containing:
                y_hat (array)     : Forecasted values.
                y_hat_std (array) : Standard deviations of the forecasts.
                metrics (dict)    : Dictionary of metrics defined in the corresponding function
        """
        random.seed(20)
        mean, std, y_train, y_test, x_train, x_pred = self.prepare(n, train_len, steps)
        self.gp.fit(x_train, y_train)
        y_hat, y_hat_std = self.gp.predict(x_pred, return_std=True)
        y_train, y_hat, y_hat_std = (
            y_train * 2 * std + mean,
            y_hat * 2 * std + mean,
            y_hat_std * 2 * std,
        )
        metrics = None
        if y_test is not None:
            metrics = Predictions.getMetrics(2, y_hat[train_len : len(self)], y_test)
        return y_hat, y_hat_std, metrics

    def prepare(self, n, train_len, steps):
        """
        Prepare data for forecasting.

        Parameters:
            n (int)         : Variable index.
            train_len (int) : Length of the training data.
            steps (int)     : Number of steps to forecast.

        Returns:
            A tuple containing:
                mean (float) : Mean of the training data.
                std (float)  : Standard deviation of the training data.
                y_train (numpy.array): Training data.
                y_test  (numpy.array): Test data.
                x_train (numpy.array): Training features.
                x_pred  (numpy.array): Prediction features.
        """
        x_train = np.linspace(0, 1, train_len).reshape(-1, 1)
        pas = x_train[1, 0] - x_train[0, 0]
        x_pred = np.arange(0, (len(self) + steps) * pas, pas).reshape(-1, 1)
        y_train = self.data[self.var + "-" + str(n)].iloc[:train_len].to_numpy()
        mean, std = np.nanmean(y_train), np.nanstd(y_train)
        y_train = (y_train - mean) / (2.0 * std)
        y_test = None
        if train_len < len(self):
            y_test = (
                self.data[self.var + "-" + str(1)]
                .iloc[train_len : len(self)]
                .to_numpy()
            )
        return mean, std, y_train, y_test, x_train, x_pred

    def show(self, n, y_hat, y_hat_std, train_len, color="tab:blue"):
        """
        Plot the forecasted time series data.

        Parameters:
            n (int)                 : Corresponding time serie/eof
            y_hat (numpy.array)     : Forecasted values.
            y_hat_std (numpy.array) : Standard deviations of the forecasts.
            train_len (int)         : Length of the training data.
            color (str, optional)   : Color for the plot. Defaults to "tab:blue".
        """
        figure = plt.figure(figsize=(7, 3))
        plt.plot(
            self.data[self.var + "-" + str(n)][:train_len],
            linestyle="dashed",
            color="black",
            alpha=0.7,
            label="Train serie",
        )
        if train_len < len(self):
            plt.plot(
                self.data[self.var + "-" + str(n)][train_len - 1 :],
                linestyle="dashed",
                color="black",
                alpha=0.5,
                label="Test serie",
            )
        plt.plot(y_hat, color=color, label=f"GP forecast")
        plt.fill_between(
            np.arange(0, len(y_hat)),
            y_hat + y_hat_std,
            y_hat - y_hat_std,
            color=color,
            alpha=0.2,
        )
        plt.title(f"Forecast of {self.var} {str(n)}")
        plt.legend()
        plt.show()
        print()

    @staticmethod
    def getMetrics(w, y_hat, y_test):
        """
        Calculate metrics for evaluating the forecast.

        Parameters:
            w (int)             : Width for moving average.
            y_hat (numpy.array) : Forecasted values.
            y_test(numpy.array) : True values.

        Returns:
            dict: Dictionary containing calculated metrics.
        """
        ma_test = np.convolve(y_test / w, np.ones(w), mode="valid")
        ma_pred = np.convolve(y_hat / w, np.ones(w), mode="valid")
        dist = np.convolve((y_hat - y_test) / w, np.ones(w), mode="valid")
        mse = np.convolve(((y_hat - y_test) ** 2) / w, np.ones(w), mode="valid")
        dist_max, std = [], []
        for i in range(w, len(y_hat) + 1):
            windowT = y_test[i - w : i]
            windowH = y_hat[i - w : i]
            # maxi/mini
            maxi = np.max(windowT) - np.mean(windowT)
            mini = np.mean(windowT) - np.min(windowT)
            dist_max.append(max(maxi, mini))
            # std
            std.append(np.std(windowT, ddof=1))
        return {
            "ma_true": ma_test,
            "ma_pred": ma_pred,
            "dist": dist,
            "dist_max": dist_max,
            "mse": mse,
            "std_true": np.array(std),
        }

    ###################
    #   Reconstruct   #
    ###################

    def reconstruct(self, predictions, n, begin=0):
        """
        Reconstruct the time series data from predictions.

        Parameters:
            predictions (DataFrame) : Forecasted values for each component.
            n (int)                 : Number of components to consider for reconstruction.
            begin (int, optional)   : Starting index for reconstruction. Defaults to 0.

        Returns:
            array: Reconstructed time series data.
        """
        rec = []
        self.int_mask = self.info["mask"].astype(np.int32).reshape(self.info["shape"])
        for t in range(begin, len(predictions)):
            map_ = np.zeros((self.info["shape"]), dtype=np.float32)
            arr = np.array(
                list(predictions.iloc[t, :n])
                + [0] * (len(self.info["pca"].components_) - n)
            )
            map_[self.int_mask == 1] = self.info["pca"].inverse_transform(arr)
            map_[self.int_mask == 0] = np.nan
            rec.append(map_)
        return np.array(rec) * 2 * self.info["desc"]["std"] + self.info["desc"]["mean"]


#################################
##                             ##
##   Forecast & reconstruct    ##
##                             ##
#################################

# NOT UP-TO-DATE WITH PREDICTION CLASS


class optimization:
    def __init__(
        self,
        ids,
        ratio,
        ncomp,
        var,
        steps=1,
        min_test=50,
        min_train=50,
        kernels=None,
        trunc=None,
    ):
        random.shuffle(ids)
        i = int(len(ids) * ratio)
        self.ids_eval = ids[:i]
        self.ids_test = ids[i:]
        self.simu_eval = [TS(id_, var) for id_ in ids[:i]]
        self.simu_test = [TS(id_, var) for id_ in ids[i:]]
        self.ncomp = ncomp
        self.var = var
        self.min_train = min_train
        self.min_test = min_test
        self.steps = steps
        self.trunc = trunc
        self.kernels = [RBF(), RationalQuadratic()] if kernels is None else kernels
        self.seed = random.randint(1, 200)

    ###___get all gp___###

    def getAllGP(self, n=4, r=RBF()):
        kernels = self.kernelCombination(r, n)
        listGP = []
        for kernel in np.array(kernels).reshape(-1):
            kernel = kernel + WhiteKernel()
            listGP.append(
                GaussianProcessRegressor(
                    kernel=kernel, normalize_y=False, n_restarts_optimizer=0
                )
            )
        self.gps = listGP

    def kernelCombination(self, r=RBF(), n=4):
        k = self.kernels
        if n == 1:
            return r
        else:
            return (
                self.kernelCombination(r + k[0], n=n - 1),
                self.kernelCombination(r * k[0], n=n - 1),
                self.kernelCombination(r + k[1], n=n - 1),
                self.kernelCombination(r * k[1], n=n - 1),
            )

    ###___evaluate current gp___###

    def evaluateCurrentProcess(self):
        random.seed(self.seed)
        results_eval = []
        print("evaluation : ")
        for simu in self.simu_eval:
            print(f"Processing simulation {simu.id}")
            if self.min_train < len(simu) - self.min_test:
                train_lens = np.arange(
                    self.min_train, len(simu) - self.min_test, self.steps
                )
                results_eval.append(
                    simu.evaluateModel(self.ncomp, train_lens, f"{self.var}-1", jobs=15)
                )
                if self.trunc is None:
                    results_eval[-1] = np.sum(result[-1])
                else:
                    results_eval[-1] = np.sum(
                        [min(val, self.trunc) for val in results_eval[-1]]
                    )
        results_test = []
        print("\ntest : ")
        for simu in self.simu_test:
            print(f"Processing simulation {simu.id}")
            if self.min_train < len(simu) - self.min_test:
                train_lens = np.arange(
                    self.min_train, len(simu) - self.min_test, self.steps
                )
                results_test.append(
                    simu.evaluateModel(self.ncomp, train_lens, f"{self.var}-1", jobs=15)
                )
                if self.trunc is None:
                    results_test[-1] = np.sum(results_test[-1])
                else:
                    results_test[-1] = np.sum(
                        [min(val, self.trunc) for val in results_test[-1]]
                    )
        self.current_score_eval = np.sum(results_eval)
        self.current_score_test = np.sum(results_test)

    ###___evaluate gp___###

    # this methode should be changed to rmse with raw simulation
    def evaluateProcess(self, simu, train_lens, process):
        currentgp = simu.gp
        simu.gp = process
        print("-", end="")
        test = simu.evaluateModel(self.ncomp, train_lens, f"{self.var}-1", jobs=15)
        simu.gp = currentgp
        if self.trunc is None:
            return np.sum(test)
        else:
            return np.sum([min(val, self.trunc) for val in test])

    def evaluateKernels(self):
        random.seed(self.seed)
        results = []
        for simu in self.simu_eval:
            print(f"Processing simulation {simu.id} ", end="")
            if self.min_train < len(simu) - self.min_test:
                train_lens = np.arange(
                    self.min_train, len(simu) - self.min_test, self.steps
                )
                results.append(
                    [
                        self.evaluateProcess(simu, train_lens, process)
                        for process in self.gps
                    ]
                )
                print("", end="\n")
        results = [
            (process, score)
            for process, score in zip(self.gps, np.sum(results, axis=0))
        ]
        self.scores_eval = sorted(results, key=lambda item: item[1], reverse=True)

    ###___Select on test___###

    def testKernels(self):
        random.seed(self.seed)
        gps_test = [
            process
            for process, score in self.scores_eval
            if score > self.current_score_eval
        ]
        results = []
        for simu in self.simu_test:
            print(f"Processing simulation {simu.id}", end="")
            if self.min_train < len(simu) - self.min_test:
                train_lens = np.arange(
                    self.min_train, len(simu) - self.min_test, self.steps
                )
                results.append(
                    [
                        self.evaluateProcess(simu, train_lens, process)
                        for process in gps_test
                    ]
                )
                print("", end="\n")
        results = [
            (process, score)
            for process, score in zip(gps_test, np.sum(results, axis=0))
        ]
        self.scores_test = sorted(results, key=lambda item: item[1], reverse=True)

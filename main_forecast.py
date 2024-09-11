import numpy as np
import pandas as pd
import os
import pickle
import sys
import random
import argparse

sys.path.insert(0, "../lib/")
from forecast import Predictions, Simulation, load_ts


file_simu_prepared = "/data/mtissot/spinup_data/simus_prepared"
file_simu_predicted = "/data/mtissot/spinup_data/simus_predicted"


def prepare(term, simu_path, start, end, ye, comp):
    simu = Simulation(
        path=simu_path, start=start, end=end, ye=ye, comp=comp, term=term
    )  # Load yearly or monthly simulations
    print(f"{term} loaded")
    simu.prepare()  # Prepare simulations : start to end - removeClosedSeas - (removeSSCA) - standardize - to numpy
    print(f"{term} prepared")
    simu.applyPCA()  # Exctract time series through PCA
    print(f"PCA applied on {term}")
    simu.save(
        file_simu_prepared, term
    )  # Create dico and save: time series - mask - desc -(ssca) - cut(=start) - x_size - y_size - (z_size) - shape
    print(f"{term} saved at {file_simu_prepared}/{term}")
    del simu  # Clean RAM


def jump(term, steps):
    df, infos = load_ts(
        f"{file_simu_prepared}/{term}", term
    )  # load dataframe and infos
    simu_ts = Predictions(term, df, infos)  # create the class to predict
    print(f"{term} time series loaded")
    y_hat, y_hat_std, metrics = simu_ts.Forecast(len(simu_ts), steps)  # Forecast
    print(f"{term} time series forcasted")
    n = len(simu_ts.info["pca"].components_)  # Reconstruct n predicted components
    predictions_zos = simu_ts.reconstruct(y_hat, n, begin=len(simu_ts))
    print(f"{term} predictions reconstructed")
    np.save(f"{file_simu_predicted}/{term}.npy", predictions_zos)  # Save
    print(f"{term} predictions saved at {file_simu_predicted}")
    del simu_ts


def emulate(simu_path, steps, ye, start, end, comp):
    for term in ["zos", "so", "thetao"]:
        print(f"Preparing {term}...")
        prepare(term, simu_path, ye, start, end, comp)
        print()
        print(f"Forecasting {term}...")
        jump(term, steps)
        print()


if __name__ == "__main__":
    # simu_path = "/scratchu/mtissot/SIMUp6Y"
    parser = argparse.ArgumentParser(description="Emulator")
    parser.add_argument("--path", type=str, help="Enter the simulation pathn")  # Path
    parser.add_argument(
        "--ye", type=bool, help="Transform monthly simulation to yearly simulation"
    )  # Transform monthly simulation to yearly simulation
    parser.add_argument(
        "--start", type=int, help="Start of the training"
    )  # Start of the simu : 0 to keep spin up / t to cut the spin up
    parser.add_argument(
        "--end", type=int, help="End of the training"
    )  # End of the simu  (end-strat = train len)
    parser.add_argument(
        "--steps", type=int, help="Number of steps to emulate"
    )  # Number of years you want to forecast
    parser.add_argument(
        "--comp", type=float, help="Explained variance ratio for the pca"
    )  # Explained variance ratio for the pca
    args = parser.parse_args()

    emulate(
        simu_path=args.path,
        steps=args.steps,
        ye=args.ye,
        start=args.start,
        end=args.end,
        comp=args.comp,
    )

    # update_restart_files

    # python SpinUp/jumper/main/main_forecast.py --ye True --start 25 --end 65 --comp 0.9 --steps 30 --path /scratchu/mtissot/SIMUp6Y

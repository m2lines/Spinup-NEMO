import numpy as np
import pandas as pd
import os
import pickle
import sys
import random
import argparse

sys.path.insert(0, "./lib/")
from forecast import Predictions, Simulation, load_ts


def prepare(term, simu_path, start, end, ye, comp):
    """
    Prepare the simulation for the forecast

    Args:
        term (str): term to forecast
        simu_path (str): path to the simulation
        start (int): start of the simulation
        end (int): end of the simulation
        ye (bool): transform monthly simulation to yearly simulation
        comp (int or float): explained variance ratio for the pcaA

    Returns:
        None

    """

    # Load yearly or monthly simulations
    simu = Simulation(path=simu_path, start=start, end=end, ye=ye, comp=comp, term=term)
    print(f"{term[0]} loaded")

    # Prepare simulations : start to end - removeClosedSeas - (removeSSCA) - standardize - to numpy
    simu.prepare()
    print(f"{term[0]} prepared")

    # Exctract time series through PCA
    simu.applyPCA()
    print(f"PCA applied on {term[0]}")

    os.makedirs(f"{simu_path}/simu_prepared/{term[0]}", exist_ok=True)
    print(f"{simu_path}/simu_prepared/{term[0]} created")

    # Create dictionary and save:
    # time series - mask - desc -(ssca) - cut(=start) - x_size - y_size - (z_size) - shape
    simu.save(f"{simu_path}/simu_prepared", term[0])
    print(f"{term[0]} saved at {simu_path}/simu_repared/{term[0]}")


def jump(simu_path, term, steps):
    """
    Forecast the simulation

    Args:
        simu_path (str): path to the simulation
        term (str): term to forecast
        steps (int): number of years to forecast

    Returns:
        None

    """

    df, infos = load_ts(
        f"{simu_path}/simu_prepared/{term}", term
    )  # load dataframe and infos

    # Create instance of prediction class
    simu_ts = Predictions(term, df, infos)
    print(f"{term} time series loaded")

    # Forecast
    y_hat, y_hat_std, metrics = simu_ts.Forecast(len(simu_ts), steps)
    print(f"{term} time series forcasted")

    # Reconstruct n predicted components
    n = len(simu_ts.info["pca"].components_)
    predictions_zos = simu_ts.reconstruct(y_hat, n, begin=len(simu_ts))
    print(f"{term} predictions reconstructed")

    os.makedirs(f"{simu_path}/simu_predicted/", exist_ok=True)
    np.save(f"{simu_path}/simu_predicted/{term}.npy", predictions_zos)  # Save
    print(f"{term} predictions saved at {simu_path}/simu_predicted/{term}.npy")


def emulate(simu_path, steps, ye, start, end, comp):
    """
    Emulate the forecast

    Args:
        simu_path (str): path to the simulation
        steps (int): number of years to forecast
        ye (bool): transform monthly simulation to yearly simulation
        start (int): start of the simulation
        end (int): end of the simulation
        comp (int or float): explained variance ratio for the pca

    Returns:
        None

    """

    # TODO: Load data from config / json file.
    dino_data = [
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("toce", "DINO_1y_grid_T.nc"),
    ]

    for term in dino_data:
        print(f"Preparing {term[0]}...")
        prepare(term, simu_path, start, end, ye, comp)
        print()
        print(f"Forecasting {term[0]}...")
        jump(simu_path, term[0], steps)
        print()


if __name__ == "__main__":
    # Perform forecast

    # Example use
    # python main_forecast.py --ye True --start 25 --end 65 --comp 0.9 --steps 30 --path /path/to/simu/data

    parser = argparse.ArgumentParser(description="Emulator")
    parser.add_argument(
        "--path", type=str, help="Path to simulation data to forecast from"
    )
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
        "--comp", type=str, help="Explained variance ratio for the pca"
    )  # Explained variance ratio for the pca
    args = parser.parse_args()

    # Convert comp to int or float if possible
    if args.comp.isdigit():
        args.comp = int(args.comp)
    elif args.comp.replace(".", "", 1).isdigit():
        args.comp = float(args.comp)
    elif args.comp == "None":
        args.comp = None

    emulate(
        simu_path=args.path,
        steps=args.steps,
        ye=args.ye,
        start=args.start,
        end=args.end,
        comp=args.comp,
    )

import numpy as np
import pandas as pd
import os
import pickle
import sys
import random
import argparse

sys.path.insert(0, "./lib/")
from forecast import Predictions, Simulation, load_ts


file_simu_prepared = "/rds/project/rds-5mCMIDBOkPU/ma595/nemo/NEMO/nemo_4.2.1/tests/DINO/concatenate_grid_150/simus_prepared_2"
file_simu_predicted = "/rds/project/rds-5mCMIDBOkPU/ma595/nemo/NEMO/nemo_4.2.1/tests/DINO/concatenate_grid_150/simus_predicted_2"


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
    simu = Simulation(
        path=simu_path, start=start, end=end, ye=ye, comp=comp, term=term
    )  # Load yearly or monthly simulations
    print(f"{term[0]} loaded")
    simu.prepare()  # Prepare simulations : start to end - removeClosedSeas - (removeSSCA) - standardize - to numpy
    print(f"{term[0]} prepared")
    simu.applyPCA()  # Exctract time series through PCA
    print(f"PCA applied on {term[0]}")

    os.makedirs(f'{file_simu_prepared}/{term[0]}', exist_ok=True)

    print(f'{file_simu_prepared}/{term[0]}')

    simu.save(
        file_simu_prepared, term[0]
    )  # Create dico and save: time series - mask - desc -(ssca) - cut(=start) - x_size - y_size - (z_size) - shape
    print(f"{term[0]} saved at {file_simu_prepared}/{term[0]}")
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

    os.makedirs(f'{file_simu_predicted}/{term[0]}', exist_ok=True)
    np.save(f"{file_simu_predicted}/{term}.npy", predictions_zos)  # Save
    print(f"{term} predictions saved at {file_simu_predicted}")
    del simu_ts


def emulate(simu_path, steps, ye, start, end, comp):
    # for term in ["zos","so","thetao"]:
    #     print(f"Preparing {term}...")
    #     prepare(term,simu_path, ye, start, end, comp)
    #     print()
    #     print(f"Forecasting {term}...")
    #     jump(term,steps)
    #     print()

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
        jump(term[0], steps)
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
        "--comp", type=str, help="Explained variance ratio for the pca"
    )  # Explained variance ratio for the pca
    args = parser.parse_args()

    # converts comp to int or float if possible
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

    # update_restart_files

    # python SpinUp/jumper/main/main_forecast.py --ye True --start 25 --end 65 --comp 0.9 --steps 30 --path /scratchu/mtissot/SIMUp6Y

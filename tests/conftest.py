import pytest


@pytest.fixture()
def setup_simulation_class():
    """
    Fixture to set up the simulation class
    """
    from lib.forecast import Simulation

    path = "tests/data/nemo_data_e3/"
    start = 20
    end = 50
    ye = True
    comp = 0.9
    term = ("toce", "DINO_1y_grid_T.nc")

    simu = Simulation(
        path=path,
        start=start,
        end=end,
        ye=ye,
        comp=comp,
        term=term,
    )

    return simu

import os
import pytest
from lib.forecast import Simulation
import numpy as np
import xarray as xr


@pytest.mark.parametrize(
    "test_case,term,expected_file_pattern,expected_count",
    [
        # Valid terms with their respective file patterns
        ("valid_temperature", ("toce", "DINO_1y_grid_T"), "1y_grid_T.nc", 1),
        ("valid_salinity", ("soce", "DINO_1y_grid_T"), "1y_grid_T.nc", 1),
        ("valid_ssh", ("ssh", "DINO_1m_To_1y_grid_T"), "1m_To_1y_grid_T.nc", 1),
    ],
)
def test_getData_valid_terms(test_case, term, expected_file_pattern, expected_count):
    """Test getData with valid terms and their expected file patterns"""

    data_path = "tests/data/nemo_data_e3"

    # Run the getData method
    files = Simulation.getData(data_path, term)
    print(files)
    # Verify results
    assert len(files) == expected_count

    # Files should be sorted
    assert files == sorted(files)

    # Check if all found files match the expected pattern for this variable
    for file in files:
        assert os.path.dirname(file) == data_path
        assert expected_file_pattern in os.path.basename(file)
        assert term[1] in os.path.basename(file)


@pytest.mark.parametrize(
    "test_case,term,expected_count",
    [
        # Valid term with nonexistent grid
        ("ssh_nonexistent_grid", ("ssh", "nonexistent"), 0),
    ],
)
def test_getData_invalid_combinations(test_case, term, expected_count):
    """Test getData with invalid term-file combinations"""
    data_path = "tests/data/nemo_data_e3"

    # Run the getData method
    files = Simulation.getData(data_path, term)

    # Verify no files are found
    assert len(files) == expected_count


@pytest.mark.parametrize(
    "setup_simulation_class, term, shape",
    [
        pytest.param(
            ("toce", "DINO_1y_grid_T.nc"),
            ("toce", "DINO_1y_grid_T.nc"),
            (36, 199, 62),
            marks=pytest.mark.xfail(
                reason="time_counter at different last index not first index"
            ),
        ),
        pytest.param(
            ("soce", "DINO_1y_grid_T.nc"),
            ("soce", "DINO_1y_grid_T.nc"),
            (36, 199, 62),
            marks=pytest.mark.xfail(
                reason="time_counter at different last index not first index"
            ),
        ),
        (
            ("ssh", "DINO_1m_To_1y_grid_T.nc"),
            ("ssh", "DINO_1m_To_1y_grid_T.nc"),
            (199, 62),
        ),
    ],
    indirect=["setup_simulation_class"],
)
# indirect parameterization of setup_simulation_class fixture
def test_getAttributes(setup_simulation_class, term, shape):
    """Tests getAttributes return the correct (x, y, z) values"""

    simulation = setup_simulation_class

    simulation.getAttributes()

    assert simulation.shape == shape
    assert simulation.term == term
    assert simulation.time_dim == "time_counter"


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_getSimu(setup_simulation_class):
    """
    Test that getSimu correctly sets the simulation DataArray and descriptive stats.
    """
    simu = setup_simulation_class

    # Check that 'simulation' attribute exists and is an xarray DataArray
    assert hasattr(simu, "simulation"), (
        "Simulation instance should have 'simulation' attribute"
    )
    assert isinstance(simu.simulation, xr.DataArray), (
        "'simulation' should be a xarray.DataArray"
    )
    # Check DataArray name matches variable name
    assert simu.simulation.name == simu.term[0], (
        f"DataArray name {simu.simulation.name} does not match term {simu.term[0]}"
    )

    # Extract data values for manual computation
    data = simu.simulation.values

    # Compute expected descriptive statistics
    expected_mean = np.nanmean(data)
    expected_std = np.nanstd(data)
    expected_min = np.nanmin(data)
    expected_max = np.nanmax(data)

    # Check that desc dictionary contains correct keys and values
    for key in ["mean", "std", "min", "max"]:
        assert key in simu.desc, f"'{key}' should be in simu.desc"

    # Compare actual vs expected values
    assert np.isclose(simu.desc["mean"], expected_mean), (
        f"Mean mismatch: {simu.desc['mean']} != {expected_mean}"
    )
    assert np.isclose(simu.desc["std"], expected_std), (
        f"Std mismatch: {simu.desc['std']} != {expected_std}"
    )
    assert np.isclose(simu.desc["min"], expected_min), (
        f"Min mismatch: {simu.desc['min']} != {expected_min}"
    )
    assert np.isclose(simu.desc["max"], expected_max), (
        f"Max mismatch: {simu.desc['max']} != {expected_max}"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        pytest.param(
            ("toce", "DINO_1y_grid_T.nc"),
            marks=pytest.mark.xfail(reason="time_counter atlast index not first index"),
        ),
        pytest.param(
            ("soce", "DINO_1y_grid_T.nc"),
            marks=pytest.mark.xfail(
                reason="time_counter at last index not first index"
            ),
        ),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_loadFile(setup_simulation_class):
    """
    Test that loadFile returns an xarray.DataArray with the correct variable and updates self.len appropriately.
    """
    simu = setup_simulation_class

    # Use the first file in the simulation's file list
    file_path = simu.files[0]

    # Reset length to zero to isolate this test
    simu.len = 0

    # Call loadFile
    data_array = simu.loadFile(file_path)

    # Ensure the return is a loaded xarray.DataArray
    assert isinstance(data_array, xr.DataArray), (
        "loadFile should return an xarray.DataArray"
    )

    # The DataArray name should match the simulation term
    assert data_array.name == simu.term[0], (
        f"DataArray name {data_array.name} does not match term {simu.term[0]}"
    )

    # After loading, self.len should equal the size along the time dimension
    assert simu.time_dim == "time_counter"
    expected_len = 50
    assert simu.len == expected_len, (
        f"Length after loadFile ({simu.len}) does not match expected ({expected_len})"
    )


@pytest.fixture()
def dummy_simu():
    """
    Create a bare Simulation instance without invoking __init__, to test prepare().
    """
    simu = Simulation.__new__(Simulation)
    return simu


def test_prepare_slices_based_on_start_end(dummy_simu):
    """
    Check data is sliced based on both start and end.
    """
    # Create a simple DataArray of length 10
    data = xr.DataArray(np.arange(10, dtype=float), dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 3
    dummy_simu.end = 8
    dummy_simu.desc = {}

    dummy_simu.prepare(stand=False)

    # After slicing, simulation should be numpy array [3,4,5,6,7]
    expected = np.arange(3, 8, dtype=float)
    assert isinstance(dummy_simu.simulation, np.ndarray)
    assert dummy_simu.len == expected.shape[0]
    np.testing.assert_array_equal(dummy_simu.simulation, expected)


def test_prepare_slices_start_specified_end_none(dummy_simu):
    """
    Check data is sliced using only start if end is not specified.
    """
    data = xr.DataArray(np.arange(10, dtype=float), dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 4
    dummy_simu.end = None
    dummy_simu.desc = {}

    dummy_simu.prepare(stand=False)

    # After slicing, simulation should be numpy array [4,5,6,7,8,9]
    expected = np.arange(4, 10, dtype=float)
    assert dummy_simu.len == expected.shape[0]
    np.testing.assert_array_equal(dummy_simu.simulation, expected)


def test_prepare_standardization_applied(dummy_simu):
    """
    Check results are standardized when stand=True.
    """
    data = xr.DataArray([0.0, 2.0, 4.0, 6.0], dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 0
    dummy_simu.end = None
    dummy_simu.desc = {}

    dummy_simu.prepare(stand=True)

    # Manually compute expected standardized values: (x - mean) / (2*std)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    expected = ((data - mean) / (2 * std)).values
    np.testing.assert_allclose(dummy_simu.simulation, expected)


def test_prepare_updates_desc_and_simulation(dummy_simu):
    """
    Check self.simulation is updated with its values and desc dict holds correct stats.
    """
    data = xr.DataArray([1.0, 2.0, 3.0, 5.0], dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 1
    dummy_simu.end = 4
    dummy_simu.desc = {}

    dummy_simu.prepare(stand=False)

    # After slicing, raw numpy array should match values[1:4]
    sliced = data.values[1:4]
    assert isinstance(dummy_simu.simulation, np.ndarray)
    np.testing.assert_array_equal(dummy_simu.simulation, sliced)

    # Check descriptive statistics in desc
    assert np.isclose(dummy_simu.desc["mean"], np.nanmean(sliced))
    assert np.isclose(dummy_simu.desc["std"], np.nanstd(sliced))
    assert np.isclose(dummy_simu.desc["min"], np.nanmin(sliced))
    assert np.isclose(dummy_simu.desc["max"], np.nanmax(sliced))


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_standardize(setup_simulation_class):
    """
    Test that calling standardize correctly transforms self.simulation using the stored mean and std,
    and that the descriptive statistics in self.desc remain unchanged.
    """
    simu = setup_simulation_class
    simu.len = 0
    # Load simulation data and compute descriptive stats
    simu.getSimu()

    # Copy original data and desc
    original_data = simu.simulation.copy().values
    original_mean = simu.desc["mean"]
    original_std = simu.desc["std"]

    # Apply standardization
    simu.standardize()

    # The simulation attribute should remain an xarray.DataArray
    assert isinstance(simu.simulation, xr.DataArray), (
        "simulation should be an xarray.DataArray after standardize"
    )

    # # Flatten arrays for comparison
    standardized_data = simu.simulation.values
    expected = (original_data - original_mean) / (2 * original_std)

    # # Check that the data was standardized correctly (accounting for NaNs)
    assert np.allclose(standardized_data, expected, equal_nan=True), (
        "standardize did not correctly transform simulation data"
    )

import os
import pytest
from lib.forecast import Simulation


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

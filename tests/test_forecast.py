"""
This file contains tests for the 'forecast' module in the 'lib' package.

The tests are written using pytest.
"""

import sys
import os
import pytest


def test_import_forecast():
    """Test that the forecast module can be imported successfully."""
    try:
        import lib.forecast

        assert True, "forecast imported successfully."
    except ImportError as e:
        pytest.fail(f"Failed to import forecast: {e}")

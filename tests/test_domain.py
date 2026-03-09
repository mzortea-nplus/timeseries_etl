"""Tests for domain pure functions."""

import numpy as np
import pytest

from timeseries_etl.domain.constants import OPERE_TO_KEY
from timeseries_etl.domain.labels import get_ylabel
from timeseries_etl.domain.stats import z_score


def test_z_score():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = z_score(x)
    assert np.allclose(z.mean(), 0.0)
    assert np.allclose(z.std(), 1.0)


def test_z_score_constant():
    x = np.array([5.0, 5.0, 5.0])
    z = z_score(x)
    assert np.allclose(z, 0.0)


def test_get_ylabel():
    assert get_ylabel("ICD_01_x") == "Rotazione longitudinale [mrad]"
    assert get_ylabel("ICD_01_y") == "Rotazione trasversale [mrad]"
    assert get_ylabel("ICD_01_t") == "Temperatura [°C]"
    assert get_ylabel("POT_01_s") == "Spostamento [mm]"
    assert get_ylabel("EST_01_e") == "Estensione [mm]"
    assert get_ylabel("unknown_zz") == "unknown_zz"


def test_opere_mapping():
    assert OPERE_TO_KEY["P001"] == "P001_Sommacampagna"
    assert OPERE_TO_KEY["P005"] == "P005_Adige_Ovest"

"""Tests for config loader."""

import pytest

from timeseries_etl.config import normalize_month_year


def test_normalize_month_year_yyyy_mm():
    year, month = normalize_month_year("2026-02")
    assert year == "2026"
    assert month == "02"


def test_normalize_month_year_yyyy_mm_dd():
    year, month = normalize_month_year("2025-12-10")
    assert year == "2025"
    assert month == "12"


def test_normalize_month_year_strip():
    year, month = normalize_month_year("  2026-02  ")
    assert year == "2026"
    assert month == "02"

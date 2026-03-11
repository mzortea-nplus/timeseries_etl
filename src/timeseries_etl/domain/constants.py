"""Shared constants for the whole project (single source of truth)."""

# Mapping from opera directory key to label/comune, used mainly in the report.
OPERE_INFO: dict[str, dict[str, str]] = {
    "P001_Sommacampagna": {
        "label": "P001 - SOMMACAMPAGNA",
        "comune": "SOMMACAMPAGNA (VR)",
    },
    "P002_Giuliari_Milani": {
        "label": "P002 - GIULIARI MILANI",
        "comune": "VERONA (VR)",
    },
    "P003_Gua": {"label": "P003 - GUA", "comune": "GUA (VR)"},
    "P004_Adige_Est": {"label": "P004 - ADIGE EST", "comune": "VERONA (VR)"},
    "P005_Adige_Ovest": {"label": "P005 - ADIGE OVEST", "comune": "VERONA (VR)"},
}

OPERE_TO_KEY: dict[str, str] = {
    "P001": "P001_Sommacampagna",
    "P002": "P002_Giuliari_Milani",
    "P003": "P003_Gua",
    "P004": "P004_Adige_Est",
    "P005": "P005_Adige_Ovest",
}


def get_opera_info(site_code: str) -> tuple[str, dict[str, str]]:
    """Return (opera_key, opera_info) for a site code."""
    opera_key = OPERE_TO_KEY.get(site_code, f"{site_code}_Unknown")
    if opera_key not in OPERE_INFO:
        return f"{site_code}_Unknown", {"label": site_code, "comune": ""}
    return opera_key, OPERE_INFO[opera_key]


MESI_IT: dict[int, str] = {
    1: "GENNAIO",
    2: "FEBBRAIO",
    3: "MARZO",
    4: "APRILE",
    5: "MAGGIO",
    6: "GIUGNO",
    7: "LUGLIO",
    8: "AGOSTO",
    9: "SETTEMBRE",
    10: "OTTOBRE",
    11: "NOVEMBRE",
    12: "DICEMBRE",
}

FONT_SIZE = 20

# Radians to milliradians conversion factor
DEG_TO_MRAD = 3.141592653589793 / 180 * 1000

# Colors previously defined in parameters.yaml
COLORS: dict[str, str] = {
    "base_blue": "#1b75b6",
    "dark_blue": "#0f4c75",
    "light_blue": "#3282b8",
}


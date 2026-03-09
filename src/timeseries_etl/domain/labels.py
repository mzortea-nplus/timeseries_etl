"""Label loading and formatting helpers."""

import os

import pandas as pd


def get_ylabel(sensor_id: str) -> str:
    """Return pretty y-label for raw units."""
    suffix = sensor_id.split("_")[-1]
    mapping = {
        "t": "Temperatura [°C]",
        "e": "Estensione [mm]",
        "s": "Spostamento [mm]",
        "x": "Rotazione longitudinale [mrad]",
        "y": "Rotazione trasversale [mrad]",
    }
    return mapping.get(suffix, sensor_id)


def load_label_dict(opera_key: str, data_dir: str = "data") -> dict[str, str]:
    """Load mapping ID -> human-readable label from label-id CSV."""
    label_path = os.path.join(data_dir, "label-id", f"{opera_key}_label-id.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"File label-id non trovato: {label_path}")

    label_df = pd.read_csv(label_path, sep=None, engine="python")
    if label_df.shape[1] < 2:
        raise ValueError(
            f"Il file {label_path} deve avere almeno 2 colonne (label, id)"
        )

    label_df.iloc[:, 0] = label_df.iloc[:, 0].astype(str).str.strip()
    label_df.iloc[:, 1] = label_df.iloc[:, 1].astype(str).str.strip()

    label_dict = {
        row_id.split("_")[0].upper(): label
        for label, row_id in zip(label_df.iloc[:, 0], label_df.iloc[:, 1])
    }
    print(f"✔ Caricate {len(label_dict)} associazioni ID → Label")
    return label_dict

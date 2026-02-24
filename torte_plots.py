"""
Torte-style pie charts using control data (output of data_preparation).
Generates "Within limits" vs "Out of limits" plots per sensor.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def run_torte_plots(
    control_csv_path: str,
    year: str,
    month: str,
    output_dir: str,
    opera_key: str = "P005",
) -> str:
    """
    Generate torte-style pie charts from control CSV.
    Each pie shows % of readings within control limits (|z|<=3) vs out of limits.
    Returns the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(control_csv_path)
    df["time"] = pd.to_datetime(df["time"])

    sensor_cols = [c for c in df.columns if c != "time"]
    if not sensor_cols:
        print("No sensor columns in control data, skipping torte plots")
        return output_dir

    num_sensors = len(sensor_cols)
    fig_width = max(5 * num_sensors, 6)
    fig_height = 6
    fig, axs = plt.subplots(1, num_sensors, figsize=(fig_width, fig_height))
    axs_list = [axs] if num_sensors == 1 else list(axs)

    for i, col in enumerate(sensor_cols):
        z = df[col].dropna()
        within = (np.abs(z) <= 3).sum()
        outside = (np.abs(z) > 3).sum()
        total = len(z) or 1
        within_pct = (within / total) * 100
        outside_pct = (outside / total) * 100

        if outside_pct < 0.1:
            pie_sizes = [100]
            pie_colors = ["#5fa3d4"]
            pie_autopct = "%1.1f%%"
        else:
            pie_sizes = [within_pct, outside_pct]
            pie_colors = ["#5fa3d4", "#F0F0F0"]
            pie_autopct = "%1.1f%%"

        axs_list[i].pie(
            pie_sizes,
            startangle=90,
            colors=pie_colors,
            autopct=pie_autopct,
            textprops={'fontsize': 18, 'color': 'black'},
        )
        circle = plt.Circle((0, 0), 1, color="black", fill=False, linewidth=1.0)
        axs_list[i].add_artist(circle)
        axs_list[i].axis("equal")
        axs_list[i].set_title(col, fontsize=14, y=0.97)

    axs_list[min(2, len(axs_list) - 1)].legend(
        ["Dati Disponibili", "Dati Mancanti"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        fancybox=True,
        edgecolor='black',
        facecolor='white',
        framealpha=1.0,
    )

    for ax in axs_list:
        ax.text(
            0.5, 0.5, 'Nplus',
            fontsize=60,
            color='gray',
            alpha=0.25,
            ha='center',
            va='center',
            rotation=30,
            transform=ax.transAxes,
            zorder=3000,
        )

    fig.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2)

    plot_name = f"{opera_key}_torte_{year}_{month}.png"
    out_path = os.path.join(output_dir, plot_name)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\033[92m✔ salvato {out_path}\033[0m")
    return output_dir


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python torte_plots.py <control_csv> <year> <month> <output_dir> [opera_key]")
        sys.exit(1)
    run_torte_plots(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
        opera_key=sys.argv[5] if len(sys.argv) > 5 else "P005",
    )

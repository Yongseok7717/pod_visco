from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Input and output paths
# ---------------------------------------------------------------------

csv_file = Path("results_longtime/probe_displacement_histories.csv")
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------

df = pd.read_csv(csv_file)

t = df["time"].to_numpy()

columns = [
    "DG-FOM",
    "DG-POD ell=10",
    "DG-POD ell=20",
    "DG-POD ell=50",
    "DG-POD ell=100",
    "DG-POD ell=200",
]

labels = [
    "DG-FOM",
    r"DG-POD, $\ell=10$",
    r"DG-POD, $\ell=20$",
    r"DG-POD, $\ell=50$",
    r"DG-POD, $\ell=100$",
    r"DG-POD, $\ell=200$",
]


# ---------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------

def plot_histories(t_min, t_max, output_name):
    mask = (t >= t_min) & (t <= t_max)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 7),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    # Common vertical range for all panels
    values = np.column_stack(
        [df[column].to_numpy()[mask] for column in columns]
    )

    y_min = np.min(values)
    y_max = np.max(values)
    padding = 0.05 * (y_max - y_min)

    for ax, column, label in zip(axes, columns, labels):
        u = df[column].to_numpy()

        ax.plot(
            t[mask],
            u[mask],
            linewidth=1.0,
            label=label,
        )

        ax.set_xlim(t_min, t_max)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="lower right",
            fontsize=9,
            frameon=True,
        )

    fig.suptitle(
        r"$u_1(\boldsymbol{x}_p,t)$ at "
        r"$\boldsymbol{x}_p=(0.203,0.497)$",
        fontsize=14,
    )

    fig.supxlabel(r"$t$", fontsize=13)
    fig.supylabel(r"$u_1(\boldsymbol{x}_p,t)$", fontsize=13)

    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.95])

    output_file = output_dir / output_name
    fig.savefig(
        output_file,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


# ---------------------------------------------------------------------
# Short-time and long-time figures
# ---------------------------------------------------------------------

plot_histories(
    t_min=0.0,
    t_max=1.0,
    output_name="probe_displacement_histories_short.pdf",
)

plot_histories(
    t_min=0.0,
    t_max=5.0,
    output_name="probe_displacement_histories_long.pdf",
)
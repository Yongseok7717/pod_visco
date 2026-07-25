from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
CSV_FILE = Path("results_temporal/dg_temporal_convergence.csv")
OUTPUT_DIR = Path("dg_temporal_plots")

ERROR_COL = "max_L2_error"
YLABEL = r"$\max_n \|u(t_n)-u_h^n\|_{L^2}$"

W_ORDER = ["I", "M", "A"]

SCHEME_ORDERS = {
    "Euler": 1,
    "Crank-Nicolson": 2,
}

MARKER_MAP = {
    "I": "o",
    "M": "s",
    "A": "^",
}


def display_label(W: str) -> str:
    return rf"DG-POD, $W_u={W}$"


def add_optimal_line(
    ax,
    x_values,
    y_anchor,
    order,
    label=None,
):
    """
    Since Delta t = T / Nt,
    O(Delta t^p) corresponds to O(Nt^{-p}).
    """
    x_ref = np.asarray(
        sorted(np.unique(x_values)),
        dtype=float,
    )

    x0 = x_ref[0]
    y_ref = y_anchor * (x_ref / x0) ** (-order)

    if label is None:
        label = rf"$O(\Delta t^{order})$"

    ax.loglog(
        x_ref,
        y_ref,
        linestyle="--",
        linewidth=1.5,
        label=label,
    )


def plot_temporal_case(
    df,
    scheme,
    output_dir,
):
    sub = df[df["scheme"] == scheme].copy()

    if sub.empty:
        print(f"No data found for {scheme}")
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    plotted_y_at_first_x = []

    
    # --------------------------------------------------------
    # DG-POD: I, M, A
    # --------------------------------------------------------
    for W in W_ORDER:
        rom = sub[
            (sub["method"] == "DG-POD")
            & (sub["W"] == W)
        ].sort_values("Nt")

        if rom.empty:
            continue

        ax.loglog(
            rom["Nt"],
            rom[ERROR_COL],
            marker=MARKER_MAP[W],
            linewidth=1.7,
            markersize=5,
            label=display_label(W),
        )

        plotted_y_at_first_x.append(
            float(rom.iloc[0][ERROR_COL])
        )

    # --------------------------------------------------------
    # Optimal temporal convergence line
    # --------------------------------------------------------
    order = SCHEME_ORDERS[scheme]

    if plotted_y_at_first_x:
        y_anchor = 0.65 * min(plotted_y_at_first_x)

        add_optimal_line(
            ax=ax,
            x_values=sub["Nt"].values,
            y_anchor=y_anchor,
            order=order,
        )

    ax.set_xlabel(r"$N_t$")
    ax.set_ylabel(YLABEL)
    ax.set_title(scheme)

    ax.grid(
        True,
        which="both",
        linewidth=0.5,
        alpha=0.5,
    )

    ax.legend(fontsize=9)

    # Display only the computed Nt values
    nt_values = sorted(sub["Nt"].unique())

    ax.set_xticks(nt_values)
    ax.set_xticklabels(
        [str(int(Nt)) for Nt in nt_values]
    )

    # Remove automatic minor ticks on the log-scaled x-axis
    ax.xaxis.set_minor_locator(
        plt.NullLocator()
    )

    fig.tight_layout()

    safe_scheme = (
        scheme.replace(" ", "_")
        .replace("-", "_")
    )

    outfile = (
        output_dir
        / f"{safe_scheme}_temporal_L2.png"
    )

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {outfile.resolve()}")


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CSV_FILE.resolve()}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(CSV_FILE)

    required_columns = {
        "method",
        "W",
        "Nt",
        "scheme",
        ERROR_COL,
    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(sorted(missing))
        )

    for scheme in [
        "Euler",
        "Crank-Nicolson",
    ]:
        plot_temporal_case(
            df=df,
            scheme=scheme,
            output_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
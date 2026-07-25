from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
CSV_FILE = Path("results_spatial/dg_spatial_convergence.csv")
OUTPUT_DIR = Path("dg_spatial_plots")

# Plot max-in-time errors. Change to rms_* if needed.
ERROR_SPECS = {
    "max_L2_error": {
        "ylabel": r"$\max_n \|u(t_n)-u_h^n\|_{L^2}$",
        "optimal_order": lambda k: k + 1,
    },
    "max_H1_error": {
        "ylabel": r"$\max_n \|u(t_n)-u_h^n\|_{H^1(\mathcal{T}_h)}$",
        "optimal_order": lambda k: k,
    },
}

W_ORDER = ["I", "M", "A"]


def display_label(method: str, W: str) -> str:
    if method == "DG-FOM":
        return "DG-FOM"

    if W == "A":
        W = r"A"

    return rf"DG-POD, $W_u={W}$"


def add_optimal_line(ax, x_values, y_anchor, order, label=None):
    """
    Since h ~ 1/Nxy, O(h^p) corresponds to O(Nxy^{-p}).
    """
    x_ref = np.asarray(sorted(np.unique(x_values)), dtype=float)
    x0 = x_ref[0]
    y_ref = y_anchor * (x_ref / x0) ** (-order)

    if label is None:
        label = rf"$O(h^{{{order}}})$"

    ax.loglog(
        x_ref,
        y_ref,
        linestyle="--",
        linewidth=1.5,
        label=label,
    )


def plot_one_case(df, scheme, k, error_col, spec, output_dir):
    sub = df[
        (df["scheme"] == scheme)
        & (df["k"] == k)
    ].copy()

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    plotted_y_at_first_x = []


    # ROM: I, M, A
    marker_map = {
    "I": "o",
    "M": "s",
    "A": "^",
    }
    for W in W_ORDER:
        rom = sub[
            (sub["method"] == "DG-POD")
            & (sub["W"] == W)
        ].sort_values("Nxy")

        if rom.empty:
            continue

        ax.loglog(
            rom["Nxy"],
            rom[error_col],
            marker=marker_map[W],
            linewidth=1.7,
            markersize=5,
            label=display_label("DG-POD", W),
        )
        plotted_y_at_first_x.append(float(rom.iloc[0][error_col]))

    # Optimal convergence line
    order = int(spec["optimal_order"](k))
    if plotted_y_at_first_x:
        # Put the reference line slightly below the numerical curves.
        y_anchor = 0.65 * min(plotted_y_at_first_x)
        add_optimal_line(
            ax,
            sub["Nxy"].values,
            y_anchor=y_anchor,
            order=order,
        )

    ax.set_xlabel(r"$N_{xy}$")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(rf"{scheme}, polynomial degree $k={k}$")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_xticks(sorted(sub["Nxy"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())

    fig.tight_layout()

    safe_scheme = scheme.replace(" ", "_").replace("-", "_")
    outfile = output_dir / f"{safe_scheme}_k{k}_{error_col}.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outfile}")


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CSV_FILE.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_FILE)

    required_columns = {
        "method",
        "W",
        "Nxy",
        "k",
        "scheme",
        *ERROR_SPECS.keys(),
    }
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(sorted(missing))
        )

    schemes = list(df["scheme"].dropna().unique())
    k_values = sorted(df["k"].dropna().unique())

    for scheme in schemes:
        for k in k_values:
            for error_col, spec in ERROR_SPECS.items():
                plot_one_case(
                    df=df,
                    scheme=scheme,
                    k=k,
                    error_col=error_col,
                    spec=spec,
                    output_dir=OUTPUT_DIR,
                )


if __name__ == "__main__":
    main()

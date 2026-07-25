from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class ModelParameters:
    T: float = 5.0

    # PDMS-like material parameters
    rho: float = 965.0
    mu: float = 455.0e3
    nu: float = 0.45

    # Relaxation function: varphi_0 + sum_q varphi_q exp(-t/tau_q)
    varphi0: float = 0.89
    varphi1: float = 0.08
    varphi2: float = 0.03
    tau1: float = 0.165
    tau2: float = 5.0

    # SIPDG parameters
    alpha: float = 10.0
    beta: float = 1.0

    @property
    def lam(self) -> float:
        """First Lamé parameter for plane strain."""
        return 2.0 * self.mu * self.nu / (1.0 - 2.0 * self.nu)


# -----------------------------------------------------------------------------
# Spatial discretization
# -----------------------------------------------------------------------------

def create_space(Nxy: int, k: int):
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        Nxy,
        Nxy,
        cell_type=mesh.CellType.triangle,
    )
    element = basix.ufl.element(
        "DG",
        domain.basix_cell(),
        k,
        shape=(2,),
    )
    return domain, fem.functionspace(domain, element)


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, p: ModelParameters):
    """Isotropic Hooke law: sigma(u)=2 mu eps(u)+lambda tr(eps(u)) I."""
    dim = u.ufl_shape[0]
    return (
        2.0 * p.mu * epsilon(u)
        + p.lam * ufl.tr(epsilon(u)) * ufl.Identity(dim)
    )


def mark_boundaries(domain):
    """Tag the Dirichlet boundary by 1 and the loaded boundary by 2."""
    fdim = domain.topology.dim - 1

    facets_D = mesh.locate_entities_boundary(
        domain,
        fdim,
        lambda x: (
            np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        ),
    )
    facets_N = mesh.locate_entities_boundary(
        domain,
        fdim,
        lambda x: np.isclose(x[0], 0.0),
    )

    facets = np.hstack((facets_D, facets_N)).astype(np.int32)
    values = np.hstack(
        (
            np.full(facets_D.size, 1, dtype=np.int32),
            np.full(facets_N.size, 2, dtype=np.int32),
        )
    )
    order = np.argsort(facets)
    return mesh.meshtags(domain, fdim, facets[order], values[order])


def petsc_to_csr(A):
    ai, aj, av = A.getValuesCSR()
    return csr_matrix((av.copy(), aj.copy(), ai.copy()), shape=A.getSize())


def assemble_operators(
    domain,
    V,
    p: ModelParameters,
    impulse_amplitude: float,
):
    """Assemble M, Hooke-tensor SIPDG operator A, jump operator J, and load."""
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    n = ufl.FacetNormal(domain)
    h = ufl.CellDiameter(domain)
    h_avg = 0.5 * (h("+") + h("-"))

    tags = mark_boundaries(domain)
    dx = ufl.dx(domain=domain)
    dS = ufl.dS(domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)

    # Elasticity-dependent SIPDG penalty.
    penalty_scale = p.lam + 2.0 * p.mu
    penalty_int = p.alpha * penalty_scale / h_avg**p.beta
    penalty_bnd = p.alpha * penalty_scale / h**p.beta

    M_form = fem.form(ufl.inner(u, v) * dx)

    A_form = fem.form(
        ufl.inner(sigma(u, p), epsilon(v)) * dx
        - ufl.inner(
            ufl.avg(sigma(u, p)),
            ufl.outer(v("+"), n("+")) + ufl.outer(v("-"), n("-")),
        ) * dS
        - ufl.inner(
            ufl.avg(sigma(v, p)),
            ufl.outer(u("+"), n("+")) + ufl.outer(u("-"), n("-")),
        ) * dS
        + penalty_int * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS
        - ufl.inner(sigma(u, p), ufl.outer(v, n)) * ds(1)
        - ufl.inner(ufl.outer(u, n), sigma(v, p)) * ds(1)
        + penalty_bnd * ufl.inner(u, v) * ds(1)
    )

    # The same penalty form acts on the velocity in the semidiscrete model.
    J_form = fem.form(
        penalty_int * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS
        + penalty_bnd * ufl.inner(u, v) * ds(1)
    )

    matrices = []
    for form in (M_form, A_form, J_form):
        Ap = assemble_matrix(form)
        Ap.assemble()
        matrices.append(petsc_to_csr(Ap))
        Ap.destroy()
    M, A, J = matrices

    x = ufl.SpatialCoordinate(domain)
    traction = ufl.as_vector(
        (impulse_amplitude * ufl.sin(ufl.pi * x[1]), 0.0)
    )
    bp = assemble_vector(fem.form(ufl.inner(traction, v) * ds(2)))
    bp.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    Gimp = bp.array.copy()
    bp.destroy()

    return M, A, J, Gimp


# -----------------------------------------------------------------------------
# Time stepping
# -----------------------------------------------------------------------------

def zero_state(V):
    ndofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    z = np.zeros(ndofs, dtype=np.float64)
    return z.copy(), z.copy(), z.copy(), z.copy()


def theta_coefficients(dt: float, omega: float, p: ModelParameters):
    d1 = dt / p.tau1
    d2 = dt / p.tau2
    den1 = 1.0 + omega * d1
    den2 = 1.0 + omega * d2
    gamma1 = omega * p.varphi1 / den1
    gamma2 = omega * p.varphi2 / den2
    return d1, d2, den1, den2, gamma1, gamma2


def advance_state(
    u,
    v,
    z1,
    z2,
    M,
    A,
    J,
    solver,
    Gimp,
    nstep,
    dt,
    omega,
    p,
    coeffs,
):
    d1, d2, den1, den2, gamma1, gamma2 = coeffs

    rhs = p.rho / (omega * dt * dt) * (M @ u)
    rhs += p.rho / (omega * dt) * (M @ v)
    rhs += -(1.0 - omega) * p.varphi0 * (A @ u)
    rhs += (1.0 / dt) * (J @ u)
    rhs += gamma1 * (A @ u) - (1.0 / den1) * (A @ z1)
    rhs += gamma2 * (A @ u) - (1.0 / den2) * (A @ z2)

    if nstep == 0:
        # Fixed total boundary impulse, independent of dt.
        rhs += Gimp / dt

    u_new = solver.solve(rhs)
    v_new = (
        (u_new - u) / (omega * dt)
        - (1.0 - omega) / omega * v
    )
    v_avg = (u_new - u) / dt

    z1_new = (
        (1.0 - (1.0 - omega) * d1) / den1 * z1
        + dt * p.varphi1 / den1 * v_avg
    )
    z2_new = (
        (1.0 - (1.0 - omega) * d2) / den2 * z2
        + dt * p.varphi2 / den2 * v_avg
    )

    return u_new, v_new, z1_new, z2_new


# -----------------------------------------------------------------------------
# Probe evaluation
# -----------------------------------------------------------------------------

class ProbeEvaluator:
    """Evaluate u_1 at one physical point (x,y)."""

    def __init__(self, V, probe_point):
        if len(probe_point) != 2:
            raise ValueError("probe_point must contain exactly two coordinates")

        self.uh = fem.Function(V)
        self.x = np.array(
            [[probe_point[0], probe_point[1], 0.0]],
            dtype=np.float64,
        )

        domain = V.mesh
        tree = geometry.bb_tree(domain, domain.topology.dim)
        candidates = geometry.compute_collisions_points(tree, self.x)
        colliding = geometry.compute_colliding_cells(
            domain,
            candidates,
            self.x,
        )
        cells = colliding.links(0)
        if cells.size == 0:
            raise RuntimeError(f"Probe point {probe_point} is outside the mesh")

        self.cell = np.array([cells[0]], dtype=np.int32)

    def evaluate_u1(self, coefficients) -> float:
        self.uh.x.array[:] = coefficients
        self.uh.x.scatter_forward()
        value = np.asarray(
            self.uh.eval(self.x, self.cell),
            dtype=np.float64,
        ).reshape(-1)
        if value.size < 2:
            raise RuntimeError("Expected a two-component DG function value")
        return float(value[0])

    def evaluate_basis_u1(self, Phi: np.ndarray) -> np.ndarray:
        return np.array(
            [self.evaluate_u1(Phi[:, j]) for j in range(Phi.shape[1])],
            dtype=np.float64,
        )


# -----------------------------------------------------------------------------
# Snapshot FOM, reference FOM, and POD
# -----------------------------------------------------------------------------

def run_snapshot_fom(
    Nxy: int,
    k: int,
    Nt: int,
    p: ModelParameters,
    impulse_amplitude: float,
    omega: float,
):
    """Run the coarse FOM and store every displacement state as a snapshot."""
    dt = p.T / Nt

    setup_start = time.perf_counter()
    domain, V = create_space(Nxy, k)
    M, A, J, Gimp = assemble_operators(domain, V, p, impulse_amplitude)
    u, v, z1, z2 = zero_state(V)
    coeffs = theta_coefficients(dt, omega, p)

    _, _, _, _, gamma1, gamma2 = coeffs
    B = (
        p.rho / (omega * dt * dt) * M
        + omega * p.varphi0 * A
        + (gamma1 + gamma2) * A
        + (1.0 / dt) * J
    ).tocsc()
    solver = splu(B)
    setup_time = time.perf_counter() - setup_start

    snapshots = np.empty((u.size, Nt + 1), dtype=np.float64)
    snapshots[:, 0] = u

    solve_time = 0.0
    march_start = time.perf_counter()
    for nstep in range(Nt):
        solve_start = time.perf_counter()
        u, v, z1, z2 = advance_state(
            u, v, z1, z2,
            M, A, J, solver, Gimp,
            nstep, dt, omega, p, coeffs,
        )
        solve_time += time.perf_counter() - solve_start
        snapshots[:, nstep + 1] = u

    march_time = time.perf_counter() - march_start

    return {
        "domain": domain,
        "V": V,
        "M": M,
        "A": A,
        "J": J,
        "Gimp": Gimp,
        "Y": snapshots,
        "times": np.linspace(0.0, p.T, Nt + 1),
        "Nt": Nt,
        "dt": dt,
        "ndofs": u.size,
        "setup_time_sec": setup_time,
        "solve_time_sec": solve_time,
        "march_time_sec": march_time,
        "total_time_sec": setup_time + march_time,
    }


def run_reference_fom(
    Nxy: int,
    k: int,
    Nt: int,
    p: ModelParameters,
    impulse_amplitude: float,
    probe_point,
    omega: float,
):
    """Run the fine DG-FOM and record u_1(x_p,t) at every time level."""
    dt = p.T / Nt

    setup_start = time.perf_counter()
    domain, V = create_space(Nxy, k)
    M, A, J, Gimp = assemble_operators(domain, V, p, impulse_amplitude)
    u, v, z1, z2 = zero_state(V)
    coeffs = theta_coefficients(dt, omega, p)

    _, _, _, _, gamma1, gamma2 = coeffs
    B = (
        p.rho / (omega * dt * dt) * M
        + omega * p.varphi0 * A
        + (gamma1 + gamma2) * A
        + (1.0 / dt) * J
    ).tocsc()
    solver = splu(B)
    probe = ProbeEvaluator(V, probe_point)
    setup_time = time.perf_counter() - setup_start

    times = np.linspace(0.0, p.T, Nt + 1)
    probe_u1 = np.empty(Nt + 1, dtype=np.float64)
    probe_u1[0] = probe.evaluate_u1(u)

    solve_time = 0.0
    march_start = time.perf_counter()
    for nstep in range(Nt):
        solve_start = time.perf_counter()
        u, v, z1, z2 = advance_state(
            u, v, z1, z2,
            M, A, J, solver, Gimp,
            nstep, dt, omega, p, coeffs,
        )
        solve_time += time.perf_counter() - solve_start
        probe_u1[nstep + 1] = probe.evaluate_u1(u)

    march_time = time.perf_counter() - march_start

    return {
        "times": times,
        "probe_u1": probe_u1,
        "Nt": Nt,
        "dt": dt,
        "ndofs": u.size,
        "setup_time_sec": setup_time,
        "solve_time_sec": solve_time,
        "march_time_sec": march_time,
        "total_time_sec": setup_time + march_time,
    }


def compute_identity_pod(Y: np.ndarray, tol: float = 1.0e-13):
    """Compute W_u=I POD from the correlation eigenproblem Y^T Y."""
    start = time.perf_counter()

    K = Y.T @ Y
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    singular_values = np.sqrt(eigenvalues)
    if singular_values.size == 0:
        raise RuntimeError("The snapshot matrix has no POD modes")

    keep = singular_values > tol * max(singular_values[0], 1.0)
    singular_values = singular_values[keep]
    eigenvectors = eigenvectors[:, keep]

    Phi = Y @ (eigenvectors / singular_values[None, :])
    pod_time = time.perf_counter() - start
    return Phi, singular_values, pod_time


# -----------------------------------------------------------------------------
# DG-POD for one ell
# -----------------------------------------------------------------------------

def run_rom(
    snapshot_data,
    Phi_all: np.ndarray,
    svals: np.ndarray,
    pod_time: float,
    ell: int,
    Nt: int,
    p: ModelParameters,
    probe_point,
    omega: float,
):
    if ell > Phi_all.shape[1]:
        raise ValueError(
            f"ell={ell} exceeds the available POD rank {Phi_all.shape[1]}"
        )

    dt = p.T / Nt
    Phi = Phi_all[:, :ell]
    M = snapshot_data["M"]
    A = snapshot_data["A"]
    J = snapshot_data["J"]
    Gimp = snapshot_data["Gimp"]
    V = snapshot_data["V"]

    setup_start = time.perf_counter()
    Mr = Phi.T @ (M @ Phi)
    Ar = Phi.T @ (A @ Phi)
    Jr = Phi.T @ (J @ Phi)
    Gimp_r = Phi.T @ Gimp

    au = np.zeros(ell, dtype=np.float64)
    av = np.zeros(ell, dtype=np.float64)
    az1 = np.zeros(ell, dtype=np.float64)
    az2 = np.zeros(ell, dtype=np.float64)

    coeffs = theta_coefficients(dt, omega, p)
    d1, d2, den1, den2, gamma1, gamma2 = coeffs
    Br = (
        p.rho / (omega * dt * dt) * Mr
        + omega * p.varphi0 * Ar
        + (gamma1 + gamma2) * Ar
        + (1.0 / dt) * Jr
    )
    solver = splu(csc_matrix(Br))

    probe = ProbeEvaluator(V, probe_point)
    probe_basis_u1 = probe.evaluate_basis_u1(Phi)
    reduced_setup_time = time.perf_counter() - setup_start

    times = np.linspace(0.0, p.T, Nt + 1)
    probe_u1 = np.empty(Nt + 1, dtype=np.float64)
    probe_u1[0] = float(probe_basis_u1 @ au)

    solve_time = 0.0
    march_start = time.perf_counter()
    for nstep in range(Nt):
        rhs = p.rho / (omega * dt * dt) * (Mr @ au)
        rhs += p.rho / (omega * dt) * (Mr @ av)
        rhs += -(1.0 - omega) * p.varphi0 * (Ar @ au)
        rhs += (1.0 / dt) * (Jr @ au)
        rhs += gamma1 * (Ar @ au) - (1.0 / den1) * (Ar @ az1)
        rhs += gamma2 * (Ar @ au) - (1.0 / den2) * (Ar @ az2)
        if nstep == 0:
            rhs += Gimp_r / dt

        solve_start = time.perf_counter()
        au_new = solver.solve(rhs)
        solve_time += time.perf_counter() - solve_start

        av_new = (
            (au_new - au) / (omega * dt)
            - (1.0 - omega) / omega * av
        )
        v_avg = (au_new - au) / dt
        az1_new = (
            (1.0 - (1.0 - omega) * d1) / den1 * az1
            + dt * p.varphi1 / den1 * v_avg
        )
        az2_new = (
            (1.0 - (1.0 - omega) * d2) / den2 * az2
            + dt * p.varphi2 / den2 * v_avg
        )

        au, av, az1, az2 = au_new, av_new, az1_new, az2_new
        probe_u1[nstep + 1] = float(probe_basis_u1 @ au)

    march_time = time.perf_counter() - march_start

    tail_abs = (
    np.sqrt(np.sum(svals[ell:] ** 2))
    if ell < len(svals)
    else 0.0
    )
    total_energy = np.sqrt(np.sum(svals**2))
    tail = (
        tail_abs / total_energy
        if total_energy > 0.0
        else 0.0
    )

    return {
        "ell": ell,
        "times": times,
        "probe_u1": probe_u1,
        "pod_tail": tail,
        "pod_time_sec": pod_time,
        "reduced_setup_time_sec": reduced_setup_time,
        "setup_time_sec": pod_time + reduced_setup_time,
        "solve_time_sec": solve_time,
        "march_time_sec": march_time,
        "total_time_sec": pod_time + reduced_setup_time + march_time,
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_snapshot_checkpoint(snapshot_data, outdir: Path):
    """Save the expensive snapshot matrix immediately after its computation."""
    np.savez_compressed(
        outdir / "snapshot_fom.npz",
        times=snapshot_data["times"],
        Y=snapshot_data["Y"],
        Nt=snapshot_data["Nt"],
        dt=snapshot_data["dt"],
    )


def save_reference_checkpoint(reference_data, outdir: Path):
    np.savez_compressed(
        outdir / "reference_probe_history.npz",
        times=reference_data["times"],
        probe_u1=reference_data["probe_u1"],
        Nt=reference_data["Nt"],
        dt=reference_data["dt"],
    )


def save_rom_checkpoint(rom_data, outdir: Path):
    ell = rom_data["ell"]
    np.savez_compressed(
        outdir / f"rom_ell_{ell}.npz",
        times=rom_data["times"],
        probe_u1=rom_data["probe_u1"],
        ell=ell,
        pod_tail=rom_data["pod_tail"],
    )


def save_probe_histories(reference_data, rom_results, outdir: Path):
    times = reference_data["times"]
    columns = {
        "time": times,
        "DG-FOM": reference_data["probe_u1"],
    }

    for ell, rom in rom_results.items():
        if not np.allclose(times, rom["times"]):
            raise RuntimeError(f"Time grids differ for ell={ell}")
        columns[f"DG-POD ell={ell}"] = rom["probe_u1"]

    frame = pd.DataFrame(columns)
    frame.to_csv(
        outdir / "probe_displacement_histories.csv",
        index=False,
        float_format="%.12e",
    )
    return frame


def plot_probe_histories(history, probe_point, ell_values, outdir: Path):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.0, 8.0),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()
    t = history["time"].to_numpy()

    axes[0].plot(t, history["DG-FOM"].to_numpy(), linewidth=1.0)
    axes[0].set_title("DG-FOM")
    axes[0].legend(["DG-FOM"], loc="lower right")

    for ax, ell in zip(axes[1:], ell_values):
        name = f"DG-POD ell={ell}"
        ax.plot(t, history[name].to_numpy(), linewidth=1.0)
        ax.set_title(rf"DG-POD, $\ell={ell}$")
        ax.legend([rf"DG-POD, $\ell={ell}$"], loc="lower right")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        rf"$u_1(\boldsymbol{{x}}_p,t)$ at "
        rf"$\boldsymbol{{x}}_p=({probe_point[0]:.3f},{probe_point[1]:.3f})$"
    )
    fig.supxlabel(r"$t$")
    fig.supylabel(r"$u_1(\boldsymbol{x}_p,t)$")
    fig.tight_layout()

    fig.savefig(outdir / "probe_displacement_histories.pdf", bbox_inches="tight")
    fig.savefig(
        outdir / "probe_displacement_histories.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_performance_table(
    snapshot_data,
    reference_data,
    rom_results,
    p: ModelParameters,
    Nxy: int,
    k: int,
    outdir: Path,
):
    rows = [
        {
            "method": "Snapshot DG-FOM",
            "ell": np.nan,
            "Nxy": Nxy,
            "k": k,
            "Nt": snapshot_data["Nt"],
            "dt": snapshot_data["dt"],
            "setup_time_sec": snapshot_data["setup_time_sec"],
            "solve_time_sec": snapshot_data["solve_time_sec"],
            "total_time_sec": snapshot_data["total_time_sec"],
            "setup_speedup": np.nan,
            "solve_speedup": np.nan,
            "total_speedup": np.nan,
            "pod_tail": np.nan,
        },
        {
            "method": "Reference DG-FOM",
            "ell": np.nan,
            "Nxy": Nxy,
            "k": k,
            "Nt": reference_data["Nt"],
            "dt": reference_data["dt"],
            "setup_time_sec": reference_data["setup_time_sec"],
            "solve_time_sec": reference_data["solve_time_sec"],
            "total_time_sec": reference_data["total_time_sec"],
            "setup_speedup": 1.0,
            "solve_speedup": 1.0,
            "total_speedup": 1.0,
            "pod_tail": np.nan,
        },
    ]

    for ell, rom in rom_results.items():
        rows.append(
            {
                "method": "DG-POD",
                "ell": ell,
                "Nxy": Nxy,
                "k": k,
                "Nt": reference_data["Nt"],
                "dt": reference_data["dt"],
                "setup_time_sec": rom["setup_time_sec"],
                "solve_time_sec": rom["solve_time_sec"],
                "total_time_sec": rom["total_time_sec"],
                "setup_speedup": (
                    reference_data["setup_time_sec"] / rom["setup_time_sec"]
                ),
                "solve_speedup": (
                    reference_data["solve_time_sec"] / rom["solve_time_sec"]
                ),
                "total_speedup": (
                    reference_data["total_time_sec"] / rom["total_time_sec"]
                ),
                "pod_tail": rom["pod_tail"],
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(
        outdir / "rom_performance_and_pod_tail.csv",
        index=False,
        float_format="%.12e",
    )
    return table


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

def main():
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run this script in serial")

    p = ModelParameters()

    Nxy = 64
    k = 2
    snapshot_Nt = 1000
    reference_Nt = 8000
    ell_values = [10, 20, 50, 100, 200]
    impulse_amplitude = 1.0e3
    probe_point = (0.203, 0.497)
    omega = 0.5

    outdir = Path("results_longtime")
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"Material: mu={p.mu:.6e}, nu={p.nu:.3f}, lambda={p.lam:.6e}",
        flush=True,
    )

    # 1. One coarse FOM trajectory: every state is a POD snapshot.
    print(f"Snapshot DG-FOM: Nt={snapshot_Nt}", flush=True)
    snapshot_data = run_snapshot_fom(
        Nxy,
        k,
        snapshot_Nt,
        p,
        impulse_amplitude,
        omega,
    )
    save_snapshot_checkpoint(snapshot_data, outdir)

    # 2. Compute the Euclidean POD basis once.
    print("Computing W_u=I POD basis by eigh(Y^T Y)", flush=True)
    Phi_all, svals, pod_time = compute_identity_pod(snapshot_data["Y"])
    np.savez_compressed(
        outdir / "pod_basis.npz",
        Phi=Phi_all,
        singular_values=svals,
        pod_time_sec=pod_time,
    )

    if max(ell_values) > Phi_all.shape[1]:
        raise RuntimeError(
            f"Requested ell={max(ell_values)}, but the POD rank is "
            f"{Phi_all.shape[1]}"
        )

    # 3. Fine DG-FOM reference: record the probe at every time level.
    print(f"Reference DG-FOM: Nt={reference_Nt}", flush=True)
    reference_data = run_reference_fom(
        Nxy,
        k,
        reference_Nt,
        p,
        impulse_amplitude,
        probe_point,
        omega,
    )
    save_reference_checkpoint(reference_data, outdir)

    # 4. Reuse the same POD basis and vary ell only.
    rom_results = {}
    for ell in ell_values:
        print(f"DG-POD: ell={ell}, Nt={reference_Nt}", flush=True)
        rom = run_rom(
            snapshot_data,
            Phi_all,
            svals,
            pod_time,
            ell,
            reference_Nt,
            p,
            probe_point,
            omega,
        )
        rom_results[ell] = rom
        save_rom_checkpoint(rom, outdir)

    history = save_probe_histories(reference_data, rom_results, outdir)
    plot_probe_histories(history, probe_point, ell_values, outdir)
    timing = save_performance_table(
        snapshot_data,
        reference_data,
        rom_results,
        p,
        Nxy,
        k,
        outdir,
    )

    print("\nPerformance summary")
    print(timing.to_string(index=False))
    print(f"\nResults written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
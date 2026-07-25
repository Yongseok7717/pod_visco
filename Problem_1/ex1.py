"""
FEniCSx DG--POD ROM spatial convergence with inexact coarse-FOM snapshots.

This script follows the legacy dolfin update structure:

    B u^{n+1} = rhs^n,

then updates

    v^{n+1}, z_q^{n+1}

algebraically.  The ROM snapshots are obtained from a coarse FOM time march rather than
from direct interpolation of the manufactured exact displacement.

Errors are computed against the exact displacement u(t) using
    L2 norm:           sqrt(e^T M e),
    broken H1 norm:    sqrt(e^T (M+G) e),
where G represents the element-wise gradient contribution.

The number of snapshots Ns and POD dimension ell are fixed in both temporal
and spatial convergence tests.

Recommended environment
-----------------------
conda create -n visco-rom python=3.11
conda activate visco-rom
conda install -c conda-forge fenics-dolfinx basix petsc4py mpi4py numpy scipy matplotlib pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from mpi4py import MPI

import basix.ufl
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import solve as dense_solve


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

@dataclass
class ModelParameters:
    T: float = 10.0
    rho: float = 1.0

    varphi1: float = 0.1
    varphi2: float = 0.4
    varphi0: float = 0.5

    tau1: float = 0.5
    tau2: float = 1.5

    alpha: float = 10.0
    beta: float = 1.5

    # Fixed POD settings
    Ns: int = 100
    ell: int = 5

    outdir: str = "ex1"


# ---------------------------------------------------------------------
# Manufactured exact fields
# ---------------------------------------------------------------------

def exact_u_values(x: np.ndarray, t: float) -> np.ndarray:
    X, Y = x[0], x[1]
    out = np.zeros((2, x.shape[1]), dtype=np.float64)
    out[0] = X * Y * np.exp(1.0 - t)
    out[1] = np.cos(t) * np.sin(X * Y)
    return out


def exact_v_values(x: np.ndarray, t: float) -> np.ndarray:
    X, Y = x[0], x[1]
    out = np.zeros((2, x.shape[1]), dtype=np.float64)
    out[0] = -X * Y * np.exp(1.0 - t)
    out[1] = -np.sin(t) * np.sin(X * Y)
    return out


def exact_acc_values(x: np.ndarray, t: float) -> np.ndarray:
    X, Y = x[0], x[1]
    out = np.zeros((2, x.shape[1]), dtype=np.float64)
    out[0] = X * Y * np.exp(1.0 - t)
    out[1] = -np.cos(t) * np.sin(X * Y)
    return out


def exact_zeta_values(x: np.ndarray, t: float, varphi: float, tau: float) -> np.ndarray:
    """
    Closed-form internal variable, following the legacy dolfin expressions.
    """
    X, Y = x[0], x[1]
    out = np.zeros((2, x.shape[1]), dtype=np.float64)

    if abs(tau - 1.0) < 1.0e-14:
        out[0] = -varphi * X * Y * t * np.exp(1.0 - t)
    else:
        out[0] = varphi * X * Y * (
            tau * np.exp(-t + 1.0) / (tau - 1.0)
            - tau * np.exp(-t / tau + 1.0) / (tau - 1.0)
        )

    out[1] = -(
        tau * tau * np.exp(-t / tau) / (tau * tau + 1.0)
        - (tau * tau * np.cos(t) - tau * np.sin(t)) / (tau * tau + 1.0)
    ) * varphi * np.sin(X * Y)

    return out


# ---------------------------------------------------------------------
# FEniCSx space and matrices
# ---------------------------------------------------------------------

def create_space(Nxy: int, k: int):
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        Nxy,
        Nxy,
        cell_type=mesh.CellType.triangle,
    )
    element = basix.ufl.element("DG", domain.basix_cell(), k, shape=(2,))
    V = fem.functionspace(domain, element)
    return domain, V

def create_exact_space(domain, degree=5):
    element = basix.ufl.element("Lagrange", domain.basix_cell(), degree, shape=(2,))
    return fem.functionspace(domain, element)

def mark_dirichlet_boundary(domain):
    fdim = domain.topology.dim - 1

    def gamma_D(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)

    facets = mesh.locate_entities_boundary(domain, fdim, gamma_D)
    values = np.full(len(facets), 1, dtype=np.int32)
    order = np.argsort(facets)
    return mesh.meshtags(domain, fdim, facets[order], values[order])


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def assemble_matrices(domain, V, p: ModelParameters):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    n = ufl.FacetNormal(domain)
    h = ufl.CellDiameter(domain)
    h_avg = (h("+") + h("-")) / 2.0

    facet_tags = mark_dirichlet_boundary(domain)
    dx = ufl.dx(domain=domain)
    dS = ufl.dS(domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    M_form = fem.form(ufl.inner(u, v) * dx)

    G_form = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)

    A_form = fem.form(
        ufl.inner(epsilon(u), epsilon(v)) * dx
        - ufl.inner(
            ufl.avg(epsilon(u)),
            ufl.outer(v("+"), n("+")) + ufl.outer(v("-"), n("-")),
        ) * dS
        - ufl.inner(
            ufl.avg(epsilon(v)),
            ufl.outer(u("+"), n("+")) + ufl.outer(u("-"), n("-")),
        ) * dS
        + p.alpha / (h_avg**p.beta) * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS
        - ufl.inner(epsilon(u), ufl.outer(v, n)) * ds(1)
        - ufl.inner(ufl.outer(u, n), epsilon(v)) * ds(1)
        + p.alpha / (h**p.beta) * ufl.inner(u, v) * ds(1)
    )

    J_form = fem.form(
        p.alpha / (h_avg**p.beta) * ufl.inner(ufl.jump(u), ufl.jump(v)) * dS
        + p.alpha / (h**p.beta) * ufl.inner(u, v) * ds(1)
    )

    mats = []
    for form in [M_form, G_form, A_form, J_form]:
        A_petsc = assemble_matrix(form)
        A_petsc.assemble()
        mats.append(petsc_to_csr(A_petsc))
        A_petsc.destroy()

    M, G, A, J = mats
    return M, G, A, J


def petsc_to_csr(A) -> csr_matrix:
    ai, aj, av = A.getValuesCSR()
    return csr_matrix((av.copy(), aj.copy(), ai.copy()), shape=A.getSize())


# ---------------------------------------------------------------------
# Exact vectors and snapshots
# ---------------------------------------------------------------------

def interpolate_exact(V, callback, t: float, *args) -> np.ndarray:
    f = fem.Function(V)
    f.interpolate(lambda x: callback(x, t, *args))
    return f.x.array.copy()


def generate_snapshots(V, times: np.ndarray, callback, *args) -> np.ndarray:
    y0 = interpolate_exact(V, callback, float(times[0]), *args)
    Y = np.zeros((len(y0), len(times)), dtype=np.float64)
    Y[:, 0] = y0
    for j, tj in enumerate(times[1:], start=1):
        Y[:, j] = interpolate_exact(V, callback, float(tj), *args)
    return Y


def exact_force_vector(V, M, A, J, p: ModelParameters, t: float) -> np.ndarray:
    """
    Semi-discrete manufactured forcing:
        F = rho M u_tt + varphi0 A u + A z1 + A z2 + J u_t.
    """
    u = interpolate_exact(V, exact_u_values, t)
    v = interpolate_exact(V, exact_v_values, t)
    acc = interpolate_exact(V, exact_acc_values, t)
    z1 = interpolate_exact(V, exact_zeta_values, t, p.varphi1, p.tau1)
    z2 = interpolate_exact(V, exact_zeta_values, t, p.varphi2, p.tau2)

    return (
        p.rho * (M @ acc)
        + p.varphi0 * (A @ u)
        + A @ z1
        + A @ z2
        + J @ v
    )


# ---------------------------------------------------------------------
# POD basis
# ---------------------------------------------------------------------

def pod_standard(Y: np.ndarray, tol: float = 1.0e-13):
    K = Y.T @ Y

    lam, V = np.linalg.eigh(K)
    idx = np.argsort(lam)[::-1]

    lam = np.maximum(lam[idx], 0.0)
    V = V[:, idx]
    s = np.sqrt(lam)

    keep = s > tol * max(s[0], 1.0)
    s = s[keep]
    V = V[:, keep]

    Phi = Y @ (V / s[None, :])
    return Phi, s


def pod_weighted(Y: np.ndarray, W: csr_matrix, tol: float = 1.0e-13):
    WY = W @ Y
    K = Y.T @ WY
    lam, V = np.linalg.eigh(K)
    idx = np.argsort(lam)[::-1]
    lam = np.maximum(lam[idx], 0.0)
    V = V[:, idx]
    s = np.sqrt(lam)

    keep = s > tol * max(s[0], 1.0)
    s = s[keep]
    V = V[:, keep]

    Phi = Y @ (V / s[None, :])
    return Phi, s


def build_basis(Y: np.ndarray, W_name: str, M: csr_matrix, A: csr_matrix, J: csr_matrix, ell: int):
    if W_name == "I":
        Phi_all, s = pod_standard(Y)
    elif W_name == "M":
        Phi_all, s = pod_weighted(Y, M)
    elif W_name == "A":
        Phi_all, s = pod_weighted(Y, A)
    else:
        raise ValueError(f"Unknown W_name={W_name}")

    if ell > Phi_all.shape[1]:
        raise ValueError(f"ell={ell} exceeds available POD rank {Phi_all.shape[1]} for W={W_name}")

    return Phi_all[:, :ell], s


def initial_coeff(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Use Euclidean least-squares coordinates for all POD bases.
    This is robust even when Phi is not Euclidean-orthonormal.
    """
    return np.linalg.lstsq(Phi, y, rcond=None)[0]


def generate_fom_snapshots(
    V,
    M: csr_matrix,
    A: csr_matrix,
    J: csr_matrix,
    p: ModelParameters,
    Nt_snapshot: int,
    Ns: int,
    omega: float,
) -> np.ndarray:
    """Generate ``Ns+1`` inexact displacement snapshots from a coarse FOM march.

    The snapshots include the initial state and are sampled approximately
    uniformly from the ``Nt_snapshot+1`` coarse time levels.  No exact-solution
    error evaluation is performed during this offline march.
    """
    if Nt_snapshot < 1:
        raise ValueError("Nt_snapshot must be positive")
    if Ns < 1:
        raise ValueError("Ns must be positive")
    if Ns > Nt_snapshot:
        raise ValueError(
            f"Ns={Ns} cannot exceed Nt_snapshot={Nt_snapshot} when snapshots "
            "are sampled from coarse FOM time levels"
        )

    T = p.T
    dt = T / Nt_snapshot

    u = interpolate_exact(V, exact_u_values, 0.0)
    v = interpolate_exact(V, exact_v_values, 0.0)
    z1 = interpolate_exact(V, exact_zeta_values, 0.0, p.varphi1, p.tau1)
    z2 = interpolate_exact(V, exact_zeta_values, 0.0, p.varphi2, p.tau2)

    sample_steps = np.rint(np.linspace(0, Nt_snapshot, Ns + 1)).astype(int)
    if np.unique(sample_steps).size != Ns + 1:
        raise ValueError("Snapshot time indices are not distinct; decrease Ns")

    Y = np.empty((len(u), Ns + 1), dtype=np.float64)
    Y[:, 0] = u
    sample_col = 1

    d1 = dt / p.tau1
    d2 = dt / p.tau2
    den1 = 1.0 + omega * d1
    den2 = 1.0 + omega * d2
    gamma1 = omega * p.varphi1 / den1
    gamma2 = omega * p.varphi2 / den2

    B = (
        p.rho / (omega * dt * dt) * M
        + omega * p.varphi0 * A
        + (gamma1 + gamma2) * A
        + (1.0 / dt) * J
    ).tocsc()
    B_lu = splu(B)

    for n in range(Nt_snapshot):
        tn = n * dt
        tnp1 = (n + 1) * dt

        F_n = exact_force_vector(V, M, A, J, p, tn)
        F_np1 = exact_force_vector(V, M, A, J, p, tnp1)
        Favg = omega * F_np1 + (1.0 - omega) * F_n

        rhs = Favg.copy()
        rhs += p.rho / (omega * dt * dt) * (M @ u)
        rhs += p.rho / (omega * dt) * (M @ v)
        rhs += -(1.0 - omega) * p.varphi0 * (A @ u)
        rhs += (1.0 / dt) * (J @ u)
        rhs += gamma1 * (A @ u) - (1.0 / den1) * (A @ z1)
        rhs += gamma2 * (A @ u) - (1.0 / den2) * (A @ z2)

        u_new = B_lu.solve(rhs)
        v_new = (u_new - u) / (omega * dt) - (1.0 - omega) / omega * v
        v_avg = (u_new - u) / dt
        z1_new = (
            (1.0 - (1.0 - omega) * d1) / den1 * z1
            + dt * p.varphi1 / den1 * v_avg
        )
        z2_new = (
            (1.0 - (1.0 - omega) * d2) / den2 * z2
            + dt * p.varphi2 / den2 * v_avg
        )
        u, v, z1, z2 = u_new, v_new, z1_new, z2_new

        step = n + 1
        if sample_col <= Ns and step == sample_steps[sample_col]:
            Y[:, sample_col] = u
            sample_col += 1

    if sample_col != Ns + 1:
        raise RuntimeError("Failed to collect the requested number of snapshots")
    return Y


# ---------------------------------------------------------------------
# ROM time stepping
# ---------------------------------------------------------------------

def run_rom_case(
    Nxy: int,
    k: int,
    Nt: int,
    W_name: str,
    p: ModelParameters,
    omega: float = 0.5,
    snapshot_Nt: int = 100,
    snapshot_omega: float | None = None,
    precomputed: dict | None = None,
):
    """Run the DG--POD ROM with inexact coarse-FOM snapshots.

    Timing convention
    -----------------
    setup_time_sec:
        DG space, full matrices, POD basis, reduced matrices, initial reduced
        coordinates, and reduced-system factorization.
    solve_time_sec:
        Cumulative reduced linear solves plus reconstruction ``Phi @ au``.
    snapshot_time_sec:
        Coarse-FOM snapshot generation, measured separately.
    total_time_sec:
        setup + solve + snapshot. Error evaluation is not timed.
    """
    T = p.T
    dt = T / Nt
    if snapshot_omega is None:
        snapshot_omega = omega

    # Spatial setup may be shared across W_u choices. Its measured cost is still
    # included in each ROM setup time according to the stated timing definition.
    if precomputed is None:
        spatial_setup_start = time.perf_counter()
        domain, V = create_space(Nxy, k)
        M, G, A, J = assemble_matrices(domain, V, p)
        spatial_setup_time = time.perf_counter() - spatial_setup_start

        snapshot_start = time.perf_counter()
        Ysnap = generate_fom_snapshots(
            V=V, M=M, A=A, J=J, p=p,
            Nt_snapshot=snapshot_Nt, Ns=p.Ns, omega=snapshot_omega,
        )
        snapshot_time = time.perf_counter() - snapshot_start
    else:
        domain = precomputed["domain"]
        V = precomputed["V"]
        M = precomputed["M"]
        G = precomputed["G"]
        A = precomputed["A"]
        J = precomputed["J"]
        Ysnap = precomputed["Ysnap"]
        spatial_setup_time = float(precomputed["spatial_setup_time_sec"])
        snapshot_time = float(precomputed["snapshot_time_sec"])

    rom_setup_start = time.perf_counter()
    Phi, svals = build_basis(Ysnap, W_name, M, A, J, p.ell)

    MP = M @ Phi
    AP = A @ Phi
    JP = J @ Phi
    Mr = Phi.T @ MP
    Ar = Phi.T @ AP
    Jr = Phi.T @ JP

    u0 = interpolate_exact(V, exact_u_values, 0.0)
    v0 = interpolate_exact(V, exact_v_values, 0.0)
    z10 = interpolate_exact(V, exact_zeta_values, 0.0, p.varphi1, p.tau1)
    z20 = interpolate_exact(V, exact_zeta_values, 0.0, p.varphi2, p.tau2)

    au = initial_coeff(Phi, u0)
    av = initial_coeff(Phi, v0)
    az1 = initial_coeff(Phi, z10)
    az2 = initial_coeff(Phi, z20)

    d1 = dt / p.tau1
    d2 = dt / p.tau2
    den1 = 1.0 + omega * d1
    den2 = 1.0 + omega * d2
    gamma1 = omega * p.varphi1 / den1
    gamma2 = omega * p.varphi2 / den2

    Br = (
        p.rho / (omega * dt * dt) * Mr
        + omega * p.varphi0 * Ar
        + (gamma1 + gamma2) * Ar
        + (1.0 / dt) * Jr
    )
    B_rom_lu = splu(Br)
    rom_specific_setup_time = time.perf_counter() - rom_setup_start
    setup_time = spatial_setup_time + rom_specific_setup_time

    max_L2 = 0.0
    max_H1 = 0.0
    rms_L2_sq = 0.0
    rms_H1_sq = 0.0
    nerr = 0
    V_exact = create_exact_space(domain, degree=max(k + 3, 5))

    solve_time = 0.0
    for n in range(Nt):
        tn = n * dt
        tnp1 = (n + 1) * dt

        F_n = exact_force_vector(V, M, A, J, p, tn)
        F_np1 = exact_force_vector(V, M, A, J, p, tnp1)
        Fr = Phi.T @ (omega * F_np1 + (1.0 - omega) * F_n)

        rhs = Fr.copy()
        rhs += p.rho / (omega * dt * dt) * (Mr @ au)
        rhs += p.rho / (omega * dt) * (Mr @ av)
        rhs += -(1.0 - omega) * p.varphi0 * (Ar @ au)
        rhs += (1.0 / dt) * (Jr @ au)
        rhs += gamma1 * (Ar @ au) - (1.0 / den1) * (Ar @ az1)
        rhs += gamma2 * (Ar @ au) - (1.0 / den2) * (Ar @ az2)

        solve_start = time.perf_counter()
        au_new = B_rom_lu.solve(rhs)
        solve_time += time.perf_counter() - solve_start

        av_new = (au_new - au) / (omega * dt) - (1.0 - omega) / omega * av
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

        reconstruction_start = time.perf_counter()
        u_rom = Phi @ au
        solve_time += time.perf_counter() - reconstruction_start

        # Error evaluation is deliberately not timed.
        u_ex_fun = fem.Function(V_exact)
        u_ex_fun.interpolate(lambda x: exact_u_values(x, tnp1))
        uh_fun = fem.Function(V)
        uh_fun.x.array[:] = u_rom
        e = u_ex_fun - uh_fun
        L2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
        H1_sq = fem.assemble_scalar(
            fem.form((ufl.inner(e, e) + ufl.inner(ufl.grad(e), ufl.grad(e))) * ufl.dx)
        )
        L2 = np.sqrt(max(L2_sq, 0.0))
        H1 = np.sqrt(max(H1_sq, 0.0))
        max_L2 = max(max_L2, L2)
        max_H1 = max(max_H1, H1)
        rms_L2_sq += L2_sq
        rms_H1_sq += H1_sq
        nerr += 1

    total_time = setup_time + solve_time + snapshot_time
    return {
        "method": "DG-POD", "W": W_name, "Nxy": Nxy, "k": k,
        "h": 1.0 / Nxy, "Nt": Nt, "dt": dt, "omega": omega,
        "Ns": p.Ns, "ell": p.ell, "ndofs": Ysnap.shape[0],
        "max_L2_error": max_L2, "max_H1_error": max_H1,
        "rms_L2_error": np.sqrt(rms_L2_sq / max(nerr, 1)),
        "rms_H1_error": np.sqrt(rms_H1_sq / max(nerr, 1)),
        "setup_time_sec": setup_time,
        "spatial_setup_time_sec": spatial_setup_time,
        "rom_specific_setup_time_sec": rom_specific_setup_time,
        "solve_time_sec": solve_time,
        "snapshot_time_sec": snapshot_time,
        "total_time_sec": total_time,
        "snapshot_Nt": snapshot_Nt, "snapshot_dt": T / snapshot_Nt,
        "snapshot_omega": snapshot_omega,
        "pod_tail": float(np.sqrt(np.sum(svals[p.ell:] ** 2))) if p.ell < len(svals) else 0.0,
    }



# ---------------------------------------------------------------------
# FOM time stepping
# ---------------------------------------------------------------------

def run_fom_case(
    Nxy: int,
    k: int,
    Nt: int,
    p: ModelParameters,
    omega: float = 0.5,
):
    """Run the full-order DG method with separated setup/solve timing."""
    T = p.T
    dt = T / Nt

    setup_start = time.perf_counter()
    domain, V = create_space(Nxy, k)
    M, G, A, J = assemble_matrices(domain, V, p)

    u = interpolate_exact(V, exact_u_values, 0.0)
    v = interpolate_exact(V, exact_v_values, 0.0)
    z1 = interpolate_exact(V, exact_zeta_values, 0.0, p.varphi1, p.tau1)
    z2 = interpolate_exact(V, exact_zeta_values, 0.0, p.varphi2, p.tau2)

    d1 = dt / p.tau1
    d2 = dt / p.tau2
    den1 = 1.0 + omega * d1
    den2 = 1.0 + omega * d2
    gamma1 = omega * p.varphi1 / den1
    gamma2 = omega * p.varphi2 / den2
    B = (
        p.rho / (omega * dt * dt) * M
        + omega * p.varphi0 * A
        + (gamma1 + gamma2) * A
        + (1.0 / dt) * J
    ).tocsc()
    B_lu = splu(B)
    setup_time = time.perf_counter() - setup_start

    max_L2 = 0.0
    max_H1 = 0.0
    rms_L2_sq = 0.0
    rms_H1_sq = 0.0
    nerr = 0
    V_exact = create_exact_space(domain, degree=max(k + 3, 5))

    solve_time = 0.0
    for n in range(Nt):
        tn = n * dt
        tnp1 = (n + 1) * dt
        F_n = exact_force_vector(V, M, A, J, p, tn)
        F_np1 = exact_force_vector(V, M, A, J, p, tnp1)
        Favg = omega * F_np1 + (1.0 - omega) * F_n

        rhs = Favg.copy()
        rhs += p.rho / (omega * dt * dt) * (M @ u)
        rhs += p.rho / (omega * dt) * (M @ v)
        rhs += -(1.0 - omega) * p.varphi0 * (A @ u)
        rhs += (1.0 / dt) * (J @ u)
        rhs += gamma1 * (A @ u) - (1.0 / den1) * (A @ z1)
        rhs += gamma2 * (A @ u) - (1.0 / den2) * (A @ z2)

        solve_start = time.perf_counter()
        u_new = B_lu.solve(rhs)
        solve_time += time.perf_counter() - solve_start

        v_new = (u_new - u) / (omega * dt) - (1.0 - omega) / omega * v
        v_avg = (u_new - u) / dt
        z1_new = (
            (1.0 - (1.0 - omega) * d1) / den1 * z1
            + dt * p.varphi1 / den1 * v_avg
        )
        z2_new = (
            (1.0 - (1.0 - omega) * d2) / den2 * z2
            + dt * p.varphi2 / den2 * v_avg
        )
        u, v, z1, z2 = u_new, v_new, z1_new, z2_new

        # Error evaluation is deliberately not timed.
        u_ex_fun = fem.Function(V_exact)
        u_ex_fun.interpolate(lambda x: exact_u_values(x, tnp1))
        uh_fun = fem.Function(V)
        uh_fun.x.array[:] = u
        e = u_ex_fun - uh_fun
        L2_sq = fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))
        H1_sq = fem.assemble_scalar(
            fem.form((ufl.inner(e, e) + ufl.inner(ufl.grad(e), ufl.grad(e))) * ufl.dx)
        )
        L2 = np.sqrt(max(L2_sq, 0.0))
        H1 = np.sqrt(max(H1_sq, 0.0))
        max_L2 = max(max_L2, L2)
        max_H1 = max(max_H1, H1)
        rms_L2_sq += L2_sq
        rms_H1_sq += H1_sq
        nerr += 1

    snapshot_time = 0.0
    total_time = setup_time + solve_time + snapshot_time
    return {
        "method": "DG-FOM", "W": "-", "Nxy": Nxy, "k": k,
        "h": 1.0 / Nxy, "Nt": Nt, "dt": dt, "omega": omega,
        "Ns": np.nan, "ell": np.nan, "ndofs": len(u),
        "max_L2_error": max_L2, "max_H1_error": max_H1,
        "rms_L2_error": np.sqrt(rms_L2_sq / max(nerr, 1)),
        "rms_H1_error": np.sqrt(rms_H1_sq / max(nerr, 1)),
        "setup_time_sec": setup_time,
        "spatial_setup_time_sec": setup_time,
        "rom_specific_setup_time_sec": np.nan,
        "solve_time_sec": solve_time,
        "snapshot_time_sec": snapshot_time,
        "total_time_sec": total_time,
        "snapshot_Nt": np.nan, "snapshot_dt": np.nan,
        "snapshot_omega": np.nan, "pod_tail": np.nan,
    }

    
def dg_spatial_convergence():
    """Requested spatial convergence experiment for FOM and inexact-snapshot ROM."""
    p = ModelParameters()
    p.T = 10.0
    p.Ns = 1000

    outdir = Path("results_spatial")
    outdir.mkdir(parents=True, exist_ok=True)

    snapshot_Nt = 1000       # coarse FOM steps for inexact snapshots
    Nt = 20000              # online FOM/ROM steps used for spatial-error tests
    Nxy_list = [4, 8, 16, 32]
    k_list = [1, 2]

    # POD inner products for the displacement snapshots.
    W_u_list = ["I", "M", "A"]

    schemes = {
        # "Euler": 1.0,
        "Crank-Nicolson": 0.5,
    }

    rows = []
    for scheme_name, omega in schemes.items():
        for k in k_list:
            for Nxy in Nxy_list:
                print(
                    f"FOM spatial: scheme={scheme_name}, k={k}, Nxy={Nxy}, "
                    f"Nt={Nt}, T={p.T}",
                    flush=True,
                )
                fom_row = run_fom_case(
                    Nxy=Nxy,
                    k=k,
                    Nt=Nt,
                    p=p,
                    omega=omega,
                )
                fom_row["scheme"] = scheme_name
                rows.append(fom_row)

                # Assemble the ROM spatial operators and generate the coarse-FOM
                # snapshot ensemble only once for this (scheme, k, Nxy) case.
                spatial_setup_start = time.perf_counter()
                domain, V = create_space(Nxy, k)
                M, G, A, J = assemble_matrices(domain, V, p)
                spatial_setup_time = time.perf_counter() - spatial_setup_start
                snapshot_start = time.perf_counter()
                Ysnap = generate_fom_snapshots(
                    V=V, M=M, A=A, J=J, p=p,
                    Nt_snapshot=snapshot_Nt, Ns=p.Ns, omega=omega,
                )
                snapshot_time = time.perf_counter() - snapshot_start
                print(
                    f"Snapshot generated once: scheme={scheme_name}, k={k}, "
                    f"Nxy={Nxy}, Nt={snapshot_Nt}, time={snapshot_time:.6e} s",
                    flush=True,
                )
                precomputed = {
                    "domain": domain, "V": V, "M": M, "G": G,
                    "A": A, "J": J, "Ysnap": Ysnap,
                    "snapshot_time_sec": snapshot_time,
                    "spatial_setup_time_sec": spatial_setup_time,
                }

                for W_u in W_u_list:
                    print(
                        f"ROM spatial: scheme={scheme_name}, W_u={W_u}, k={k}, "
                        f"Nxy={Nxy}, snapshot_Nt={snapshot_Nt}, Nt={Nt}, "
                        f"Ns={p.Ns}, ell={p.ell}, T={p.T}",
                        flush=True,
                    )
                    rom_row = run_rom_case(
                        Nxy=Nxy,
                        k=k,
                        Nt=Nt,
                        W_name=W_u,
                        p=p,
                        omega=omega,
                        snapshot_Nt=snapshot_Nt,
                        snapshot_omega=omega,
                        precomputed=precomputed,
                    )
                    rom_row["scheme"] = scheme_name
                    rows.append(rom_row)

    df = pd.DataFrame(rows)
    df["order_L2"] = np.nan
    df["order_H1"] = np.nan

    for (scheme, method, W, k), sub in df.groupby(
        ["scheme", "method", "W", "k"], dropna=False
    ):
        sub = sub.sort_values("h", ascending=False)
        h = sub["h"].to_numpy()
        eL2 = sub["max_L2_error"].to_numpy()
        eH1 = sub["max_H1_error"].to_numpy()

        order_L2 = [np.nan]
        order_H1 = [np.nan]
        for i in range(1, len(sub)):
            order_L2.append(
                np.log(eL2[i - 1] / eL2[i]) / np.log(h[i - 1] / h[i])
            )
            order_H1.append(
                np.log(eH1[i - 1] / eH1[i]) / np.log(h[i - 1] / h[i])
            )
        df.loc[sub.index, "order_L2"] = order_L2
        df.loc[sub.index, "order_H1"] = order_H1

    csv_path = outdir / "dg_spatial_convergence.csv"
    df.to_csv(csv_path, index=False)

    timing_cols = [
        "scheme", "method", "W", "k", "Nxy", "ndofs",
        "setup_time_sec", "solve_time_sec",
        "snapshot_time_sec", "total_time_sec",
    ]
    print("\nTiming summary (error evaluation is not timed):")
    print(df[timing_cols].to_string(index=False))
    print(f"\nSaved: {csv_path}")
    return df


def dg_temporal_convergence():
    p = ModelParameters()
    p.T = 10.0
    p.Ns = 100
    p.ell = 20

    outdir = Path("results_temporal")
    outdir.mkdir(parents=True, exist_ok=True)

    # Fixed spatial discretization
    Nxy = 128
    k = 2

    # Snapshot discretization
    snapshot_Nt = 100

    # Online temporal refinements
    Nt_list = [200, 400, 800, 1600]    
    

    W_u_list = ["I", "M", "A"]

    schemes = {
        "Euler": 1.0,
        "Crank-Nicolson": 0.5,
    }

    rows = []

    for scheme_name, omega in schemes.items():

        # ------------------------------------------------------------
        # Construct the spatial operators and snapshot ensemble once
        # for each temporal scheme.
        # ------------------------------------------------------------
        spatial_setup_start = time.perf_counter()

        domain, V = create_space(Nxy, k)
        M, G, A, J = assemble_matrices(domain, V, p)

        spatial_setup_time = (
            time.perf_counter() - spatial_setup_start
        )

        snapshot_start = time.perf_counter()

        Ysnap = generate_fom_snapshots(
            V=V,
            M=M,
            A=A,
            J=J,
            p=p,
            Nt_snapshot=snapshot_Nt,
            Ns=p.Ns,
            omega=omega,
        )

        snapshot_time = time.perf_counter() - snapshot_start

        print(
            f"Snapshot generated once: scheme={scheme_name}, "
            f"k={k}, Nxy={Nxy}, snapshot_Nt={snapshot_Nt}, "
            f"Ns={p.Ns}, time={snapshot_time:.6e} s",
            flush=True,
        )

        precomputed = {
            "domain": domain,
            "V": V,
            "M": M,
            "G": G,
            "A": A,
            "J": J,
            "Ysnap": Ysnap,
            "snapshot_time_sec": snapshot_time,
            "spatial_setup_time_sec": spatial_setup_time,
        }

        for Nt in Nt_list:
            dt = p.T / Nt

            # --------------------------------------------------------
            # Full-order model
            # --------------------------------------------------------
            print(
                f"FOM temporal: scheme={scheme_name}, "
                f"k={k}, Nxy={Nxy}, Nt={Nt}, dt={dt:.6e}",
                flush=True,
            )

            fom_row = run_fom_case(
                Nxy=Nxy,
                k=k,
                Nt=Nt,
                p=p,
                omega=omega,
            )

            fom_row["scheme"] = scheme_name
            rows.append(fom_row)

            # --------------------------------------------------------
            # Reduced-order models
            # --------------------------------------------------------
            for W_u in W_u_list:
                print(
                    f"ROM temporal: scheme={scheme_name}, "
                    f"W_u={W_u}, k={k}, Nxy={Nxy}, "
                    f"snapshot_Nt={snapshot_Nt}, Nt={Nt}, "
                    f"Ns={p.Ns}, ell={p.ell}, dt={dt:.6e}",
                    flush=True,
                )

                rom_row = run_rom_case(
                    Nxy=Nxy,
                    k=k,
                    Nt=Nt,
                    W_name=W_u,
                    p=p,
                    omega=omega,
                    snapshot_Nt=snapshot_Nt,
                    snapshot_omega=omega,
                    precomputed=precomputed,
                )

                rom_row["scheme"] = scheme_name
                rows.append(rom_row)

    # ================================================================
    # Data frame and observed temporal orders
    # ================================================================
    df = pd.DataFrame(rows)

    df["order_L2"] = np.nan
    df["order_H1"] = np.nan

    for (scheme, method, W), sub in df.groupby(
        ["scheme", "method", "W"],
        dropna=False,
    ):
        # Coarse-to-fine ordering
        sub = sub.sort_values("dt", ascending=False)

        dt = sub["dt"].to_numpy()
        eL2 = sub["max_L2_error"].to_numpy()
        eH1 = sub["max_H1_error"].to_numpy()

        order_L2 = [np.nan]
        order_H1 = [np.nan]

        for i in range(1, len(sub)):
            order_L2.append(
                np.log(eL2[i - 1] / eL2[i])
                / np.log(dt[i - 1] / dt[i])
            )

            order_H1.append(
                np.log(eH1[i - 1] / eH1[i])
                / np.log(dt[i - 1] / dt[i])
            )

        df.loc[sub.index, "order_L2"] = order_L2
        df.loc[sub.index, "order_H1"] = order_H1

    # ================================================================
    # Save CSV
    # ================================================================
    csv_path = outdir / "dg_temporal_convergence.csv"
    df.to_csv(csv_path, index=False)

    # ================================================================
    # Print summary
    # ================================================================
    timing_cols = [
        "scheme",
        "method",
        "W",
        "Nt",
        "dt",
        "max_L2_error",
        "order_L2",
        "max_H1_error",
        "order_H1",
        "setup_time_sec",
        "solve_time_sec",
        "snapshot_time_sec",
        "total_time_sec",
    ]

    print(
        "\nTemporal convergence and timing summary "
        "(error evaluation is not timed):"
    )
    print(df[timing_cols].to_string(index=False))
    print(f"\nSaved: {csv_path}")

    return df





if __name__ == "__main__":
    dg_spatial_convergence()
    dg_temporal_convergence()

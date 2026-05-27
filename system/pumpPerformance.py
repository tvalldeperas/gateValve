#!/usr/bin/env python3
import os, glob, math, sys, re
import numpy as np
import pandas as pd

# non-interactive backend for batch runs
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# Avoid EPS transparency warning (PostScript has no alpha)
mpl.rcParams["legend.framealpha"] = 1.0

# ---------------- user inputs ----------------
rpm    = 1450.0                                                         # impeller rpm
rho    = 1000.0                                                         # kg/m^3
g      = 9.81                                                           # m/s^2
axis   = "z"                                                            # rotation axis: "x" | "y" | "z"
csvOut = os.path.join("postProcessing", "pumpPerformance.csv")          # output path inside postProcessing

degPerTimeStep = 1.0                                                    # rotor angular advance per timestep [deg]

nFFTrev    = 10                                                         # revolutions used for SST/orders window
monitorFFT = 4                                                          # monitor index used for SST/orders (1-based)
max_order  = 10                                                         # compute orders 1..max_order

nPlotRev   = 1                                                          # revolutions shown in *_last<nPlotRev>rev plots

# monitor head reference:
#   integer -> monitor number (1-based), e.g. 4
#   "i"     -> inlet p0 reference
#   "o"     -> outlet p0 reference
monitorRef = "o"

# Cp reference diameter for blade tip speed U2 = omega * Dimpeller / 2
Dimpeller = 0.2                                                         # [m] diameter used for blade tip speed in Cp definition

# Reference diameter for non-dimensional coefficients
Doutput   = 0.1                                                         # [m] reference diameter for phi, psi, lambda

# Optional area-based flow coefficient:
#   if True, compute phi_A = Q / (A2 * U2_ref), with A2 = pi * Doutput * b2
#   if False, phi_A is written as NaN
use_area_based_phi = False
b2 = None                                                               # [m] impeller outlet width (only used if use_area_based_phi=True)
# ---------------------------------------------


omega = 2.0 * math.pi * rpm / 60.0   # rad/s
Trev  = 60.0 / rpm                   # s (1 revolution, 360 deg)

if degPerTimeStep <= 0.0:
    raise ValueError(f"degPerTimeStep must be > 0, got {degPerTimeStep}")

dt_expected = Trev * degPerTimeStep / 360.0   # expected physical timestep [s]
tol_time = 0.49 * dt_expected                 # tolerance for time alignment (merge_asof)

# Cp reference quantities based on Dimpeller
r2 = 0.5 * Dimpeller                # m
U2 = omega * r2                     # m/s
q2 = 0.5 * rho * U2 * U2            # Pa  -> dynamic pressure based on blade tip speed

# Non-dimensional reference quantities based on Doutput
U_ref = omega * Doutput / 2.0       # m/s -> reference tip speed based on Doutput


def latest_file(pattern):
    """Return (latest_file_path, latest_time) based on time directory name."""
    files = glob.glob(pattern)
    if not files:
        return None, None

    def time_of(f):
        try:
            return float(os.path.basename(os.path.dirname(f)))
        except Exception:
            return -1.0

    files.sort(key=time_of)
    return files[-1], time_of(files[-1])


def load_surface_dat(path, two_cols=True):
    """Generic reader for surfaceFieldValue.dat style files."""
    if not path or not os.path.isfile(path):
        return np.empty((0, 2))

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#%/":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            if two_cols:
                rows.append([float(parts[0]), float(parts[1])])
            else:
                rows.append([float(x) for x in parts])

    return np.array(rows) if rows else np.empty((0, 2))


# --- robust parser for forces/moment.dat (parenthesized vectors) ---
_paren_re = re.compile(
    r'^\s*'
    r'([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)'
    r'\s*\(\s*([^)]+?)\s*\)\s*'
    r'\(\s*([^)]+?)\s*\)\s*'
    r'\(\s*([^)]+?)\s*\)\s*$'
)


def _try_parse_paren_line(line):
    m = _paren_re.match(line)
    if not m:
        return None
    t = float(m.group(1))

    def vec3(s):
        parts = s.replace(',', ' ').split()
        if len(parts) != 3:
            raise ValueError("Vector does not have 3 components")
        return [float(parts[0]), float(parts[1]), float(parts[2])]

    tot = vec3(m.group(2))
    pres = vec3(m.group(3))
    visc = vec3(m.group(4))
    return (t, tot, pres, visc)


def load_moment_dat(path):
    """
    Load postProcessing/impellerForces/*/moment.dat into an array.

    Returns ndarray with columns:
      time, total_x, total_y, total_z, pressure_x, pressure_y, pressure_z, viscous_x, viscous_y, viscous_z
    If only totals are available: time, total_x, total_y, total_z
    """
    if not path or not os.path.isfile(path):
        return np.empty((0, 4))

    rows = []
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] in "#%/":
                continue

            parsed = _try_parse_paren_line(line)
            if parsed is not None:
                t, tot, pres, visc = parsed
                rows.append([t] + tot + pres + visc)
                continue

            parts = (
                line.replace('(', ' ')
                    .replace(')', ' ')
                    .replace(',', ' ')
                    .split()
            )
            if len(parts) < 4:
                continue
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(nums)

    if not rows:
        return np.empty((0, 4))

    arr = np.array(rows, dtype=float)
    ncol = arr.shape[1]

    if ncol >= 10:
        tot = arr[:, 1:4]
        pres = arr[:, 4:7]
        visc = arr[:, 7:10]
        if np.allclose(tot, pres + visc, rtol=1e-6, atol=1e-6):
            return np.column_stack([arr[:, 0], tot, pres, visc])
        return np.column_stack([arr[:, 0], arr[:, 1:4]])
    elif ncol == 4:
        return arr
    else:
        return np.column_stack([arr[:, 0], arr[:, -3:]])


def pick_total_component(mom_arr, axis):
    """Return array with [time, TOTAL moment component along given axis]."""
    if mom_arr.size == 0:
        return np.empty((0, 2))

    ax = {"x": 0, "y": 1, "z": 2}[axis]
    ncol = mom_arr.shape[1]

    if ncol == 4:
        tot = mom_arr[:, 1:4]
    elif ncol >= 10:
        tot = mom_arr[:, 1:4]
    else:
        tot = mom_arr[:, -3:]

    return np.column_stack([mom_arr[:, 0], tot[:, ax]])


def find_existing_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


def load_probes_scalar_all_times(monitor_root, field_name):
    """
    Load scalar probe outputs across ALL time folders:
      postProcessing/volutMonitors/<time>/<field_name>

    Returns:
      times (N,)
      values (N, nProbes)
    """
    if not monitor_root or not os.path.isdir(monitor_root):
        return np.empty((0,)), np.empty((0, 0))

    pattern = os.path.join(monitor_root, "*", field_name)
    files = glob.glob(pattern)
    if not files:
        return np.empty((0,)), np.empty((0, 0))

    all_rows = []
    for fp in files:
        if not os.path.isfile(fp):
            continue
        with open(fp, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s[0] in "#%/":
                    continue
                parts = s.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
                if len(parts) < 2:
                    continue
                try:
                    t = float(parts[0])
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                all_rows.append([t] + vals)

    if not all_rows:
        return np.empty((0,)), np.empty((0, 0))

    arr = np.array(all_rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]

    # Drop exact-duplicate times keeping the last occurrence
    tcol = arr[:, 0]
    keep = np.ones(len(tcol), dtype=bool)
    for i in range(len(tcol) - 1):
        if tcol[i] == tcol[i + 1]:
            keep[i] = False
    arr = arr[keep]

    return arr[:, 0], arr[:, 1:]


def df_from_2col(arr, colname):
    return pd.DataFrame({"time [s]": arr[:, 0], colname: arr[:, 1]}).sort_values("time [s]")


def merge_on_time(df_left, df_right, tol):
    """
    Nearest-time merge within tolerance. Prevents losing points due to float formatting differences.
    """
    return pd.merge_asof(
        df_left.sort_values("time [s]"),
        df_right.sort_values("time [s]"),
        on="time [s]",
        direction="nearest",
        tolerance=tol
    )


def add_spins_top_axis(ax, rpm):
    Trev_local = 60.0 / rpm
    x0, x1 = ax.get_xlim()
    ax_top = ax.twiny()
    ax_top.set_xlim(x0 / Trev_local, x1 / Trev_local)
    ax_top.set_xlabel("Revolutions [-]")
    return ax_top


def mask_lastNrevs_by_time(t, rpm, n_rev):
    Trev_local = 60.0 / rpm
    t_end = t[-1]
    return t >= (t_end - n_rev * Trev_local)


def order_amplitudes_by_sinefit(t, y, rpm, max_order=10):
    """
    Robust order amplitude estimation by least-squares sine/cos fit at each order.
    Returns peak amplitude (same units as y).
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if len(t) < 8:
        return np.arange(1, max_order + 1), np.full(max_order, np.nan)

    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if len(t) < 8:
        return np.arange(1, max_order + 1), np.full(max_order, np.nan)

    # remove mean
    y0 = y - np.mean(y)

    f_rot = rpm / 60.0
    orders = np.arange(1, max_order + 1, dtype=float)
    amps = np.zeros(len(orders), dtype=float)

    for i, o in enumerate(orders):
        f = o * f_rot
        w = 2.0 * np.pi * f
        s = np.sin(w * t)
        c = np.cos(w * t)

        # Solve y0 ≈ a*s + b*c (least squares)
        A = np.column_stack([s, c])
        coeff, _, _, _ = np.linalg.lstsq(A, y0, rcond=None)
        a, b = coeff
        amps[i] = float(np.sqrt(a * a + b * b))  # peak amplitude

    return orders.astype(int), amps


def plot_sst_orders_only(
    df, mask_win, rpm, n_rev, mon_col,
    png_orders, eps_orders, max_order=10
):
    """
    Make only:
      (1) "SST/orders" bar chart (1..max_order) for selected monitor,
          using last n_rev revolutions in mask_win.
    """
    t = df["time [s]"].values
    y = df[mon_col].values

    t_win = t[mask_win]
    y_win = y[mask_win]

    if t_win.size < 8:
        print(f"[info] Not enough points in last {n_rev} rev(s) to analyze {mon_col}.")
        return

    orders, amps = order_amplitudes_by_sinefit(t_win, y_win, rpm, max_order=max_order)

    print(f"\n[SST/orders] {mon_col} [m] (last {n_rev} rev):")
    for o, a in zip(orders, amps):
        print(f"  {int(o):2d}x : {a:.6g}")

    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.bar(orders.astype(int), amps)
    ax.set_xlabel("Order (× shaft frequency)")
    ax.set_ylabel(f"Amplitude of {mon_col}")
    ax.set_title(f"SST / Order amplitudes of {mon_col} (last {n_rev} rev)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(png_orders, dpi=300)
    fig.savefig(eps_orders, format="eps")
    plt.close(fig)
    print(f"Written plot: {png_orders}")
    print(f"Written plot: {eps_orders}")


def save_fig_png_eps(fig, png_path, eps_path):
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)
    print(f"Written plot: {png_path}")
    print(f"Written plot: {eps_path}")


def plot_pump(df, mask, rpm, axis_cap, T_col, png_path, eps_path):
    t = df["time [s]"].values
    H = df["H [m]"].values
    eta = df["eta [-]"].values
    Tax = df[T_col].values
    Cp = df["Cp [-]"].values

    fig = plt.figure(figsize=(24, 4))

    ax1 = plt.subplot(1, 4, 1)
    ax1.plot(t[mask], H[mask], linewidth=0.8, color="black")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("H [m]")
    ax1.set_title("Head Convergence")
    ax1.grid(True)
#    ax1.set_ylim(9, 12)         ################# PUMP
#    ax1.set_ylim(-30, 0)        ################# PAT
    add_spins_top_axis(ax1, rpm)

    ax2 = plt.subplot(1, 4, 2)
    ax2.plot(t[mask], eta[mask], linewidth=0.8, color="green")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("η [-]")
    ax2.set_title("Efficiency Convergence")
    ax2.grid(True)
#    ax2.set_ylim(0.6, 1)        ################# PUMP
#    ax2.set_ylim(0.3, 1)        ################# PAT
    add_spins_top_axis(ax2, rpm)

    ax3 = plt.subplot(1, 4, 3)
    ax3.plot(t[mask], Tax[mask], linewidth=0.8, color="blue")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel(T_col)
    ax3.set_title(f"Impeller {axis_cap}-Moment Convergence")
    ax3.grid(True)
#    ax3.set_ylim(11.5, 12.5)    ################# PUMP
#    ax3.set_ylim(-60, -20)      ################# PAT
    add_spins_top_axis(ax3, rpm)

    ax4 = plt.subplot(1, 4, 4)
    ax4.plot(t[mask], Cp[mask], linewidth=0.8, color="red")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Cp [-]")
    ax4.set_title("Pressure Coefficient Convergence")
    ax4.grid(True)
    add_spins_top_axis(ax4, rpm)

    save_fig_png_eps(fig, png_path, eps_path)


def plot_cp(df, mask, rpm, png_path, eps_path):
    t = df["time [s]"].values
    Cp = df["Cp [-]"].values

    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.plot(t[mask], Cp[mask], linewidth=0.8, color="red")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cp [-]")
    ax.set_title("Pressure Coefficient")
    ax.grid(True)
    add_spins_top_axis(ax, rpm)
#    ax.set_ylim(9, 10.5)        ################# PUMP
#    ax.set_ylim(-3.3, -2.2)     ################# PAT

    save_fig_png_eps(fig, png_path, eps_path)


def plot_non_dim_coeff(df, mask, rpm, colname, title, ylabel, png_path, eps_path):
    t = df["time [s]"].values
    y = df[colname].values

    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.plot(t[mask], y[mask], linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    add_spins_top_axis(ax, rpm)

    save_fig_png_eps(fig, png_path, eps_path)


def plot_volute_monitors(df, mon_H_cols, mask, rpm, png_path, eps_path):
    t = df["time [s]"].values
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    for i, col in enumerate(mon_H_cols):
        ax.plot(t[mask], df[col].values[mask], linewidth=0.8, label=f"Monitor {i+1}")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Head [m]")
    ax.set_title("Volute Monitor Head from p")
    ax.grid(True)
#    ax.set_ylim(9, 10.5)        ################# PUMP
#    ax.set_ylim(-25, 50)        ################# PAT

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
        framealpha=1.0
    )

    add_spins_top_axis(ax, rpm)
    save_fig_png_eps(fig, png_path, eps_path)


def parse_monitor_reference(ref_value, nMon_use):
    """
    monitorRef can be:
      - integer/float/string number: monitor index (1-based)
      - 'i': inlet p0 reference
      - 'o': outlet p0 reference

    Returns:
      ('monitor', idx0) or ('inlet', None) or ('outlet', None)
    """
    if isinstance(ref_value, str):
        s = ref_value.strip().lower()
        if s == "i":
            return ("inlet", None)
        if s == "o":
            return ("outlet", None)
        try:
            idx = int(s) - 1
        except ValueError:
            raise ValueError(
                f"Invalid monitorRef={ref_value!r}. Use an integer monitor number, 'i', or 'o'."
            )
    else:
        try:
            idx = int(ref_value) - 1
        except Exception:
            raise ValueError(
                f"Invalid monitorRef={ref_value!r}. Use an integer monitor number, 'i', or 'o'."
            )

    if idx < 0 or idx >= nMon_use:
        raise ValueError(
            f"monitorRef={ref_value!r} selects monitor {idx+1}, but available monitors are 1..{nMon_use}."
        )

    return ("monitor", idx)


# ---- locate latest data files
Q_in_path, _   = latest_file("postProcessing/Q_inlet/*/surfaceFieldValue.dat")
p0_in_path, _  = latest_file("postProcessing/p0_mdotAvg_inlet/*/surfaceFieldValue.dat")
p0_out_path, _ = latest_file("postProcessing/p0_mdotAvg_outlet/*/surfaceFieldValue.dat")
mom_path, _    = latest_file("postProcessing/impellerForces/*/moment.dat")

Q_in   = load_surface_dat(Q_in_path)
p0_in  = load_surface_dat(p0_in_path)
p0_out = load_surface_dat(p0_out_path)
momRaw = load_moment_dat(mom_path)
mom    = pick_total_component(momRaw, axis)

if Q_in.size == 0 or p0_in.size == 0 or p0_out.size == 0 or mom.size == 0:
    print("Exit due to the lack of Q_in / p0_in / p0_out / moment")
    print("sizes:", Q_in.size, p0_in.size, p0_out.size, mom.size)
    sys.exit(0)

# ---- volute monitors
mon_root = find_existing_dir([
    os.path.join("postProcessing", "volutMonitors"),
    os.path.join("postprocessing", "volutMonitors"),
])

mon_t0, mon_p0 = load_probes_scalar_all_times(mon_root, "p0")
mon_tp, mon_p  = load_probes_scalar_all_times(mon_root, "p")   # optional, only if you still want to keep raw p available

have_p0 = (mon_t0.size > 0 and mon_p0.size > 0)
have_p  = (mon_tp.size > 0 and mon_p.size > 0)

# monitor head plots must use total pressure probes
have_monitors = have_p
nMon = mon_p.shape[1] if have_p else 0
nMon_use = nMon

if have_p0 and have_p:
    nMon_use = min(nMon_use, mon_p0.shape[1])

# ---- build merged dataframe using tolerant time alignment
dfQ    = df_from_2col(Q_in,   "Q [m3/s]")
dfPin  = df_from_2col(p0_in,  "p0_in [Pa]")
dfPout = df_from_2col(p0_out, "p0_out [Pa]")
dfT    = df_from_2col(mom,    f"T{axis.upper()} [N·m]")

dfm = dfQ
dfm = merge_on_time(dfm, dfPin,  tol_time)
dfm = merge_on_time(dfm, dfPout, tol_time)
dfm = merge_on_time(dfm, dfT,    tol_time)

mon_p0_cols = []
mon_H_cols = []
mon_p_cols = []

if have_p0 and nMon_use > 0:
    mon_p0_cols = [f"p0_mon{i+1} [Pa]" for i in range(nMon_use)]
    dfMon0 = pd.DataFrame(mon_p0[:, :nMon_use], columns=mon_p0_cols)
    dfMon0.insert(0, "time [s]", mon_t0)
    dfMon0 = dfMon0.sort_values("time [s]")
    dfm = merge_on_time(dfm, dfMon0, tol_time)

if have_p and nMon_use > 0:
    mon_p_cols = [f"p_mon{i+1} [m2/s2]" for i in range(nMon_use)]
    mon_H_cols = [f"H_mon{i+1} [m]" for i in range(nMon_use)]

    dfMonP = pd.DataFrame(mon_p[:, :nMon_use], columns=mon_p_cols)
    dfMonP.insert(0, "time [s]", mon_tp)
    dfMonP = dfMonP.sort_values("time [s]")
    dfm = merge_on_time(dfm, dfMonP, tol_time)

# keep only synchronized rows
dfm = dfm.dropna().reset_index(drop=True)

# ---- compute performance
dfm["rpm"] = rpm
dfm["degPerTimeStep [deg]"] = degPerTimeStep
dfm["U2 [m/s]"] = U2
dfm["U_ref [m/s]"] = U_ref
dfm["Δp0 [Pa]"] = dfm["p0_out [Pa]"] - dfm["p0_in [Pa]"]
dfm["H [m]"] = dfm["Δp0 [Pa]"] / (rho * g)
dfm["Ph [W]"] = rho * g * dfm["Q [m3/s]"] * dfm["H [m]"]

T_col = f"T{axis.upper()} [N·m]"
dfm["Pm [W]"] = dfm[T_col] * omega

# Pressure coefficient based on blade tip speed U2
if q2 > 0.0:
    dfm["Cp [-]"] = dfm["Δp0 [Pa]"] / q2
else:
    dfm["Cp [-]"] = np.nan

# True for pump mode (mechanical power larger than hydraulic), False for turbine mode
dfm["pump"] = dfm["Pm [W]"] > dfm["Ph [W]"]

# Efficiency: always (smaller magnitude) / (larger magnitude) -> always <= 1
Pm_abs = np.abs(dfm["Pm [W]"].to_numpy())
Ph_abs = np.abs(dfm["Ph [W]"].to_numpy())

den = np.maximum(Pm_abs, Ph_abs)
num = np.minimum(Pm_abs, Ph_abs)

eta = np.where(den > 0.0, num / den, np.nan)

# Optional: protect against tiny numerical overshoots
dfm["eta [-]"] = np.clip(eta, 0.0, np.nextafter(1.0, 0.0))

# ---- non-dimensional coefficients
# Flow coefficient (based on omega and Doutput)
if omega > 0.0 and Doutput > 0.0:
    dfm["phi [-]"] = dfm["Q [m3/s]"] / (omega * Doutput**3)
else:
    dfm["phi [-]"] = np.nan

# Head coefficient
if omega > 0.0 and Doutput > 0.0:
    dfm["psi [-]"] = g * dfm["H [m]"] / (omega**2 * Doutput**2)
else:
    dfm["psi [-]"] = np.nan

# Power coefficients (both hydraulic and mechanical)
if rho > 0.0 and omega > 0.0 and Doutput > 0.0:
    dfm["lambda_h [-]"] = dfm["Ph [W]"] / (rho * omega**3 * Doutput**5)
    dfm["lambda_m [-]"] = dfm["Pm [W]"] / (rho * omega**3 * Doutput**5)
else:
    dfm["lambda_h [-]"] = np.nan
    dfm["lambda_m [-]"] = np.nan

# Optional: area-based flow coefficient (disabled by default)
if use_area_based_phi and b2 is not None and Doutput > 0.0 and b2 > 0.0 and U_ref > 0.0:
    A2 = math.pi * Doutput * b2
    dfm["A2 [m2]"] = A2
    dfm["phi_A [-]"] = dfm["Q [m3/s]"] / (A2 * U_ref)
else:
    dfm["A2 [m2]"] = np.nan
    dfm["phi_A [-]"] = np.nan

# ---- monitor heads from total pressure p0
if have_p and nMon_use > 0:
    ref_kind, ref_idx = parse_monitor_reference(monitorRef, nMon_use)

    if ref_kind == "monitor":
        p_ref_series = dfm[mon_p_cols[ref_idx]]
        ref_label = f"monitor {ref_idx + 1}"
    elif ref_kind == "inlet":
        # p0_in is in Pa -> convert to kinematic pressure for consistency with p probes
        p_ref_series = dfm["p0_in [Pa]"] / rho
        ref_label = "inlet"
    elif ref_kind == "outlet":
        # p0_out is in Pa -> convert to kinematic pressure for consistency with p probes
        p_ref_series = dfm["p0_out [Pa]"] / rho
        ref_label = "outlet"
    else:
        raise RuntimeError(f"Unhandled monitor reference type: {ref_kind}")

    print(f"Monitor head reference: {ref_label}")

    for i in range(nMon_use):
        dfm[mon_H_cols[i]] = (dfm[mon_p_cols[i]] - p_ref_series) / g


# ---- write CSV
os.makedirs("postProcessing", exist_ok=True)

if os.path.isfile(csvOut):
    try:
        old = pd.read_csv(csvOut)
        if "time [s]" in old.columns:
            old = old.sort_values("time [s]")
            new = dfm.sort_values("time [s]")
            merged = pd.merge_asof(
                old, new, on="time [s]",
                direction="nearest", tolerance=tol_time,
                suffixes=("", "_new")
            )
            for c in new.columns:
                if c == "time [s]":
                    continue
                if c not in merged.columns and (c + "_new") in merged.columns:
                    merged[c] = merged[c + "_new"]
                elif (c + "_new") in merged.columns:
                    merged[c] = merged[c].combine_first(merged[c + "_new"]) if c in merged.columns else merged[c + "_new"]
                if (c + "_new") in merged.columns:
                    merged.drop(columns=[c + "_new"], inplace=True)
            df_out = merged
        else:
            df_out = dfm
    except Exception:
        df_out = dfm
else:
    df_out = dfm

df_out = df_out.sort_values("time [s]")
df_out.to_csv(csvOut, index=False, float_format="%.8g")
print(f"Written CSV: {csvOut}")

# ---- masks
t_arr = dfm["time [s]"].values
mask_all = np.ones_like(t_arr, dtype=bool)

nFFTrev = max(1, int(nFFTrev))
mask_lastFFT = mask_lastNrevs_by_time(t_arr, rpm, nFFTrev)

nPlotRev = max(1, int(nPlotRev))
mask_lastPlot = mask_lastNrevs_by_time(t_arr, rpm, nPlotRev)

# ---- SST/orders for selected monitor
if have_p and nMon_use > 0:
    mon_idx = int(monitorFFT) - 1  # 1-based -> 0-based
    if mon_idx < 0 or mon_idx >= nMon_use:
        print(f"[info] monitorFFT={monitorFFT} not available. Available monitors: 1..{nMon_use}. Skipping SST/orders.")
    else:
        mon_col = mon_H_cols[mon_idx]
        plot_sst_orders_only(
            dfm, mask_lastFFT, rpm, nFFTrev, mon_col,
            os.path.join("postProcessing", f"mon{monitorFFT}_SST_orders_last{nFFTrev}rev.png"),
            os.path.join("postProcessing", f"mon{monitorFFT}_SST_orders_last{nFFTrev}rev.eps"),
            max_order=max_order
        )
else:
    print("[info] No p monitor data found (skipping SST/orders).")

# ---- pump performance plots
plot_pump(
    dfm, mask_all, rpm, axis.upper(), T_col,
    os.path.join("postProcessing", "pumpPerformance_plots_all.png"),
    os.path.join("postProcessing", "pumpPerformance_plots_all.eps"),
)
plot_pump(
    dfm, mask_lastPlot, rpm, axis.upper(), T_col,
    os.path.join("postProcessing", f"pumpPerformance_plots_last{nPlotRev}rev.png"),
    os.path.join("postProcessing", f"pumpPerformance_plots_last{nPlotRev}rev.eps"),
)

# ---- dedicated Cp plots
plot_cp(
    dfm, mask_all, rpm,
    os.path.join("postProcessing", "Cp_all.png"),
    os.path.join("postProcessing", "Cp_all.eps"),
)
plot_cp(
    dfm, mask_lastPlot, rpm,
    os.path.join("postProcessing", f"Cp_last{nPlotRev}rev.png"),
    os.path.join("postProcessing", f"Cp_last{nPlotRev}rev.eps"),
)

# ---- non-dimensional coefficient plots
plot_non_dim_coeff(
    dfm, mask_all, rpm,
    "phi [-]",
    r"Flow coefficient $\phi$",
    r"$\phi$ [-]",
    os.path.join("postProcessing", "phi_all.png"),
    os.path.join("postProcessing", "phi_all.eps"),
)
plot_non_dim_coeff(
    dfm, mask_lastPlot, rpm,
    "phi [-]",
    rf"Flow coefficient $\phi$ (last {nPlotRev} rev)",
    r"$\phi$ [-]",
    os.path.join("postProcessing", f"phi_last{nPlotRev}rev.png"),
    os.path.join("postProcessing", f"phi_last{nPlotRev}rev.eps"),
)

plot_non_dim_coeff(
    dfm, mask_all, rpm,
    "psi [-]",
    r"Head coefficient $\psi$",
    r"$\psi$ [-]",
    os.path.join("postProcessing", "psi_all.png"),
    os.path.join("postProcessing", "psi_all.eps"),
)
plot_non_dim_coeff(
    dfm, mask_lastPlot, rpm,
    "psi [-]",
    rf"Head coefficient $\psi$ (last {nPlotRev} rev)",
    r"$\psi$ [-]",
    os.path.join("postProcessing", f"psi_last{nPlotRev}rev.png"),
    os.path.join("postProcessing", f"psi_last{nPlotRev}rev.eps"),
)

plot_non_dim_coeff(
    dfm, mask_all, rpm,
    "lambda_h [-]",
    r"Hydraulic power coefficient $\lambda_h$",
    r"$\lambda_h$ [-]",
    os.path.join("postProcessing", "lambda_h_all.png"),
    os.path.join("postProcessing", "lambda_h_all.eps"),
)
plot_non_dim_coeff(
    dfm, mask_lastPlot, rpm,
    "lambda_h [-]",
    rf"Hydraulic power coefficient $\lambda_h$ (last {nPlotRev} rev)",
    r"$\lambda_h$ [-]",
    os.path.join("postProcessing", f"lambda_h_last{nPlotRev}rev.png"),
    os.path.join("postProcessing", f"lambda_h_last{nPlotRev}rev.eps"),
)

plot_non_dim_coeff(
    dfm, mask_all, rpm,
    "lambda_m [-]",
    r"Mechanical power coefficient $\lambda_m$",
    r"$\lambda_m$ [-]",
    os.path.join("postProcessing", "lambda_m_all.png"),
    os.path.join("postProcessing", "lambda_m_all.eps"),
)
plot_non_dim_coeff(
    dfm, mask_lastPlot, rpm,
    "lambda_m [-]",
    rf"Mechanical power coefficient $\lambda_m$ (last {nPlotRev} rev)",
    r"$\lambda_m$ [-]",
    os.path.join("postProcessing", f"lambda_m_last{nPlotRev}rev.png"),
    os.path.join("postProcessing", f"lambda_m_last{nPlotRev}rev.eps"),
)

# ---- volute monitor plots
if have_p and nMon_use > 0:
    plot_volute_monitors(
        dfm, mon_H_cols, mask_all, rpm,
        os.path.join("postProcessing", "voluteMonitors_head_all.png"),
        os.path.join("postProcessing", "voluteMonitors_head_all.eps"),
    )
    plot_volute_monitors(
        dfm, mon_H_cols, mask_lastPlot, rpm,
        os.path.join("postProcessing", f"voluteMonitors_head_last{nPlotRev}rev.png"),
        os.path.join("postProcessing", f"voluteMonitors_head_last{nPlotRev}rev.eps"),
    )
else:
    print("No p monitor data found in postProcessing/volutMonitors/*/p (skipping monitor plots).")

# ---- sanity check
if t_arr.size > 1 and np.any(mask_lastPlot):
    twin = t_arr[mask_lastPlot]
    nPtsPerRev_expected = 360.0 / degPerTimeStep
    nPtsPlot_expected = nPlotRev * nPtsPerRev_expected

    print("Rotor step per timestep [deg]:", degPerTimeStep)
    print("Expected timestep dt [s]:", dt_expected)
    print("Expected 1-rev time [s]:", Trev)
    print(f"Expected plot window ({nPlotRev} rev) [s]:", nPlotRev * Trev)
    print(f"Actual plot window last{nPlotRev}rev [s]:", float(twin[-1] - twin[0]))
    print(f"Points in last{nPlotRev}rev window:", int(twin.size))
    print(f"Expected points per rev: ~{nPtsPerRev_expected:.3f}")
    print(f"Expected points in last{nPlotRev}rev window: ~{nPtsPlot_expected:.3f}")

print(f"Blade tip speed U2 based on Dimpeller [m/s]: {U2:.8g}")
print(f"Dynamic pressure based on U2 [Pa]: {q2:.8g}")
print(f"Reference speed U_ref based on Doutput [m/s]: {U_ref:.8g}")
print(f"Dimpeller [m]: {Dimpeller:.8g}")
print(f"Doutput [m]: {Doutput:.8g}")
if use_area_based_phi and b2 is not None:
    print(f"Area-based phi enabled with b2 [m]: {b2:.8g}")
else:
    print("Area-based phi disabled.")

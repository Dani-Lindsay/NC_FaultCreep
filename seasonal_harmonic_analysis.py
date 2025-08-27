
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from typing import Dict, Tuple

# ---------- Core utilities ----------

def harmonic_model(t: np.ndarray, v0: float, a1: float, b1: float) -> np.ndarray:
    """v0 + a1*sin(2πt) + b1*cos(2πt); t is in years, period=1 year."""
    return v0 + a1*np.sin(2*np.pi*t) + b1*np.cos(2*np.pi*t)

def fit_harmonic_with_uncertainty(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Linear least squares fit for v0, a1, b1 and uncertainties for phase and peak time.
    Returns dict with v0, a1, b1, amplitude A, phase phi (rad), peak_frac (0..1),
    phi_std (rad), and peak_std (fraction of year).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(t), np.sin(2*np.pi*t), np.cos(2*np.pi*t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    v0, a1, b1 = beta
    A = np.hypot(a1, b1)
    phi = np.arctan2(b1, a1)
    peak_frac = (0.25 - phi/(2*np.pi)) % 1.0

    # Covariance-based uncertainties
    resid = y - X @ beta
    dof = len(y) - 3
    sigma2 = np.sum(resid**2) / max(dof, 1)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    var_a1, var_b1, cov_ab = cov_beta[1,1], cov_beta[2,2], cov_beta[1,2]

    dphi_da = -b1/(a1**2 + b1**2 + 1e-12)
    dphi_db =  a1/(a1**2 + b1**2 + 1e-12)
    var_phi = (dphi_da**2)*var_a1 + (dphi_db**2)*var_b1 + 2*dphi_da*dphi_db*cov_ab
    phi_std = float(np.sqrt(max(var_phi, 0.0)))
    peak_std = phi_std/(2*np.pi)

    return dict(v0=float(v0), a1=float(a1), b1=float(b1),
                A=float(A), phi=float(phi), peak_frac=float(peak_frac),
                phi_std=float(phi_std), peak_std=float(peak_std))

def decimal_year_from_datetime(dt: pd.Timestamp) -> float:
    year = dt.year
    start = pd.Timestamp(year=year, month=1, day=1, tz=dt.tz)
    end = pd.Timestamp(year=year+1, month=1, day=1, tz=dt.tz)
    return year + (dt - start) / (end - start)

def frac_to_month_day(frac: float) -> Tuple[int, int]:
    """Map fraction of (calendar) year to approximate month/day (non-leap)."""
    doy = frac * 365.2425
    remain = doy
    for m in range(1, 13):
        dim = calendar.monthrange(2001, m)[1]
        if remain <= dim:
            return m, int(round(remain))
        remain -= dim
    return 12, 31

# ---------- Main analysis entry point ----------

def analyze_velocity_and_precip(
    years_season: np.ndarray,
    vel_season: np.ndarray,
    precip_daily: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "rain_mm",
    decimal_year_col: str = "decimal_year",
    title_prefix: str = "MWIL"
) -> Dict[str, float]:
    """
    Run harmonic fits for velocity and daily precipitation (1992–2007),
    produce two separate figures (velocity and precipitation), and print
    useful stats. Returns a dict of key numbers.
    """
    # ---- Velocity fit ----
    vfit = fit_harmonic_with_uncertainty(years_season, vel_season)
    v0, A_v, phi_v = vfit["v0"], vfit["A"], vfit["phi"]
    peak_v, peak_v_std = vfit["peak_frac"], vfit["peak_std"]

    # ---- Precip daily prep ----
    df = precip_daily.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if decimal_year_col not in df.columns:
        df[decimal_year_col] = df[date_col].apply(decimal_year_from_datetime)

    df = df[(df[decimal_year_col] >= 1992) & (df[decimal_year_col] <= 2007)]
    t_daily = df[decimal_year_col].to_numpy()
    y_daily = pd.to_numeric(df[value_col], errors="coerce").fillna(0).to_numpy()

    # ---- Precip fit ----
    pfit = fit_harmonic_with_uncertainty(t_daily, y_daily)
    peak_p, peak_p_std = pfit["peak_frac"], pfit["peak_std"]

    # ---- Offsets ----
    signed_delta = ((peak_p - peak_v + 0.5) % 1.0) - 0.5  # precip - velocity
    delta_days = signed_delta * 365.2425

    # ---- Print stats ----
    p_mo, p_day = frac_to_month_day(peak_p)
    p_day_std = int(round(peak_p_std * 365.2425))
    v_mo, v_day = frac_to_month_day(peak_v)
    v_day_std = int(round(peak_v_std * 365.2425))

    print(f"[Velocity]  v0 = {v0:.3f} mm/yr, A = {A_v:.3f} mm/yr, "
          f"phi = {np.degrees(phi_v):.1f}°")
    print(f"[Velocity]  Peak fraction: {peak_v:.3f} ± {peak_v_std:.3f} "
          f"(~{calendar.month_name[v_mo]} {v_day} ± {v_day_std} d)")
    print(f"[Precip]    Peak fraction: {peak_p:.3f} ± {pfit['peak_std']:.3f} "
          f"(~{calendar.month_name[p_mo]} {p_day} ± {p_day_std} d)")
    print(f"[Offset]    Precip - Velocity = {signed_delta:+.3f} yr "
          f"(~{delta_days:+.1f} days)")

    # ---- Plot 1: Velocity (panel C style, no legend) ----
    t_fit_v = np.linspace(years_season.min(), years_season.max(), 1000)
    y_fit_v = harmonic_model(t_fit_v, vfit["v0"], vfit["a1"], vfit["b1"])

    fig1, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(years_season, vel_season, marker="o", linewidth=1.6)
    ax1.plot(t_fit_v, y_fit_v, linestyle="--", linewidth=2.0)
    years_int = np.arange(int(np.floor(years_season.min())),
                          int(np.ceil(years_season.max())) + 1)
    for y in years_int:
        ax1.axvspan(y + peak_v - peak_v_std, y + peak_v + peak_v_std, alpha=0.15)
    ax1.set_ylabel("Velocity (mm/yr)")
    ax1.set_xlabel("Year")
    ax1.set_title(f"{title_prefix} Velocity with Harmonic Seasonal Peak (1992–2007)")
    ax1.grid(False)
    ax1.text(0.02, 0.03,
        f"Peak ~ {calendar.month_name[v_mo]} {v_day} (±{v_day_std} d)",
        transform=ax1.transAxes, ha="left", va="bottom")
    plt.tight_layout()

    # ---- Plot 2: Daily precipitation (panel F style, no legend) ----
    t_fit_p = np.linspace(t_daily.min(), t_daily.max(), 1000)
    y_fit_p = harmonic_model(t_fit_p, pfit["v0"], pfit["a1"], pfit["b1"])

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.plot(t_daily, y_daily, ".", markersize=1, alpha=0.5)
    ax2.plot(t_fit_p, y_fit_p, linestyle="--", linewidth=2.0)
    for y in years_int:
        ax2.axvspan(y + peak_p - peak_p_std, y + peak_p + peak_p_std, alpha=0.15)
    ax2.set_ylabel("Daily precipitation (mm)")
    ax2.set_xlabel("Year")
    ax2.set_title(f"{title_prefix} Daily Precip with Harmonic Seasonal Peak (1992–2007)")
    ax2.grid(False)
    ax2.text(0.02, 0.03,
        f"Peak ~ {calendar.month_name[p_mo]} {p_day} (±{p_day_std} d)\n"
        f"Precip - Velocity offset: {signed_delta:+.2f} yr (~{delta_days:+.1f} d)",
        transform=ax2.transAxes, ha="left", va="bottom")
    plt.tight_layout()

    return {
        "velocity_peak_fraction": peak_v,
        "velocity_peak_fraction_std": peak_v_std,
        "precip_peak_fraction": peak_p,
        "precip_peak_fraction_std": peak_p_std,
        "delta_precip_minus_velocity_years": signed_delta,
        "delta_precip_minus_velocity_days": delta_days,
        "v0": v0, "A_velocity": A_v
    }

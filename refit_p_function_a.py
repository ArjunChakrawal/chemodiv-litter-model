# refit_p_function_a.py
# Refit the lignin protection function parameter a:
#   k = A * exp[-(x/a)^2]
# using both aromatic C fraction (old predictor) and MMM lignin C fraction (new predictor).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.metrics import r2_score
import helper_function_MMM as mmm

# ============================================================
# Constants
# ============================================================
NMR_BANDS = [
    "A ALKYL 0–45 ppm", "B METHOX 45–60 ppm", "C O-ALKYL 60-95 ppm",
    "D DI-O-ALK 95–110ppm", "E AROM 110–145 ppm", "F PHEN 145–165 ppm",
    "G CARBOX 165-210 ppm",
]

P0     = [0.15, 10.0]                 # initial guess [a, A]
BOUNDS = ([1e-6, 1e-6], [2.0, 1e4])  # a > 0, A > 0

# ============================================================
# Model and fitting helpers
# ============================================================
def p_model(x, a, A):
    """Protection function p(L) = A * exp[-(L/a)^2]"""
    return A * np.exp(-((x / a) ** 2))


def fit_p_function(x, k, label):
    """Fit k = A exp[-(x/a)^2]. Returns fitted parameters and diagnostics."""
    x = np.asarray(x, dtype=float)
    k = np.asarray(k, dtype=float)
    mask = np.isfinite(x) & np.isfinite(k) & (x > 0) & (k > 0)
    x, k = x[mask], k[mask]

    popt, pcov = curve_fit(p_model, x, k, p0=P0, bounds=BOUNDS, maxfev=100_000)
    a_hat, A_hat = popt
    a_se, A_se = np.sqrt(np.diag(pcov))
    k_pred = p_model(x, a_hat, A_hat)

    return {
        "label": label, "x": x, "k": k,
        "a": a_hat, "A": A_hat,
        "a_se": a_se, "A_se": A_se,
        "r2": r2_score(k, k_pred),
        "rmse": np.sqrt(np.mean((k - k_pred) ** 2)),
        "n": len(x),
    }


def load_and_prepare_bonanomi_data(excel_file):
    """Load Bonanomi data and prepare dataframes with NMR normalization."""
    cn      = pd.read_excel(excel_file, sheet_name="CN")
    nmr_raw = pd.read_excel(excel_file, sheet_name="NMR", header=None)

    nmr = nmr_raw.iloc[1:].reset_index(drop=True)
    nmr.columns = [
        "Species", "Days",
        "G CARBOX 165-210 ppm", "F PHEN 145–165 ppm", "E AROM 110–145 ppm",
        "D DI-O-ALK 95–110ppm", "C O-ALKYL 60-95 ppm", "B METHOX 45–60 ppm",
        "A ALKYL 0–45 ppm", "_70_75", "_52_57", "lignin_amount", "lignin_percent",
    ]

    cn["Species"]               = cn["Species"].astype(str).str.strip()
    nmr["Species"]              = nmr["Species"].astype(str).str.strip()
    cn["Days of decomposition"] = pd.to_numeric(cn["Days of decomposition"], errors="coerce")
    cn["mass reamainng %"]      = pd.to_numeric(cn["mass reamainng %"], errors="coerce")
    nmr["Days"]                 = pd.to_numeric(nmr["Days"], errors="coerce")

    for col in NMR_BANDS:
        nmr[col] = pd.to_numeric(nmr[col], errors="coerce")

    # Normalize NMR bands to sum = 1
    nmr[NMR_BANDS] = nmr[NMR_BANDS].div(nmr[NMR_BANDS].sum(axis=1), axis=0)

    return cn, nmr


def build_fitting_dataset(cn, nmr):
    """Extract species-level early-stage k, aromatic fraction, and MMM lignin fraction."""
    records = []

    for species in sorted(nmr["Species"].dropna().unique()):
        nmr_sp = nmr[nmr["Species"] == species].sort_values("Days").reset_index(drop=True)
        cn_sp  = cn[cn["Species"]  == species].sort_values("Days of decomposition").reset_index(drop=True)

        if len(nmr_sp) < 2 or len(cn_sp) < 2:
            continue

        t0, t1 = nmr_sp.loc[0, "Days"], nmr_sp.loc[1, "Days"]
        if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
            continue

        # Aromatic C fraction at t=0
        aromatic_frac_init = (
            nmr_sp.loc[0, "F PHEN 145–165 ppm"] + nmr_sp.loc[0, "E AROM 110–145 ppm"]
        )

        # MMM lignin at t=0
        cn_cn  = cn_sp[["Days of decomposition", "C-to-N ratio"]].rename(
            columns={"Days of decomposition": "Days"}
        )
        nmr_sp = nmr_sp.merge(cn_cn, on="Days", how="left")
        cn_ratio = nmr_sp.iloc[0].get("C-to-N ratio", np.nan)
        if pd.isna(cn_ratio):
            continue

        nmr_input = nmr_sp.iloc[0].copy()
        nmr_input["C:N"] = cn_ratio
        for band_key in "abcdefg":
            nmr_input[band_key] = band_key

        try:
            frac, *_ = mmm.molecular_mixing_model(nmr_input)
            lignin_frac_mmm = frac[2]
        except Exception:
            continue

        # Early-stage k from non-aromatic C decay
        cn_mass = cn_sp[["Days of decomposition", "mass reamainng %"]].rename(
            columns={"Days of decomposition": "Days"}
        )
        nmr_sp = nmr_sp.merge(cn_mass, on="Days", how="left")
        mass_remaining = nmr_sp["mass reamainng %"].to_numpy(dtype=float)
        if np.isnan(mass_remaining[0]) or np.isnan(mass_remaining[1]):
            continue

        arom_series = (
            nmr_sp["F PHEN 145–165 ppm"] + nmr_sp["E AROM 110–145 ppm"]
        ).to_numpy(dtype=float)
        non_arom_C = (1.0 - arom_series) * mass_remaining

        if non_arom_C[0] <= 0 or non_arom_C[1] <= 0:
            continue

        k_early = -(365.0 / (t1 - t0)) * np.log(non_arom_C[1] / non_arom_C[0])

        records.append({
            "Species":               species,
            "t1_days":               t1,
            "k_early_per_year":      k_early,
            "aromatic_frac_initial": aromatic_frac_init,
            "lignin_frac_initial":   lignin_frac_mmm,
            "non_arom_C0":           non_arom_C[0],
            "non_arom_C1":           non_arom_C[1],
        })

    return pd.DataFrame(records)


def fit_lignin_protection(excel_file):
    """
    Main entry point: load data, build fitting dataset, and fit protection function.
    
    Returns dict with keys:
      - fit_arom: aromatic C fraction fit results
      - fit_lignin: MMM lignin C fraction fit results
      - fit_df: full fitting dataset (16 species)
      - s: zero-intercept scaling factor (Lig = s * Arom)
    """
    cn, nmr = load_and_prepare_bonanomi_data(excel_file)
    fit_df = build_fitting_dataset(cn, nmr)
    
    fit_arom   = fit_p_function(fit_df["aromatic_frac_initial"], fit_df["k_early_per_year"],
                                 "Aromatic C fraction")
    fit_lignin = fit_p_function(fit_df["lignin_frac_initial"],   fit_df["k_early_per_year"],
                                 "MMM lignin C fraction")
    
    # Scaling diagnostic
    Arom  = fit_df["aromatic_frac_initial"].to_numpy(dtype=float)
    Lig   = fit_df["lignin_frac_initial"].to_numpy(dtype=float)
    valid = np.isfinite(Arom) & np.isfinite(Lig) & (Arom > 0) & (Lig > 0)
    s     = np.dot(Arom[valid], Lig[valid]) / np.dot(Arom[valid], Arom[valid])
    
    return {
        "fit_arom": fit_arom,
        "fit_lignin": fit_lignin,
        "fit_df": fit_df,
        "s": s,
    }


def print_fit(res):
    """Pretty-print fit results."""
    print(f"\n{'=' * 70}\n{res['label']}\n{'=' * 70}")
    for key in ("n", "a", "a_se", "A", "A_se"):
        fmt = "d" if key == "n" else ".6f"
        print(f"{key:<8} = {res[key]:{fmt}}")
    print(f"R2       = {res['r2']:.4f}")
    print(f"RMSE     = {res['rmse']:.4f}")


# ============================================================
# Main script: run analysis and generate plots
# ============================================================
if __name__ == "__main__":
    EXCEL_FILE = "data\\Bonanomi 2011 NPH\\C and N.xlsx"
    FIG_OUT    = "data\\Bonanomi 2011 NPH\\refit_p_function_a.png"
    CSV_OUT    = "data\\Bonanomi 2011 NPH\\p_function_refit_data.csv"
    SAVE_FIG   = True
    
    results = fit_lignin_protection(EXCEL_FILE)
    
    fit_arom = results["fit_arom"]
    fit_lignin = results["fit_lignin"]
    fit_df = results["fit_df"]
    s = results["s"]
    
    print(f"\nSpecies used: {len(fit_df)}")
    print(fit_df[["Species", "aromatic_frac_initial", "lignin_frac_initial", "k_early_per_year"]])
    
    print_fit(fit_arom)
    print_fit(fit_lignin)
    
    print(f"\n{'=' * 70}\nScaling diagnostic\n{'=' * 70}")
    print(f"MMM lignin = s x aromatic,  s = {s:.4f}  (lignin > aromatic when s > 1)")
    print(f"Algebraic a_lignin = s x a_arom = {s * fit_arom['a']:.4f}")
    print(f"Directly fitted a_lignin        = {fit_lignin['a']:.4f}")
    
    
    # ============================================================
    # Plot
    # ============================================================
    # Scatter: aromatic vs MMM lignin
    slope, intercept, r_value, p_value, _ = linregress(
        fit_df["lignin_frac_initial"], fit_df["aromatic_frac_initial"]
    )

    fig0, ax0 = plt.subplots(figsize=(4.5, 4.0))
    ax0.scatter(fit_df["lignin_frac_initial"], fit_df["aromatic_frac_initial"],
                s=40, color="black", zorder=3)
    lim = max(fit_df["lignin_frac_initial"].max(), fit_df["aromatic_frac_initial"].max()) * 1.1
    x_s = np.linspace(0, lim, 200)
    ax0.plot([0, lim], [0, lim], "k--", linewidth=1, label="1:1")
    ax0.plot(x_s, x_s / s, color="steelblue", linewidth=1.5,
             label=fr"zero-intercept: $y = x/{s:.2f}$")
    ax0.plot(x_s, slope * x_s + intercept, color="tomato", linewidth=1.5,
             label=fr"OLS: $y = {slope:.2f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.3f}$"
                   fr"  ($R^2={r_value**2:.2f}$, $p={p_value:.3f}$)")
    ax0.set_xlabel("MMM lignin C fraction")
    ax0.set_ylabel("Aromatic C fraction (NMR)")
    ax0.set_xlim(0, lim)
    ax0.set_ylim(0, lim)
    ax0.legend(frameon=False, fontsize=9)
    ax0.set_title("Aromatic vs MMM Lignin")
    ax0.grid(False)
    fig0.tight_layout()
    if SAVE_FIG:
        fig0.savefig(FIG_OUT.replace(".png", "_arom_vs_lignin.png"), dpi=300, bbox_inches="tight")

    x_arom = np.linspace(0, max(fit_arom["x"].max(), 0.5), 300)
    x_lig  = np.linspace(0, max(fit_lignin["x"].max(), 0.5), 300)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=False)

    for ax, res, x_grid, xlabel, title in [
        (axes[0], fit_arom,   x_arom, "Initial aromatic C fraction",   "Aromatic-C predictor"),
        (axes[1], fit_lignin, x_lig,  "Initial lignin C fraction", "MMM Lignin-C predictor"),
    ]:
        ax.scatter(res["x"], res["k"] / res["A"], s=35, color="black", label="observed")
        ax.plot(
            x_grid, p_model(x_grid, res["a"], res["A"]) / res["A"],
            linewidth=2, color="black",
            label=fr"fit: $a$={res['a']:.3f}, $R^2$={res['r2']:.2f}",
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(False)
        ax.legend(frameon=False)

    # Reference candidate curves on the lignin panel
    for a_val, lbl, ls in [
        (0.15,   r"$a=0.15$ (aromatic fit)", "--"),
        # (0.28,   r"$a=0.28$",               "-."),
    ]:
        axes[1].plot(x_lig, np.exp(-((x_lig / a_val) ** 2)),
                     linewidth=1.2, linestyle=ls, label=lbl)
    axes[1].legend(frameon=False, fontsize=8)

    axes[0].set_ylabel(r"Normalized early decay rate, $k/A$")
    axes[1].set_ylabel(r"Normalized early decay rate, $k/A$")

    fig.tight_layout()

    if SAVE_FIG:
        fig.savefig(FIG_OUT, dpi=400, bbox_inches="tight")

    plt.show()

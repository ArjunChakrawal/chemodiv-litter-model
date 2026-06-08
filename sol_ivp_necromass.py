# -*- coding: utf-8 -*-
"""
Chemically resolved litter decomposition model with an explicit microbial necromass pool.

Revision relative to sol_ivp_.py:
    1. Plant-derived litter pools and microbial-necromass pools are tracked separately.
    2. Lignin protection is applied only to plant-derived carbohydrate and protein pools.
    3. Necromass-derived pools decompose without lignin protection, i.e., p = 1.
    4. The observable model outputs keep the original names and are calculated as
       plant-derived pool + necromass-derived pool, so the fitting interface is unchanged.
    5. Microbial biomass is retained as quasi-steady state: T = G = CUE * total uptake.

Author: Arjun Chakrawal
"""

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error
from refit_p_function_a import fit_lignin_protection


POOL_NAMES = ["carbohydrate", "protein", "lignin", "lipid", "carbonyl"]
POOL_COLS = [f"{p}_gC" for p in POOL_NAMES]
PLANT_COLS = [f"{p}_plant_gC" for p in POOL_NAMES]
NEC_COLS = [f"{p}_nec_gC" for p in POOL_NAMES]


def load_fix_par_and_data():
    """
    Loads fixed model parameters and processed plant data.
    """
    # Q10 values taken from Allison et al. 2018 GCB.
    # Lipid is assumed to be same as lignin; carbonyl same as carbohydrate.
    results = fit_lignin_protection("data\\Bonanomi 2011 NPH\\C and N.xlsx")
    a = results["fit_lignin"]["a"]          # 0.240  — use this in the model
    print(f"Refit p-function a parameter: {a:.3f}")

    Q10 = {"Ch": 1.6, "P": 2.25, "Lig": 1.65, "Lip": 1.65, "Cr": 1.6}

    Ea_dict = {}
    T = 273 + 15
    for item, q10_value in Q10.items():
        # EAs are currently set to zero in the fixed parameter set.
        Ea_dict[item] = 0 * 8.314 * np.log(q10_value) * T * (T + 10) / 10

    fixed_param = {
        "CNB": 16,
        "Inorg": 1e-9,
        "a": a, # estimated from refit_p_function_a.py function
        "b": 2,
        "nosc": np.array([0, 0.034, -0.381, -1.471, 3]).reshape((1, 5)),
        "CNP": 1 / (0.27 * 14 / 12),
        "mLg": 0.1,
        "mLp": 0.4,
        "mCr": 0.05,
        "Ea": Ea_dict,
    }

    fixed_param["mP"] = fixed_param["CNP"] / fixed_param["CNB"]
    fixed_param["mC"] = (
        1
        - fixed_param["mP"]
        - fixed_param["mLp"]
        - fixed_param["mLg"]
        - fixed_param["mCr"]
    )

    plant_data = pd.read_excel("data/processed_data.xlsx")
    plant_data = plant_data[~plant_data["Study"].isin(["Xu et al 2017 SBB"])].reset_index(drop=True)
    plant_data = plant_data[~(plant_data["R-squared"] < 0)].reset_index(drop=True)

    rows_to_delete = (plant_data["Study"] == "Preston et al. 2009") & (plant_data["time day"] == 730)
    plant_data.loc[rows_to_delete, ["protein_gC", "protein_gN"]] = np.nan

    mask = (plant_data["protein_gC"] < 1e-6) & (plant_data["time day"] != 0)
    plant_data.loc[mask, "protein_gC"] = np.nan

    mask = (plant_data["carbonyl_gC"] < 1e-6) & (plant_data["time day"] != 0)
    plant_data.loc[mask, "carbonyl_gC"] = np.nan

    return fixed_param, plant_data


def efficiency(gamma):
    """
    Calculate carbon use efficiency as a function of substrate degree of reduction.
    """
    gamma = np.asarray(gamma, dtype=float)
    gamma_O2 = 4
    dfGH2O = -237.2
    dfGO2aq = 16.5
    dGred_O2 = 2 * dfGH2O - dfGO2aq

    def dGox_func(x):
        return 60.3 - 28.5 * (4 - x)

    def dCG_O2(x):
        return dGox_func(x) + (x / gamma_O2) * dGred_O2

    gamma_B = 4.2
    dGred_eA = 2 * dfGH2O - dfGO2aq
    dGrX = np.where(gamma < 4.67, -(666.7 / gamma + 243.1), -(157 * gamma - 339))

    dCGX = dCG_O2(gamma_B)
    dGana = (gamma_B / gamma) * dCG_O2(gamma) - dCGX
    dG_ox = dGox_func(gamma)
    dGcat = dG_ox + gamma * dGred_eA / gamma_O2
    Y = dGcat / (dGrX - dGana + gamma_B / gamma * dGcat)

    if Y.size == 1:
        return float(Y.item())
    return Y


def _as_rate_array(guess_param_val):
    """Accept either a dict or a length-5 iterable of rate constants."""
    if isinstance(guess_param_val, dict):
        return np.array(list(guess_param_val.values()), dtype=float)
    return np.array(guess_param_val, dtype=float)


def _safe_divide(num, den, default=0.0):
    if np.abs(den) < 1e-30:
        return default
    return num / den


def litter_decay_model(tsim, init_fracC, guess_param_val, fixed_param, adapt_flag, protection, CUEflag, voflag):
    """
    Solve the litter decomposition model with separate plant-derived and microbial-necromass pools.

    State variables
    ---------------
    plant pools:
        carbohydrate_plant, protein_plant, lignin_plant, lipid_plant, carbonyl_plant
    necromass pools:
        carbohydrate_nec, protein_nec, lignin_nec, lipid_nec, carbonyl_nec
    cumulative CO2:
        CO2

    Observable pool i is reported as:
        M_i_observed = M_i_plant + M_i_nec

    Model assumptions
    -----------------
    - Plant carbohydrate and protein uptake are reduced by lignin shielding when protection=True.
    - Necromass uptake is not reduced by lignin shielding.
    - Necromass decay uses the same five fitted rate constants as plant litter decay.
    - Microbial biomass is at quasi-steady state: T = G = CUE * U_tot.
    - Under N-Retention, CUE remains at the C-limited value and eta is adjusted.
    - Under Flexible CUE, eta = 0 and CUE is reduced when N limitation occurs.
    """

    tsim = np.asarray(tsim, dtype=float)
    init_fracC = np.asarray(init_fracC, dtype=float)

    def derivatives(t, state, guess_param_val, fixed_param, adapt_flag, protection, CUEflag, voflag):
        state = np.maximum(np.asarray(state, dtype=float), 0.0)
        plant = state[0:5]
        nec = state[5:10]
        fCO2 = state[10]

        T_kelvin = 273.0
        vh_max, vp_max, vlig_max, vlip_max, vCr_max = _as_rate_array(guess_param_val)
        vmax = np.array([vh_max, vp_max, vlig_max, vlip_max, vCr_max], dtype=float)

        a = fixed_param["a"]
        b = fixed_param["b"]
        CNB = fixed_param["CNB"]
        CNP = fixed_param["CNP"]
        Inorg = fixed_param["Inorg"]
        R = 8.314

        EaCh, EaP, EaLig, EaLip, EaCr = list(fixed_param["Ea"].values())
        temp_modifier = np.array(
            [
                np.exp(-EaCh / (R * T_kelvin)),
                np.exp(-EaP / (R * T_kelvin)),
                np.exp(-EaLig / (R * T_kelvin)),
                np.exp(-EaLip / (R * T_kelvin)),
                np.exp(-EaCr / (R * T_kelvin)),
            ],
            dtype=float,
        )
        vmax = vmax * temp_modifier

        plant_C = plant.sum()
        L_plant = _safe_divide(plant[2], plant_C, default=0.0)
        p_lignin = np.exp(-((L_plant / a) ** b))

        plant_mod = np.ones(5)
        if protection:
            plant_mod[0] = p_lignin  # carbohydrate protection
            plant_mod[1] = p_lignin  # protein protection

        if voflag:
            plant_mod[2] = 1.0 - p_lignin
        else:
            plant_mod[2] = 1.0

        # Necromass is not protected by plant lignin. Use p = 1 for all necromass pools.
        U_plant = vmax * plant_mod * plant
        U_nec = vmax * nec *1 # if factor>1 assume necromass pools are more accessible, so use a higher effective rate constant for necromass decay
        U_total_by_pool = U_plant + U_nec
        U_tot = U_total_by_pool.sum()


        if U_tot > 1e-30:
            uptake_composition = U_total_by_pool / U_tot
        else:
            uptake_composition = np.zeros(5)

        DR = 4 - np.dot(fixed_param["nosc"], uptake_composition).item()
        CUE_max = efficiency(DR)

        U_plant_tot = U_plant.sum()
        f_plant_uptake = _safe_divide(U_plant_tot, U_tot, default=0.0)
        if CUEflag:
            # Oxidative-enzyme cost is applied only to the plant-derived
            # fraction of uptake. Necromass-derived uptake dilutes this cost.
            enzyme_cost_fraction = (1.0 - p_lignin) * f_plant_uptake
            CUE_cost = CUE_max * (1.0 - enzyme_cost_fraction)
        else:
            enzyme_cost_fraction = 0.0
            CUE_cost = CUE_max

        G_C_limited = CUE_cost * U_tot
        N_uptake = U_plant[1] / CNP + U_nec[1] / CNP
        Mnet_C_limited = N_uptake - G_C_limited / CNB

        eta = 0.0
        CUE = CUE_cost
        G = G_C_limited
        Mnet = Mnet_C_limited

        if adapt_flag == "Flexible CUE":
            eta = 0.0
            if Mnet_C_limited < -Inorg and U_tot > 1e-30:
                CUE_N = CNB * (N_uptake + Inorg) / U_tot
                CUE = max(0.0, min(CUE_cost, CUE_N))
                G = CUE * U_tot
                Mnet = -Inorg

        elif adapt_flag == "N-Retention":
            if Mnet_C_limited < -Inorg and G_C_limited > 1e-30:
                eta_candidate = 1.0 - CNB * (N_uptake + Inorg) / G_C_limited
                eta = max(0.0, min(eta_candidate, 1.0 - 1e-12))
                CUE = CUE_cost
                G = G_C_limited
                Mnet = N_uptake - (1.0 - eta) * G / CNB
            else:
                eta = 0.0
                CUE = CUE_cost
                G = G_C_limited
                Mnet = Mnet_C_limited
        else:
            raise ValueError("adapt_flag must be either 'Flexible CUE' or 'N-Retention'.")

        # Mortality equals growth under quasi-steady-state microbial biomass.
        T_mort = G

        mP_eff = (1.0 - eta) * fixed_param["mP"]
        mLg = fixed_param["mLg"]
        mLp = fixed_param["mLp"]
        mCr = fixed_param["mCr"]
        mC = 1.0 - mP_eff - mLg - mLp - mCr
        mC = max(mC, 0.0)
        nec_input_frac = np.array([mC, mP_eff, mLg, mLp, mCr], dtype=float)

        dplant_dt = -U_plant
        dnec_dt = nec_input_frac * T_mort - U_nec
        dCO2dt = U_tot - G

        out1 = np.concatenate([dplant_dt, dnec_dt, [dCO2dt]])

        observed_pools = plant + nec
        totCg = observed_pools.sum()
        totNg = observed_pools[1] / CNP
        sumPool = totCg + fCO2

        out2 = {
            "totCg": totCg,
            "totNg": totNg,
            "CUE": CUE,
            "ETA": eta,
            "MNet [gN/day]": Mnet,
            "Growth rate [gC/day]": G,
            "Mortality rate [gC/day]": T_mort,
            "DR": DR,
            "sumPool": sumPool,
            "CUE_star": CUE_max,
            "CUE_cost": CUE_cost,
            "L_plant": L_plant,
            "Utot": U_tot,
            "N_uptake [gN/day]": N_uptake,
        }
        return out1, out2

    def odefun(t, state, guess_param_val, fixed_param, adapt_flag, protection, CUEflag, voflag):
        out1, _ = derivatives(t, state, guess_param_val, fixed_param, adapt_flag, protection, CUEflag, voflag)
        return out1

    # Initial condition: all observed initial material is plant-derived; necromass starts at zero.
    init_plant = init_fracC.copy()
    init_nec = np.zeros(5)
    init_state = np.concatenate([init_plant, init_nec, [0.0]])

    sol = solve_ivp(
        odefun,
        (tsim[0], tsim[-1]),
        init_state,
        args=(guess_param_val, fixed_param, adapt_flag, protection, CUEflag, voflag),
        method="Radau",
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    z = sol.sol(tsim).T
    df_ivp = pd.DataFrame(
        np.hstack((tsim.reshape(len(tsim), 1), z)),
        columns=["time"] + PLANT_COLS + NEC_COLS + ["CO2_gC"],
    )

    # Aggregated observable pools for fitting against NMR-derived pool sizes.
    for pool, plant_col, nec_col in zip(POOL_NAMES, PLANT_COLS, NEC_COLS):
        df_ivp[f"{pool}_gC"] = df_ivp[plant_col] + df_ivp[nec_col]

    diagnostics = []
    diag_keys = None
    for i in range(len(df_ivp)):
        state = df_ivp.loc[i, PLANT_COLS + NEC_COLS + ["CO2_gC"]].values
        _, out2 = derivatives(tsim[i], state, guess_param_val, fixed_param, adapt_flag, protection, CUEflag, voflag)
        if diag_keys is None:
            diag_keys = list(out2.keys())
        diagnostics.append([out2[k] for k in diag_keys])

    tempdf = pd.DataFrame(diagnostics, columns=diag_keys)
    df_ivp = pd.concat((df_ivp, tempdf), axis=1)

    # Keep a convenient column order: time, aggregate pools, plant pools, nec pools, CO2, diagnostics.
    ordered_cols = ["time"] + POOL_COLS + PLANT_COLS + NEC_COLS + ["CO2_gC"] + diag_keys
    return df_ivp[ordered_cols]


def residual_fun(x, data, data_col, tsim, init_fracC, fixed_param, adapt_flag, protection, CUEflag, voflag):
    """Residuals between normalized simulated and observed pool sizes."""
    df = litter_decay_model(tsim, init_fracC, x, fixed_param, adapt_flag, protection, CUEflag, voflag)
    S_cols = df[data_col]
    splines = {col: interp1d(tsim, S_cols[col], kind="linear", fill_value="extrapolate") for col in data_col}

    simC = np.array([splines[col](data["time day"]) / data[col].max() for col in data_col])
    obs = np.array([data[col] / data[col].max() for col in data_col])
    res = simC.flatten() - obs.flatten()
    return res[~np.isnan(res)]


def fit_data(
    guess_param,
    init_fracC,
    fixed_param,
    tsim,
    Temperature,
    data,
    data_col,
    adapt_flag,
    protection,
    CUEflag,
    voflag,
    loss="soft_l1",
):
    """Fit the five decomposition rate constants by nonlinear least squares."""
    print("fitting in progress...")
    initial_guess = np.array(list(guess_param.values()), dtype=float)
    lb = np.ones(len(initial_guess)) * 1e-5
    ub = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)

    res_lsq = least_squares(
        residual_fun,
        initial_guess,
        loss=loss,
        f_scale=0.5,
        bounds=(lb, ub),
        args=(data, data_col, tsim, init_fracC, fixed_param, adapt_flag, protection, CUEflag, voflag),
        verbose=0,
    )

    est_par_name = list(guess_param.keys())
    est_pars = {est_par_name[i]: res_lsq.x[i] for i in range(len(res_lsq.x))}

    try:
        cov_matrix = np.linalg.pinv(res_lsq.jac.T @ res_lsq.jac)
        parameter_uncertainties = np.sqrt(np.diag(cov_matrix))
        est_pars_se = {
            est_par_name[i] + "_se": parameter_uncertainties[i]
            for i in range(len(parameter_uncertainties))
        }
    except Exception:
        est_pars_se = {name + "_se": np.nan for name in est_par_name}

    print("fitting completed...")
    return est_pars, est_pars_se


def cal_perf_matrix(est_pars, init_fracC, fixed_param, tsim, Temperature, data, data_col, adapt_flag, protection, CUEflag, voflag):
    """Calculate model performance metrics and return model-data pairs."""
    df = litter_decay_model(tsim, init_fracC, est_pars, fixed_param, adapt_flag, protection, CUEflag, voflag)
    S_cols = df[data_col]

    perf_matrix = pd.DataFrame(
        index=["r2", "rmse", "AIC"],
        columns=["overall", "totCg", "totNg"] + POOL_COLS,
    )
    splines = {col: interp1d(tsim, S_cols[col], kind="linear", fill_value="extrapolate") for col in data_col}

    for col in data_col:
        y_pred = splines[col](data["time day"])
        y_true = data[col]
        valid_indices = ~np.isnan(y_true)
        if not np.any(valid_indices):
            perf_matrix.loc[["r2", "rmse", "AIC"], col] = np.nan
        else:
            y_true_valid = y_true[valid_indices]
            y_pred_valid = y_pred[valid_indices]
            perf_matrix.loc["r2", col] = r2_score(y_true_valid, y_pred_valid)
            perf_matrix.loc["rmse", col] = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))

    y_pred = np.array([splines[col](data["time day"]) for col in data_col]).flatten()
    y_true = np.array([data[col] for col in data_col]).flatten()
    valid_indices = ~np.isnan(y_true)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]
    perf_matrix.loc["r2", "overall"] = r2_score(y_true, y_pred)
    perf_matrix.loc["rmse", "overall"] = np.sqrt(mean_squared_error(y_true, y_pred))

    temp_col = data_col.copy()
    if "DR" not in temp_col:
        temp_col.append("DR")
    splines = {col: interp1d(tsim, df[col], kind="linear", fill_value="extrapolate") for col in temp_col}

    y_pred1 = np.array([splines[col](data["time day"]) for col in temp_col]).flatten()
    y_true_list = []
    for col in temp_col:
        if col in data.columns:
            y_true_list.append(data[col])
        else:
            y_true_list.append(np.full(len(data["time day"]), np.nan))
    y_true1 = np.array(y_true_list).flatten()

    cat = [np.repeat(col, len(data["time day"])) for col in temp_col]
    cat = np.array(cat).flatten().tolist()
    obstime = np.tile(data["time day"].values, len(temp_col))
    model_data = pd.DataFrame({"time": obstime, "pool": cat, "obs": y_true1, "sim": y_pred1})
    return perf_matrix, model_data


def plot_model(
    tsim,
    fixed_param,
    est_pars,
    init_fracC,
    data_col=None,
    data=None,
    adapt_flag="N-Retention",
    protection=True,
    CUEflag=True,
    voflag=True,
):
    """Plot model trajectories and diagnostic quantities."""
    Temperature = np.ones(len(tsim)) * 273.15
    rate_values = _as_rate_array(est_pars)
    vh_max, vp_max, vlig, vlip, vCr = rate_values
    df = litter_decay_model(tsim, init_fracC, est_pars, fixed_param, adapt_flag, protection, CUEflag, voflag)
    a = fixed_param["a"]

    col = POOL_COLS
    plt.style.use("ggplot")
    palette = sns.color_palette()
    plt.style.use("default")

    fig, ax = plt.subplots(4, 4, figsize=(14, 8))
    ax = ax.flatten()

    ax[0].plot(df["time"], df["totCg"], linewidth=1.5)
    if data is not None:
        ax[0].scatter(data["time day"], data["totCg"], linewidth=1.5)
    ax[0].set_xlabel("Time [d]")
    ax[0].set_ylabel("total C [g]")

    ax[1].plot(df["time"], df["totNg"], linewidth=1.5, label="model")
    if data is not None:
        if "totNg" in data.columns:
            ax[1].scatter(data["time day"], data["totNg"], linewidth=1.5, label="Ng")
        if "protein_gN" in data.columns:
            ax[1].scatter(data["time day"], data["protein_gN"], linewidth=1.5, label="protein_gN")
    ax[1].set_xlabel("Time [d]")
    ax[1].set_ylabel("total N [g]")
    ax[1].legend(fontsize=8, loc="best", frameon=False)

    for i, column in enumerate(col):
        color = palette[i]
        denom = df[column].iloc[0] if df[column].iloc[0] != 0 else 1.0
        ax[2].plot(df["time"], df[column] / denom, label=column, linewidth=1.5, color=color)
        if data is not None and column in data.columns:
            data_denom = data[column].iloc[0] if data[column].iloc[0] != 0 else 1.0
            ax[2].scatter(data["time day"], data[column] / data_denom, linewidth=1.5, color=color)
    ax[2].set_xlabel("Time [d]")
    ax[2].set_ylabel("gC/gC[0]")

    for i, column in enumerate(col):
        color = palette[i]
        ax[3].plot(df["time"], df[column], label=column, linewidth=1.5, color=color)
        if data is not None and column in data.columns:
            ax[3].scatter(data["time day"], data[column], linewidth=1.5, color=color)
    ax[3].set_xlabel("Time [d]")
    ax[3].set_ylabel("gC")
    ax[3].legend(fontsize=8, loc="best", frameon=False)

    ax[4].plot(df["time"], df["DR"], linewidth=1.5)
    if data is not None and "DR" in data.columns:
        ax[4].scatter(data["time day"], data["DR"], linewidth=1.5)
    ax[4].set_xlabel("Time [d]")
    ax[4].set_ylabel("DR")

    ax[5].plot(df["time"], df["MNet [gN/day]"], linewidth=1.5)
    ax[5].set_xlabel("Time [d]")
    ax[5].set_ylabel("MNet [gN/day]")

    ax[6].plot(df["time"], df["CUE"], linewidth=1.5, label="realized CUE")
    ax[6].plot(df["time"], df["CUE_cost"], linewidth=1.0, linestyle="--", label="CUE after enzyme cost")
    ax[6].set_xlabel("Time [d]")
    ax[6].set_ylabel("CUE")
    ax[6].legend(fontsize=8, frameon=False)

    ax[7].plot(df["time"], df["Growth rate [gC/day]"], linewidth=1.5)
    ax[7].set_xlabel("Time [d]")
    ax[7].set_ylabel("Growth rate [gC/day]")

    L = df["L_plant"]
    ax[8].plot(df["time"], L, linewidth=1.5)
    ax[8].set_xlabel("Time [d]")
    ax[8].set_ylabel("Plant lignin fraction")

    if protection:
        pfunc = np.exp(-((L / a) ** 2))
        Ch_protection = vh_max * pfunc
        P_protection = vp_max * pfunc
        lig_modifier = 1.0 - pfunc
        lig_protection = vlig * lig_modifier

        ax[9].plot(L, Ch_protection, linewidth=3, label="Carb")
        ax[9].plot(L, P_protection, linewidth=3, label="Prot")
        if voflag:
            ax[9].plot(L, lig_protection, linewidth=3, label="Lignin")
        ax[9].set_xlabel("Plant lignin fraction")
        ax[9].set_ylabel("effective rate")
        ax[9].legend(fontsize=8, frameon=False)

        Lgrid = np.arange(0, 0.8, 0.01)
        pgrid = np.exp(-((Lgrid / a) ** 2))
        ax[10].plot(Lgrid, pgrid, linewidth=1.5, label="plant C/P modifier")
        if voflag:
            ax[10].plot(Lgrid, 1.0 - pgrid, linewidth=1.5, label="plant lignin modifier")
        ax[10].set_xlabel("Plant lignin fraction")
        ax[10].legend(fontsize=8, frameon=False)

    ax[11].plot(df["time"], df["sumPool"], linewidth=1.5)
    ax[11].set_xlabel("Time [d]")
    ax[11].set_ylabel("mass balance check")

    ax[12].plot(df["time"], df["totCg"] / df["totNg"], linewidth=1.5)
    ax[12].set_xlabel("Time [d]")
    ax[12].set_ylabel(r"$CN$")

    ax[13].scatter(df["totCg"] / df["totNg"], df["ETA"])
    ax[13].set_xlabel(r"$CN$")
    ax[13].set_ylabel(r"$\eta$")

    ax[14].plot(df["time"], df["ETA"], linewidth=1.5)
    ax[14].set_xlabel("Time [d]")
    ax[14].set_ylabel(r"$\eta$")

    ax[15].plot(df["time"], df["protein_nec_gC"], linewidth=1.5, label="protein necromass")
    ax[15].plot(df["time"], df["carbohydrate_nec_gC"], linewidth=1.5, label="carbohydrate necromass")
    ax[15].set_xlabel("Time [d]")
    ax[15].set_ylabel("necromass C [gC]")
    ax[15].legend(fontsize=8, frameon=False)

    for axs in ax:
        axs.grid(True, color="grey", linestyle="-", linewidth=0.25)

    if data is not None and data_col is not None:
        perf_matrix, _ = cal_perf_matrix(
            est_pars,
            init_fracC,
            fixed_param,
            tsim,
            Temperature,
            data,
            data_col,
            adapt_flag,
            protection,
            CUEflag,
            voflag,
        )
        study = str(data["Study"].iloc[0]) if "Study" in data.columns else "study"
        species = str(data["Species"].iloc[0]) if "Species" in data.columns else "species"
        suptitle_str = (
            study
            + "_"
            + species
            + f" [Overall rmse: {perf_matrix.loc['rmse', 'overall']:.2E}, Overall r2: {perf_matrix.loc['r2', 'overall']:.2E}]"
        )
        plt.suptitle(suptitle_str, fontsize=12, va="top")

    plt.tight_layout()
    return fig, df

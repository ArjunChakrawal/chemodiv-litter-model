"""
Figure 2, Figure 3, and supplementary diagnostics for the necromass-explicit
litter decomposition model.

Main changes relative to the older Figure_2_and_3.py script:
1. Uses L_plant, not total lignin fraction, when plotting the lignin-dependent
   protection/enzyme-cost functions.
2. Adds diagnostics for the new model mechanism:
   - plant-derived vs necromass-derived pool fractions,
   - effective protection of the observed carbohydrate and protein pools,
   - plant vs necromass uptake contributions.
3. Keeps aggregate observable pools as plant + necromass, matching the NMR
   calibration target.
"""
#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sol_ivp_necromass as svp

#%%

# -----------------------------------------------------------------------------
# Plot and model setup
# -----------------------------------------------------------------------------

with plt.style.context("ggplot"):
    palette = sns.color_palette()

plt.style.use("default")
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["axes.edgecolor"] = (0.25, 0.25, 0.25)
plt.rcParams["text.color"] = "black"

fixed_param, plant_data = svp.load_fix_par_and_data()

init_fracC = {
    "carbohydrate_gC": 0.0,
    "protein_gC": 0.02,
    "lignin_gC": 0.30,
    "lipid_gC": 0.20,
    "carbonyl_gC": 0.05,
}
init_fracC["carbohydrate_gC"] = (
    1
    - init_fracC["protein_gC"]
    - init_fracC["lignin_gC"]
    - init_fracC["lipid_gC"]
    - init_fracC["carbonyl_gC"]
)
init_vec = np.array(list(init_fracC.values()), dtype=float)

fixed_param["Inorg"] = 1e-5
a, b = fixed_param["a"], fixed_param["b"]

guess_param = {
    "vh_max": 0.008,
    "vp_max": 0.008,
    "vlig": 0.008,
    "vlip": 0.009,
    "vCr": 0.01,
}

v_arr = np.array(list(guess_param.values()), dtype=float)
vh_max, vp_max, vlig_max, vlip_max, vCr_max = v_arr

tsim = np.linspace(0, 365 * 2, 200)

pool_cols = [
    "carbohydrate_gC",
    "protein_gC",
    "lignin_gC",
    "lipid_gC",
    "carbonyl_gC",
]
plant_cols = [
    "carbohydrate_plant_gC",
    "protein_plant_gC",
    "lignin_plant_gC",
    "lipid_plant_gC",
    "carbonyl_plant_gC",
]
nec_cols = [
    "carbohydrate_nec_gC",
    "protein_nec_gC",
    "lignin_nec_gC",
    "lipid_nec_gC",
    "carbonyl_nec_gC",
]

new_col = [
    "Carbohydrate [gC]",
    "Protein [gC]",
    "Lignin [gC]",
    "Lipid [gC]",
    "Carbonyl [gC]",
]
title_str = ["Carbohydrate", "Protein", "Lignin", "Lipid", "Carbonyl"]

lstyle = ["-", "--", ":", "-."]
model_name = ["NPNC", "NPC", "PC", "PCV"]


def run_scenario(mdnam,adapt_flag):
    """Run one scenario and return model output with derived necromass diagnostics."""
    if mdnam == "NPNC":
        protection, CUEflag, voflag = False, False, False
    elif mdnam == "NPC":
        protection, CUEflag, voflag = False, True, False
    elif mdnam == "PC":
        protection, CUEflag, voflag = True, True, False
    elif mdnam == "PCV":
        protection, CUEflag, voflag = True, True, True
    else:
        raise ValueError(f"Unknown model scenario: {mdnam}")

    df = svp.litter_decay_model(
        tsim,
        init_vec,
        guess_param,
        fixed_param,
        adapt_flag=adapt_flag,
        protection=protection,
        CUEflag=CUEflag,
        voflag=voflag,
    )

    eps = 1e-30
    L_plant = df["L_plant"].to_numpy()
    p_plant = np.exp(-((L_plant / a) ** b))

    # Scenario-specific plant modifiers. Necromass modifiers are always 1.
    if protection:
        p_ch_plant = p_plant.copy()
        p_p_plant = p_plant.copy()
    else:
        p_ch_plant = np.ones(len(df))
        p_p_plant = np.ones(len(df))

    if voflag:
        p_lig_plant = 1.0 - p_plant
    else:
        p_lig_plant = np.ones(len(df))

    # Effective protection experienced by the aggregate observable pool.
    # This is the key new diagnostic: as necromass dominates, p_eff approaches 1.
    df["p_Ch_plant"] = p_ch_plant
    df["p_P_plant"] = p_p_plant
    df["p_Lig_plant"] = p_lig_plant

    df["p_eff_Ch"] = (
        p_ch_plant * df["carbohydrate_plant_gC"] + df["carbohydrate_nec_gC"]
    ) / (df["carbohydrate_gC"] + eps)
    df["p_eff_P"] = (
        p_p_plant * df["protein_plant_gC"] + df["protein_nec_gC"]
    ) / (df["protein_gC"] + eps)
    df["p_eff_Lig"] = (
        p_lig_plant * df["lignin_plant_gC"] + df["lignin_nec_gC"]
    ) / (df["lignin_gC"] + eps)

    # Necromass fraction in each observable pool.
    df["f_nec_Ch"] = df["carbohydrate_nec_gC"] / (df["carbohydrate_gC"] + eps)
    df["f_nec_P"] = df["protein_nec_gC"] / (df["protein_gC"] + eps)
    df["f_nec_Lig"] = df["lignin_nec_gC"] / (df["lignin_gC"] + eps)
    df["f_nec_total"] = df[nec_cols].sum(axis=1) / (df[pool_cols].sum(axis=1) + eps)

    # Plant and necromass C uptake fluxes by pool. These are not returned by the
    # solver, but can be reconstructed because we use first-order kinetics.
    df["U_Ch_plant"] = vh_max * p_ch_plant * df["carbohydrate_plant_gC"]
    df["U_P_plant"] = vp_max * p_p_plant * df["protein_plant_gC"]
    df["U_Lig_plant"] = vlig_max * p_lig_plant * df["lignin_plant_gC"]
    df["U_Lp_plant"] = vlip_max * df["lipid_plant_gC"]
    df["U_Cr_plant"] = vCr_max * df["carbonyl_plant_gC"]

    df["U_Ch_nec"] = vh_max * df["carbohydrate_nec_gC"]
    df["U_P_nec"] = vp_max * df["protein_nec_gC"]
    df["U_Lig_nec"] = vlig_max * df["lignin_nec_gC"]
    df["U_Lp_nec"] = vlip_max * df["lipid_nec_gC"]
    df["U_Cr_nec"] = vCr_max * df["carbonyl_nec_gC"]

    df["U_plant_total"] = df[
        ["U_Ch_plant", "U_P_plant", "U_Lig_plant", "U_Lp_plant", "U_Cr_plant"]
    ].sum(axis=1)
    df["U_nec_total"] = df[
        ["U_Ch_nec", "U_P_nec", "U_Lig_nec", "U_Lp_nec", "U_Cr_nec"]
    ].sum(axis=1)
    df["f_U_nec_total"] = df["U_nec_total"] / (df["U_plant_total"] + df["U_nec_total"] + eps)
    df["f_U_P_nec"] = df["U_P_nec"] / (df["U_P_plant"] + df["U_P_nec"] + eps)

    return df


# Run all scenarios once and reuse outputs across figures.
adapt_flags = ["N-Retention", "Flexible CUE"]
scenario_outputs = {}
for adapt_flag in adapt_flags:
    scenario_outputs[adapt_flag] = {
        mdnam: run_scenario(mdnam, adapt_flag=adapt_flag)
        for mdnam in model_name
    }
    # OUTDIR = Path("figs"+adapt_flag.replace(" ", "_"))
    # OUTDIR.mkdir(parents=True, exist_ok=True)

#%%
adapt_flag = "N-Retention"
OUTDIR = "figs/"

# -----------------------------------------------------------------------------
# Figure 2: structural functions plotted against plant lignin fraction
# -----------------------------------------------------------------------------

Lt = np.arange(0, 0.6 + 0.01, 0.01)
CUEmax1 = svp.efficiency(np.ones(len(Lt)) * 3.8)
CUEmax2 = svp.efficiency(np.ones(len(Lt)) * 4.8)
p_grid = np.exp(-((Lt / a) ** b))

fig2, ax2 = plt.subplots(1, 4, figsize=(10.5, 2.5), sharex=False, sharey=False)

ax2[0].fill_between(Lt, CUEmax1, CUEmax2, color="red", alpha=0.3, edgecolor=None)
ax2[0].fill_between(Lt, CUEmax1 * p_grid, CUEmax2 * p_grid, color="red", alpha=0.3, edgecolor=None)

ax2[1].plot(Lt, p_grid, linewidth=1.0, linestyle="-", color="red", alpha=0.5)
ax2[1].plot(Lt, np.ones(len(Lt)), linewidth=1.0, linestyle="-", color="red", alpha=0.5)

ax2[2].plot(Lt, 1 - p_grid, linewidth=1.0, linestyle="-", color="red", alpha=0.5)
ax2[2].plot(Lt, np.ones(len(Lt)), linewidth=1.0, linestyle="-", color="red", alpha=0.5)

ax2[3].plot(Lt, np.ones(len(Lt)), linewidth=1.0, linestyle="-", color="red", alpha=0.5)

# New-model diagnostic panel: effective protection for the observed protein pool.
# This is simulation-derived, so the x-axis is the realized plant-lignin fraction.
for ls, mdnam in zip(lstyle, model_name):
    df = scenario_outputs[adapt_flag][mdnam]
    ax2[0].plot(df["L_plant"], df["CUE"], label=mdnam, linewidth=1, linestyle=ls, color="black")
    ax2[1].plot(df["L_plant"], df["p_Ch_plant"], label=mdnam, linewidth=1, linestyle=ls, color="black")
    ax2[2].plot(df["L_plant"], df["p_Lig_plant"], label=mdnam, linewidth=1, linestyle=ls, color="black")
    ax2[3].plot(df["L_plant"], np.ones(len(df)), label=mdnam, linewidth=1, linestyle=ls, color="black")
    # ax2[4].plot(df["L_plant"], df["p_eff_P"], label=mdnam, linewidth=1, linestyle=ls, color="black")

for i in range(4):
    ax2[i].set_xlabel(r"Plant lignin C fraction ($L_{plant}$)", fontsize=11)
    # ax2[i].set_xticks([0, 0.25, 0.5])
    ax2[i].tick_params(axis="both", labelsize=8)
    ax2[i].grid(False)

ax2[0].set_title("(A)", fontsize=12)
ax2[1].set_title("(B)", fontsize=12)
ax2[2].set_title("(C)", fontsize=12)
ax2[3].set_title("(D)", fontsize=12)
# ax2[4].set_title("(E)", fontsize=12)

ax2[0].set_ylabel("CUE", fontsize=12)
ax2[1].set_ylabel(r"$p_{C_h,plant}=p_{P,plant}$", fontsize=12)
ax2[2].set_ylabel(r"$p_{L_g,plant}$", fontsize=12)
ax2[3].set_ylabel(r"$p_{L_p}=p_{C_n}=1$", fontsize=12)
# ax2[4].set_ylabel(r"$p_{P,eff}$", fontsize=12)
ax2[3].set_ylim(0.95, 1.05)
ax2[3].set_yticks([0.9, 1.0, 1.1])
# ax2[4].set_ylim(0, 1.05)
ax2[0].legend(fontsize=8, loc="best", frameon=False)
# ax2[4].legend(fontsize=8, loc="best", frameon=False)

fig2.tight_layout()
fig2.savefig(OUTDIR + adapt_flag.replace(" ", "_") + "_Figure2_necromass.jpg", bbox_inches="tight", dpi=600)

#% 
# -----------------------------------------------------------------------------
# Figure 3: aggregate observable pools and core model diagnostics
# -----------------------------------------------------------------------------

fig3_main, ax = plt.subplots(2, 5, figsize=(12, 4.6), sharex=True, sharey=False)
fig3_main.subplots_adjust(top=0.925, bottom=0.15, left=0.05, right=0.99, wspace=0.25, hspace=0.6)
sttr = "ABCDEFGHIJ"

for ls, mdnam in zip(lstyle, model_name):
    df = scenario_outputs[adapt_flag][mdnam].copy()
    df_plot = df.rename(columns=dict(zip(pool_cols, new_col)))

    for n, column in enumerate(new_col):
        ax[0, n].plot(
            df_plot["time"] / 365,
            df_plot[column] / df_plot[column].iloc[0],
            label=mdnam,
            linewidth=2,
            linestyle=ls,
            color=palette[n],
        )
        ax[0, n].set_title(f"({sttr[n]}) {title_str[n]}", fontsize=12, color="black")

    ax[1, 0].plot(df["time"] / 365, df["ETA"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    ax[1, 0].set_title(r"(F) $\eta$", fontsize=12, color="black")

    ax[1, 1].plot(df["time"] / 365, df["CUE"], label=mdnam, linewidth=2.5, linestyle=ls, color="black")
    ax[1, 1].set_title("(G) CUE", fontsize=12, color="black")

    # Plant-rate constants, not aggregate effective rates.
    ax[1, 2].plot(
        df["time"] / 365,
        df["p_Ch_plant"] * vh_max,
        label=mdnam,
        linewidth=2.5,
        linestyle=ls,
        color="black",
    )
    ax[1, 2].set_title(r"(H) $v_{C_h,plant}=v_{P,plant}$", fontsize=12, color="black")

    ax[1, 3].plot(
        df["time"] / 365,
        df["p_Lig_plant"] * vlig_max,
        label=mdnam,
        linewidth=2.5,
        linestyle=ls,
        color="black",
    )
    ax[1, 3].set_title(r"(I) $v_{L_g,plant}$", fontsize=12, color="black")

    ax[1, 4].plot(
        df["time"] / 365,
        np.ones(len(df)) * vlip_max,
        label=mdnam,
        linewidth=2.5,
        linestyle=ls,
        color="black",
    )
    ax[1, 4].plot(
        df["time"] / 365,
        np.ones(len(df)) * vCr_max,
        label=mdnam,
        linewidth=2.5,
        linestyle=ls,
        color=[0.7, 0.7, 0.7],
    )
    ax[1, 4].set_title(r"(J) $v_{L_p}, v_{C_n}$", fontsize=12, color="black")

for i in range(5):
    ax[1, i].set_xlabel("Time [Y]", fontsize=11)

ax[0, 0].set_ylabel(r"C/C$_0$", fontsize=12)
ax[1, 0].set_ylim(bottom=0)
ax[1, 1].set_ylim(bottom=0)
ax[1, 2].set_ylim(bottom=0.0)
ax[1, 3].set_ylim(bottom=0.0)
ax[1, 4].set_ylim(bottom=0.008, top=0.012)
ax[1, 4].set_yticks([0.008, 0.009, 0.010, 0.011, 0.012])

line1 = ax[1, 4].plot(np.nan, np.nan, label=r"$v_{L_p}$", linewidth=2, linestyle="-", color="black")
line2 = ax[1, 4].plot(np.nan, np.nan, label=r"$v_{C_n}$", linewidth=2, linestyle="-", color=[0.7, 0.7, 0.7])
ax[1, 4].legend(handles=[line1[0], line2[0]], fontsize=12, ncol=1, loc="upper right", frameon=False)

pts = []
for ls, mdnam in zip(lstyle, model_name):
    pt = ax[0, 0].plot(np.nan, np.nan, label=mdnam, linewidth=2, linestyle=ls, color="black")
    pts.append(pt[0])
ax[0, 0].legend(pts, model_name, fontsize=11, loc="upper right", frameon=False)

for axx in ax.flatten():
    axx.tick_params(axis="both", labelsize=10, direction="out")
    axx.grid(visible=False, which="both")

fig3_main.tight_layout()
fig3_main.savefig(OUTDIR + adapt_flag.replace(" ", "_") + "_Figure3_necromass.jpg", dpi=600, bbox_inches="tight")

#%%
# -----------------------------------------------------------------------------
# Figure S4: total C/N, plant lignin fraction, DR, and CUE diagnostics
# -----------------------------------------------------------------------------

figS4, axS4 = plt.subplots(2, 3, figsize=(11, 5), sharex=True, sharey=False)
axS4 = axS4.flatten()

for ls, mdnam in zip(lstyle, model_name):
    df = scenario_outputs[adapt_flag][mdnam].copy()
    CUE_max = np.array([svp.efficiency(np.array([dr])) for dr in df["DR"].values])

    axS4[0].plot(df["time"] / 365, df["totCg"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS4[1].plot(df["time"] / 365, df["totNg"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS4[2].plot(df["time"] / 365, df["L_plant"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS4[3].plot(df["time"] / 365, df["DR"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS4[4].plot(df["time"] / 365, CUE_max, label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS4[5].plot(
        df["time"] / 365,
        df["CUE_star"] - df["CUE"],
        label=mdnam,
        linewidth=2,
        linestyle=ls,
        color="black",
    )

for i in range(6):
    axS4[i].set_xlabel("Time [Y]", fontsize=10)
    axS4[i].tick_params(axis="both", labelsize=9)
    axS4[i].grid(False)

axS4[0].set_ylabel("total C [g]", fontsize=10)
axS4[1].set_ylabel("total N [g]", fontsize=10)
axS4[2].set_ylabel(r"Plant lignin C fraction [gC/gC]", fontsize=10)
axS4[3].set_ylabel("Degree of reduction", fontsize=10)
axS4[4].set_ylabel(r"CUE$_{max}$", fontsize=10)
axS4[5].set_ylabel(r"CUE$^{*}$ - CUE", fontsize=10)
axS4[0].legend(fontsize=11, loc="upper right", frameon=False)

for i, label in enumerate(["A", "B", "C", "D", "E", "F"]):
    axS4[i].set_title(f"({label})", fontsize=12, color="black")

figS4.tight_layout()
figS4.savefig(OUTDIR + adapt_flag.replace(" ", "_") + "_FigureS4_necromass.jpg", bbox_inches="tight", dpi=600)

#%%
# -----------------------------------------------------------------------------
# Figure S5a: Necromass fractions
# -----------------------------------------------------------------------------

figS5a, axS5a = plt.subplots(1, 3, figsize=(11, 3), sharex=True, sharey=False)
axS5a = axS5a.flatten()

for ls, mdnam in zip(lstyle, model_name):
    df = scenario_outputs[adapt_flag][mdnam].copy()

    axS5a[0].plot(df["time"] / 365, df["f_nec_Ch"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS5a[1].plot(df["time"] / 365, df["f_nec_P"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS5a[2].plot(df["time"] / 365, df["f_nec_Lig"], label=mdnam, linewidth=2, linestyle=ls, color="black")

for i in range(3):
    axS5a[i].set_xlabel("Time [Y]", fontsize=10)
    axS5a[i].tick_params(axis="both", labelsize=9)
    axS5a[i].grid(False)
    axS5a[i].set_ylim(bottom=0)

axS5a[0].set_ylabel(r"$M_{Ch,nec}/M_{Ch,total}$", fontsize=10)
axS5a[1].set_ylabel(r"$M_{P,nec}/M_{P,total}$", fontsize=10)
axS5a[2].set_ylabel(r"$M_{Lg,nec}/M_{Lg,total}$", fontsize=10)

for i, label in enumerate(["A", "B", "C"]):
    axS5a[i].set_title(f"({label})", fontsize=12, color="black")

axS5a[0].legend(fontsize=11, loc="best", frameon=False)
figS5a.tight_layout()
figS5a.savefig(OUTDIR + adapt_flag.replace(" ", "_") + "_FigureS5a_necromass_fractions.jpg", bbox_inches="tight", dpi=600)

# -----------------------------------------------------------------------------
# Figure S5b: Necromass p_eff and uptake fluxes
# -----------------------------------------------------------------------------

figS5b, axS5b = plt.subplots(1, 3, figsize=(11, 3), sharex=True, sharey=False)
axS5b = axS5b.flatten()

for ls, mdnam in zip(lstyle, model_name):
    df = scenario_outputs[adapt_flag][mdnam].copy()

    axS5b[0].plot(df["time"] / 365, df["p_eff_Ch"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS5b[1].plot(df["time"] / 365, df["p_eff_P"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS5b[2].plot(df["time"] / 365, df["f_U_nec_total"], label=mdnam, linewidth=2, linestyle=ls, color="black")

for i in range(3):
    axS5b[i].set_xlabel("Time [Y]", fontsize=10)
    axS5b[i].tick_params(axis="both", labelsize=9)
    axS5b[i].grid(False)
    axS5b[i].set_ylim(bottom=0)

axS5b[0].set_ylabel(r"$p_{Ch,eff}$", fontsize=10)
axS5b[1].set_ylabel(r"$p_{P,eff}$", fontsize=10)
axS5b[2].set_ylabel(r"$U_{nec}/U_{total}$", fontsize=10)

for i, label in enumerate(["D", "E", "F"]):
    axS5b[i].set_title(f"({label})", fontsize=12, color="black")

figS5b.tight_layout()
figS5b.savefig(OUTDIR + adapt_flag.replace(" ", "_") + "_FigureS5b_necromass_uptake.jpg", bbox_inches="tight", dpi=600)



#%%
# -----------------------------------------------------------------------------
# Figure S6: plant vs necromass protein/carbohydrate pools and uptake fluxes
# -----------------------------------------------------------------------------

figS6, axS6 = plt.subplots(2, 4, figsize=(13, 5), sharex=True, sharey=False)
axS6 = axS6.flatten()

for ls, mdnam in zip(lstyle, model_name):
    df = scenario_outputs[adapt_flag][mdnam].copy()

    axS6[0].plot(df["time"] / 365, df["carbohydrate_plant_gC"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS6[1].plot(df["time"] / 365, df["carbohydrate_nec_gC"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS6[2].plot(df["time"] / 365, df["protein_plant_gC"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS6[3].plot(df["time"] / 365, df["protein_nec_gC"], label=mdnam, linewidth=2, linestyle=ls, color="black")

    axS6[4].plot(df["time"] / 365, df["U_Ch_plant"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS6[5].plot(df["time"] / 365, df["U_Ch_nec"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS6[6].plot(df["time"] / 365, df["U_P_plant"], label=mdnam, linewidth=2, linestyle=ls, color="black")
    axS6[7].plot(df["time"] / 365, df["U_P_nec"], label=mdnam, linewidth=2, linestyle=ls, color="black")

for i in range(8):
    axS6[i].set_xlabel("Time [Y]", fontsize=10)
    axS6[i].tick_params(axis="both", labelsize=9)
    axS6[i].grid(False)
    axS6[i].set_ylim(bottom=0)

axS6[0].set_ylabel(r"$M_{Ch,plant}$ [gC]", fontsize=10)
axS6[1].set_ylabel(r"$M_{Ch,nec}$ [gC]", fontsize=10)
axS6[2].set_ylabel(r"$M_{P,plant}$ [gC]", fontsize=10)
axS6[3].set_ylabel(r"$M_{P,nec}$ [gC]", fontsize=10)
axS6[4].set_ylabel(r"$U_{Ch,plant}$ [gC d$^{-1}$]", fontsize=10)
axS6[5].set_ylabel(r"$U_{Ch,nec}$ [gC d$^{-1}$]", fontsize=10)
axS6[6].set_ylabel(r"$U_{P,plant}$ [gC d$^{-1}$]", fontsize=10)
axS6[7].set_ylabel(r"$U_{P,nec}$ [gC d$^{-1}$]", fontsize=10)

for i, label in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
    axS6[i].set_title(f"({label})", fontsize=12, color="black")

axS6[0].legend(fontsize=11, loc="best", frameon=False)
figS6.tight_layout()
figS6.savefig(OUTDIR + adapt_flag.replace(" ", "_") + "_FigureS6_necromass_pools_fluxes.jpg", bbox_inches="tight", dpi=600)



# ============================================================
# Figure S7. Implied necromass C:N dynamics
# ============================================================
eps = 1e-30
figS7, axS7 = plt.subplots(1, 2, figsize=(7.5, 3.0), sharex=True)
axS7 = axS7.flatten()
for ls, mdnam in zip(lstyle, model_name):
    df = scenario_outputs[adapt_flag][mdnam]

    nec_C_total = df[nec_cols].sum(axis=1)
    nec_N_total = df["protein_nec_gC"] / fixed_param["CNP"]

    CN_nec_total = nec_C_total / (nec_N_total + eps)
    CN_total_model = df["totCg"] / (df["totNg"] + eps)

    axS7[0].plot(
        df["time"] / 365,
        CN_nec_total,
        label=mdnam,
        linewidth=2,
        linestyle=ls,
        color="black",
    )

    axS7[1].plot(
        df["time"] / 365,
        CN_total_model,
        label=mdnam,
        linewidth=2,
        linestyle=ls,
        color="black",
    )

# Format and save necromass C:N diagnostic figure
axS7[0].set_title("(A) Necromass C:N", fontsize=12, color="black")
axS7[1].set_title("(B) Total organic C:N", fontsize=12, color="black")

axS7[0].set_ylabel("C:N [gC gN$^{-1}$]", fontsize=11)
axS7    [1].set_ylabel("C:N [gC gN$^{-1}$]", fontsize=11)

for axx in axS7:
    axx.set_xlabel("Time [Y]", fontsize=11)
    axx.tick_params(axis="both", labelsize=10, direction="out")
    axx.grid(False)

axS7[0].legend(
    fontsize=11,
    loc="best",
    frameon=False,
    labelcolor="black",
)

figS7.tight_layout()
figS7.savefig(OUTDIR + adapt_flag.replace(" ", "_") + "_FigureS7_necromass_CN.jpg", dpi=600, bbox_inches="tight")
#%%

# ===============================================================================================
# Figure SX. Enzyme cost through time for both N strategies, and necromass contribution to uptake
# ===============================================================================================
fig_enz, ax_enz = plt.subplots(2, 2, figsize=(7,7), sharex=False)
# ===============================================================================================
# Figure SX. Enzyme cost through time for both N strategies, and necromass contribution to uptake
# Columns = N strategy; Rows = variable
# ===============================================================================================

adapt_flags = ["N-Retention", "Flexible CUE"]
row_vars = ["enzyme_cost", "f_nec_uptake"]

for col_idx, adapt_flag in enumerate(adapt_flags):

    for ls, mdnam in zip(lstyle, model_name):

        df = scenario_outputs[adapt_flag][mdnam].copy()

        # --------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------
        enzyme_CUE_cost = df["CUE_star"] - df["CUE_cost"]

        if "f_U_nec_total" in df.columns:
            f_nec_uptake = df["f_U_nec_total"]
        elif "f_nec_uptake" in df.columns:
            f_nec_uptake = df["f_nec_uptake"]
        else:
            raise KeyError(
                "Could not find necromass uptake fraction column. "
                "Expected either 'f_U_nec_total' or 'f_nec_uptake'."
            )

        # --------------------------------------------------------
        # Row 0: enzyme CUE cost over time
        # --------------------------------------------------------
        ax_enz[0, col_idx].plot(
            df["time"] / 365,
            enzyme_CUE_cost,
            label=mdnam,
            linewidth=2.0,
            linestyle=ls,
            color="black",
        )

        # --------------------------------------------------------
        # Row 1: necromass contribution to uptake
        # --------------------------------------------------------
        ax_enz[1, col_idx].plot(
            df["time"] / 365,
            f_nec_uptake,
            label=mdnam,
            linewidth=2.0,
            linestyle=ls,
            color="black",
        )
        # ax_enz[2, col_idx].plot(df["time"] / 365, df["f_nec_total"], label=mdnam, linewidth=2.0, linestyle=ls, color="black")

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

col_titles = ["N-retention", "Flexible CUE"]
row_ylabels = [
    "Enzyme cost\nCUE$^*$ - CUE$_{cost}$",
    "Fraction of necromass uptake\n$U_{nec}/U_{tot}$",
    # "Fraction of necromass C\n$f_{nec}= C_{nec}/C_{tot}$",
]
panel_labels = [["(A)", "(B)"], ["(C)", "(D)"]]

for col_idx, title in enumerate(col_titles):
    ax_enz[0, col_idx].set_title(
        f"{panel_labels[0][col_idx]} {title}",
        fontsize=12,
        color="black",
    )

for row_idx in range(2):
    ax_enz[row_idx, 0].set_ylabel(row_ylabels[row_idx], fontsize=11)

for row_idx in range(2):
    for col_idx in range(2):
        ax_enz[row_idx, col_idx].tick_params(axis="both", labelsize=10, direction="out")
        ax_enz[row_idx, col_idx].grid(False)
        ax_enz[row_idx, col_idx].set_xlabel("Time [Y]", fontsize=11)

# Add panel labels to bottom row too
for col_idx in range(2):
    ax_enz[1, col_idx].set_title(
        f"{panel_labels[1][col_idx]}",
        fontsize=12,
        color="black",
    )

# One shared legend
handles, labels = ax_enz[0, 0].get_legend_handles_labels()
fig_enz.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    fontsize=14,
    frameon=False,
)

fig_enz.tight_layout(rect=[0, 0.08, 1, 1])
fig_enz.savefig(
    OUTDIR + "Figure_enzyme_cost_time.jpg",
    dpi=600,
    bbox_inches="tight",
)
# %%

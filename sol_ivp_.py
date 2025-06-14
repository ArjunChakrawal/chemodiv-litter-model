# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:51:34 2024

@author: Arjun Chakrawal (arjun.chakrawal@pnnl.gov)

"""

import seaborn as sns
import numpy as np
from scipy.integrate import solve_ivp
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,pchip_interpolate
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

#%%

def load_fix_par_and_data():
    """
    Loads fixed model parameters and plant data for the chemodiversity litter model.
    Returns:
        fixed_param (dict): Dictionary containing fixed model parameters such as Q10 values, activation energies, 
                            stoichiometric ratios, and other constants.
        plant_data (pd.DataFrame): DataFrame containing processed plant data
    """
    # Q10 values taken from Allison et al. 2018 GCB https://onlinelibrary.wiley.com/doi/abs/10.1111/gcb.14045 
    Q10 = {'Ch':1.6,'P':2.25,'Lig':1.65,'Lip':1.65,'Cr':1.6} # Lipid is assumed to be same as lignin, carbonyl same as carbohydrate  

    # Ea = R log(Q10) T1 T2/(T1-T2)
    Ea_dict = {}
    T=273+15
    for item, q10_value in Q10.items():
        Ea_dict[item] =  0*8.314* np.log(q10_value)* T*(T+10)/10 # J/mol # EAs are set to zero in fixed parameter
        
    # CNP is fixed from prorien formula assumed in molecular mixing model
    fixed_param = {'CNB': 16, 'Inorg': 1e-9, 'a': 0.28, 'b': 2, 'nosc': np.array([0, 0.034, -0.381, -1.471, 3]).reshape((1, 5)),
                   'CNP': 1/(0.27*14/12), 'mLg': 0.1,  'mLp': 0.4, 'mCr': 0.05, 'Ea':Ea_dict}

    fixed_param['mP'] = fixed_param['CNP']/fixed_param['CNB']
    fixed_param['mC'] = 1 - fixed_param['mP'] - fixed_param['mLp']-fixed_param['mLg']-fixed_param['mCr']

    
    plant_data = pd.read_excel('../data/processed_data.xlsx')
    plant_data = plant_data[~plant_data['Study'].isin(['Xu et al 2017 SBB'])].reset_index(drop=True)
    plant_data = plant_data[~(plant_data['R-squared'] < 0)].reset_index(drop=True)
    rows_to_delete = (plant_data['Study'] == 'Preston et al. 2009') & (plant_data['time day'] == 730)
    # protien calculation at 730 for preston is not correct so setting them nan
    plant_data.loc[rows_to_delete, ['protein_gC', 'protein_gN']] = np.nan
    
   # For 'protein_gC' column
    mask = (plant_data['protein_gC'] < 1e-6) & (plant_data['time day'] != 0)
    plant_data.loc[mask, 'protein_gC'] = np.nan
    
    # For 'carbonyl_gC' column
    mask = (plant_data['carbonyl_gC'] < 1e-6) & (plant_data['time day'] != 0)
    plant_data.loc[mask, 'carbonyl_gC'] = np.nan
    
    return fixed_param,plant_data


def efficiency(gamma):
    """
    Calculate the carbon use efficiency (CUE) as a function of the degree of reduction (gamma).
    Parameters
    ----------
    gamma : float or array-like
        Degree of reduction of the substrate.
    Returns
    -------
    Y : float or ndarray
        Carbon use efficiency (CUE) corresponding to the input gamma.
    Notes
    -----
    The function computes CUE based on thermodynamic properties and the degree of reduction.
    """
    gamma_O2 = 4
    dfGH2O = -237.2  # kJ/mol
    dfGO2aq = 16.5  # kJ/mol
    dGred_O2 = 2 * dfGH2O - dfGO2aq

    def dGox_func(gamma):
        return 60.3 - 28.5 * (4 - gamma)

    def dCG_O2(gamma):
        return dGox_func(gamma) + (gamma / gamma_O2) * dGred_O2

    # Ts = 298
    # pre-assign std G of formation from CHNOSZ
    gamma_B = 4.2  # e- mol/Cmol
    # define electron acceptor and some other DR
    dGred_eA = 2 * dfGH2O - dfGO2aq

    # growth yield calculations
    dGrX = np.where(gamma < 4.67, -(666.7 / gamma + 243.1), -
                    (157 * gamma - 339))  # kJ/Cmol biomass

    # Anabolic reaction
    dCGX = dCG_O2(gamma_B)
    dGana = (gamma_B / gamma) * dCG_O2(gamma) - dCGX
    dGana1 = dGana

    dG_ox = dGox_func(gamma)
    dGcat = dG_ox + gamma * dGred_eA / gamma_O2
    Y = dGcat / (dGrX - dGana1 + gamma_B / gamma * dGcat)
    if len(Y) == 1:
        Y = Y.item(0)
    return Y

def litter_decay_model(tsim,init_fracC, guess_param_val, fixed_param,adapt_flag, protection, CUEflag, voflag):
    """
    ### Litter Decay Model: Dynamic Carbon and Nitrogen Pool Simulation
    
    Simulates the decomposition of organic litter through carbon and nitrogen pool transformations over time. 
    The model implements enzymatic processes and microbial activity based on various parameters and flags.

    Parameters
    ----------
    - `tsim` (array-like): Simulation time vector (days).
    - `init_fracC` (array-like): Initial fractions of Carbon pools [Carbohydrates, Proteins, Lignins, Lipids, Carbonyls].
    - `guess_param_val` (array-like): Maximum rates for enzymatic reactions per pool [vh_max, vp_max, vlig_max, vlip_max, vCr_max].
    - `fixed_param` (dict): Fixed parameters including constants (e.g., `a`, `b`, activation energy, CN ratios).
    - `adapt_flag` (str): Microbial adaptation strategy (`Flexible CUE` or `N-Retention`).
    - `protection` (bool): Determines whether protection mechanisms against enzymatic decay are active.
    - `CUEflag` (bool): Whether carbon use efficiency is spatially-modulated.
    - `voflag` (bool): Enables labilization of lignin pools.
    
    #### Returns
    - `df_ivp` (DataFrame): A pandas DataFrame containing time evolution of each pool and calculated metrics:
        - `carbohydrate_gC`: Carbohydrate pool (gC).
        - `protein_gC`: Protein pool (gC).
        - `lignin_gC`: Lignin pool (gC).
        - `lipid_gC`: Lipid pool (gC).
        - `carbonyl_gC`: Carbonyl group pool (gC).
        - `CO2_gC`: CO2 produced (gC).
        - `totCg`: Total carbon in pools (gC).
        - `totNg`: Total nitrogen in pools (gN).
        - `CUE`: Carbon use efficiency.
        - `ETA`: Retention efficiency.
        - `MNet [gN/day]`: Net nitrogen mineralization rate.
        - `Growth rate [gC/day]`: Microbial growth rate.
        - `DR`: Degree of reduction.
        - `sumPool`: Sum of all pools (mass balance check).

    Mass Balance Equations
    ----------------------
    The system solves rates of change for each pool (dpool/dt) based on microbial activity, enzymatic decay, and assimilation.

    Carbon Balance:
        dCarbohydrate/dt = mC * G - DC
        dProtein/dt      = (1 - eta) * mP * G - DP
        dLignin/dt       = mLg * G - DLig
        dLipid/dt        = mLp * G - DLip
        dCarbonyl/dt     = mCr * G - DCr

    CO2 Evolution:
        dCO2/dt = (DC + DP + DLig + DLip + DCr) - G

    Growth Rate:
        G = CUE * (DC + DP + DLig + DLip + DCr)

    Nitrogen Balance:
        MNet = (DP / CNP) - (G * C / CNB)

    Internal Functions
    ------------------
    derivatives : Computes instantaneous rates of change for all pools and fluxes.
    odefun      : Prepares system for numerical integration using solve_ivp.
    
    """    
    
    def derivatives(t, state, guess_param_val, fixed_param,adapt_flag, protection, CUEflag, voflag):
        fC, fP, fLg, fLp, fCr, fCO2 = state
        T =273
        vh_max, vp_max, vlig_max, vlip_max, vCr_max = guess_param_val
        a, b = fixed_param['a'], fixed_param['b']    
        EaCh, EaP, EaLig, EaLip, EaCr = fixed_param['Ea'].values()  
        R = 8.314 # J/mol K
        if protection:
            def vH(l, vh_max,T): return vh_max*np.exp(-(l / a)**b)*np.exp(-EaCh/(R*T))
            def vP(l, vp_max,T): return vp_max*np.exp(-(l / a)**b)*np.exp(-EaP/(R*T))
        else:
            def vH(l, vh_max,T): return vh_max*np.exp(-EaCh/(R*T))
            def vP(l, vp_max,T): return vp_max*np.exp(-EaP/(R*T))
            
        if voflag:
            def vlig(l, vlig_max,T): return vlig_max*(1-np.exp(-(l / a)**b))*np.exp(-EaLig/(R*T))
        else:
            def vlig(l, vlig_max,T): return vlig_max*np.exp(-EaLig/(R*T))
        
        def vlip(T): return vlip_max*np.exp(-EaLip/(R*T))
        def vCr(T): return vCr_max*np.exp(-EaCr/(R*T))
    
    
        def CUE_func(fC, fP, fLg, fLp, fCr):
            totC = fC + fP + fLg + fLp + fCr
            L = fLg / totC
            frac = np.array([fC, fP, fLg, fLp, fCr])
            frac = frac/np.sum(frac)
            litDR = 4-np.dot(fixed_param['nosc'], frac)
            if CUEflag:
                return efficiency(litDR)*np.exp(-(L / a)**b)
            else:
                return efficiency(litDR)
            
        def eta_func(T,fC, fP, fLg, fLp, fCr):
            totC = fC + fP + fLg + fLp + fCr
            L = fLg / totC
            DC = vH(L, vh_max,T) * fC
            DP = vP(L, vp_max,T) * fP
            DLig = vlig(L, vlig_max,T) * fLg
            DLip = vlip(T) * fLp
            DCr = vCr(T) * fCr
            feff=CUE_func(fC, fP, fLg, fLp, fCr)
            GC = feff * (DC + DP + DLig+DLip+DCr) # growth rate under C limitation
            eta = 1 - (fixed_param['CNB']/GC)*(DP/fixed_param['CNP'] + fixed_param['Inorg'])
            return eta
        
        totC = fC + fP + fLg + fLp + fCr
        L = fLg / totC
        DC = vH(L, vh_max,T) * fC
        DP = vP(L, vp_max,T) * fP
        DLig = vlig(L, vlig_max,T) * fLg
        DLip = vlip(T) * fLp
        DCr = vCr(T) * fCr
        CUE=CUE_func(fC, fP, fLg, fLp, fCr)
        GC = CUE * (DC + DP + DLig+DLip+DCr) # growth rate under C limitation
        Mnet = DP / fixed_param['CNP']- GC / fixed_param['CNB']
        f_eta=-999
        G=-999
        if adapt_flag =="Flexible CUE":
            f_eta=0
            if Mnet > 0:
                CUE=CUE
                Mnet = Mnet
                G = GC # growth rate under C limitation
            else:
                if -Mnet <  fixed_param['Inorg']:
                    CUE=CUE
                    Mnet = Mnet
                    G = GC # growth rate under C limitation
                else:
                    CUE=fixed_param['CNB'] * (fixed_param['Inorg'] + DP / fixed_param['CNP']
                                                   )/(DC + DP + DLig + DLip + DCr)
                    Mnet = -fixed_param['Inorg']
                    G = (fixed_param['CNB']/(1-f_eta))*(DP / fixed_param['CNP'] +fixed_param['Inorg']) # growth rate under N limitation
        elif adapt_flag =="N-Retention":
            if Mnet > 0:
                f_eta = 0
                Mnet = Mnet
                G = GC # growth rate under C limitation
            else:
                if -Mnet <  fixed_param['Inorg']:
                    f_eta=0
                    Mnet = Mnet
                    G = GC # growth rate under C limitation
                else:
                    f_eta = eta_func(T,fC, fP, fLg, fLp, fCr)
                    Mnet = -fixed_param['Inorg']
                    G = (fixed_param['CNB']/(1-f_eta))*(DP / fixed_param['CNP'] +fixed_param['Inorg']) # growth rate under N limitation
    
    
        mC = 1 - (1-f_eta)*fixed_param['mP'] - fixed_param['mLp']-fixed_param['mLg']-fixed_param['mCr']
        
        dCarbdt = mC * G - DC
        dProtdt = (1-f_eta)*fixed_param['mP']  * G - DP
        dLigdt = fixed_param['mLg'] * G - DLig
        dLipdt = fixed_param['mLp'] * G - DLip
        dCarbonyldt = fixed_param['mCr'] * G - DCr
        dCO2dt = (DC + DP + DLig+DLip+DCr) -G
        out1 = [dCarbdt, dProtdt, dLigdt, dLipdt, dCarbonyldt, dCO2dt]
        
        totCg =  fC + fP + fLg + fLp + fCr
        totNg = fP/fixed_param['CNP']
        frac = np.array([fC, fP, fLg, fLp, fCr])
        frac = frac/np.sum(frac)
        DR = 4-np.dot(fixed_param['nosc'], frac)
        s = fC + fP + fLg + fLp + fCr+ fCO2
        out2 = [totCg, totNg, CUE,f_eta, Mnet, G, DR[0], s]
        return out1, out2


    def odefun(t, state, guess_param_val, fixed_param, adapt_flag,protection, CUEflag, voflag):
        out1,_ = derivatives(t, state, guess_param_val, fixed_param, adapt_flag,protection, CUEflag, voflag)
        return out1
    
    init_fracC = np.hstack((init_fracC, 0))
    sol = solve_ivp(odefun, (tsim[0], tsim[-1]), init_fracC, 
                    args=(guess_param_val, fixed_param,adapt_flag,protection, CUEflag, voflag), 
                    method='RK45',dense_output=True, rtol=1e-12, atol=1e-12)
    z = sol.sol(tsim).T
    
    out = np.hstack((tsim.reshape(len(tsim), 1), z))
    df_ivp = pd.DataFrame(out, columns=["time", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC','carbonyl_gC', "CO2_gC"])

    temp = np.empty((0, 8))
    for i in range(len(df_ivp)):
        state = df_ivp.loc[i, ['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC','carbonyl_gC',"CO2_gC"]].values
        out1, out2 = derivatives(tsim, state, guess_param_val, fixed_param, adapt_flag, protection, CUEflag, voflag)
        out = np.array(out2).reshape(1, len(out2))
        temp = np.vstack((temp, out))


    tempdf = pd.DataFrame(temp, columns=['totCg', 'totNg', 'CUE','ETA', 'MNet [gN/day]', 'Growth rate [gC/day]', 'DR', 'sumPool'])
    df_ivp = pd.concat((df_ivp, tempdf), axis=1)
    
    return df_ivp


def residual_fun(x,data, data_col,tsim,init_fracC, fixed_param,adapt_flag, protection, CUEflag, voflag):
    """
    Computes the residuals between simulated and observed litter decay mass loss.
    Parameters:
        x (array-like): Model parameters to optimize.
        data (pd.DataFrame): Observed data containing time and measured columns.
        data_col (list of str): Column names in data to compare.
        tsim (array-like): Time points for simulation.
        init_fracC (array-like): Initial fraction of carbon pools.
        fixed_param (dict): Fixed model parameters.
        adapt_flag (bool): Flag to enable/disable N adaptation strategy in the model.
        protection (bool): Flag to enable/disable lignin protection mechanism.
        CUEflag (bool): Flag to enable/disable variation in Carbon Use Efficiency as a function of lignin fraction.
        voflag (bool): Flag to enable/disable variation in lignin decay rate constant as a function of lignin fraction.
    Returns:
        np.ndarray: Residuals between normalized simulated and observed data, with NaNs removed.
    """
    df = litter_decay_model(tsim,init_fracC, x, fixed_param,adapt_flag, protection, CUEflag, voflag)
    S_cols = df[data_col]
    splines = {col: interp1d(tsim, S_cols[col], kind='linear', fill_value='extrapolate') for col in data_col}

    # simC = np.array([splines[col](data['time day']) / splines[col](tsim)[0] if splines[col](tsim)[0] != 0
    #                  else splines[col](data['time day']) / splines[col](tsim)[1] for col in data_col])
    simC = np.array([splines[col](data['time day']) / data[col].max() for col in data_col])
    # obs = np.array([data[col] / data[col].iloc[0] if data[col].iloc[0] != 0
    #                 else data[col] / data[col].iloc[1] for col in data_col])

    obs = np.array([data[col] / data[col].max() for col in data_col])
    res = simC.flatten() - obs.flatten()
    res_without_nan = res[~np.isnan(res)]
    return res_without_nan

def fit_data(guess_param,  init_fracC, fixed_param, tsim,Temperature, data, data_col,adapt_flag,protection, CUEflag,voflag, loss='soft_l1'):
    """
    Fit model parameters to data using non-linear least squares optimization.
    Parameters:
        guess_param (dict): Initial guesses for parameters to be estimated.
        init_fracC (array-like): Initial fraction of carbon pools.
        fixed_param (dict): Parameters to be held constant during fitting.
        tsim (array-like): Time points for simulation.
        Temperature (float or array-like): Temperature(s) for the model. NOT used.
        data (array-like): Observed data to fit.
        data_col (list): Columns of data to use for fitting.
        adapt_flag (bool): Flag to enable/disable N adaptation strategy in the model.
        protection (bool): Flag to enable/disable lignin protection mechanism.
        CUEflag (bool): Flag to enable/disable variation in Carbon Use Efficiency as a function of lignin fraction.
        voflag (bool): Flag to enable/disable variation in lignin decay rate constant as a function of lignin fraction.
        loss (str, optional): Loss function for least squares ('soft_l1' by default).
    Returns:
        est_pars (dict): Estimated parameter values.
        est_pars_se (dict): Standard errors of the estimated parameters.
    """
    print("fitting in progress...")
    inital_guess = np.array(list(guess_param.values()))
    lb, ub = np.ones(len(inital_guess))*1e-5, np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    # start_time = time.time()
    res_lsq = least_squares(residual_fun, inital_guess, loss='soft_l1', f_scale=0.5, bounds=(lb, ub),
                            args=(data, data_col, tsim, init_fracC, fixed_param, adapt_flag, protection, CUEflag, voflag),
                            verbose=0)
    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time:.6f} seconds") 
    
    # estiamte parameter uncertainty
    cov_matrix = np.linalg.inv(res_lsq.jac.T @ res_lsq.jac)
    # Extract the standard errors (square root of the diagonal elements of the covariance matrix)
    parameter_uncertainties = np.sqrt(np.diag(cov_matrix))
    est_par_name = list(guess_param.keys())
    est_pars = {est_par_name[i]: res_lsq.x[i] for i in range(len(res_lsq.x))}
    est_pars_se = {est_par_name[i]+"_se": parameter_uncertainties[i] for i in range(len(parameter_uncertainties))}
    print("fitting completed...")
    return est_pars,est_pars_se

def cal_perf_matrix(est_pars,init_fracC, fixed_param, tsim,Temperature, data, data_col, adapt_flag,protection,CUEflag,voflag):
    df = litter_decay_model(tsim,init_fracC, est_pars, fixed_param,adapt_flag, protection, CUEflag, voflag)
    S_cols = df[data_col]

    perf_matrix = pd.DataFrame(index=['r2', 'rmse','AIC'], columns=["overall", "totCg", "totNg", 'carbohydrate_gC',
                                                              'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC'])
    splines = {col: interp1d(tsim, S_cols[col], kind='linear', fill_value='extrapolate') for col in data_col}
    
    # AICfun = lambda n, p, mse: n * np.log(mse) + 2 * n * p / (n - p - 1) # n= # observation, p= # parameters, mse= mean squared error
    AICfun = lambda n, p, mse: n * np.log(mse) + 2 *p # n= # observation, p= # parameters, mse= mean squared error

    for col in data_col:
        # y_pred = splines[col](data['time day'])/ data[col].max()
        # y_true = data[col]/ data[col].max()
        y_pred = splines[col](data['time day'])
        y_true = data[col]

        valid_indices = ~np.isnan(y_true)
        # Check if all valid_indices are False
        if not np.any(valid_indices):
            # If all valid_indices are False, y_true is empty or contains only NaN values
            perf_matrix.loc[['r2','rmse','AIC'], col] = np.nan
        else:
            # Otherwise, there are valid values in y_true
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            perf_matrix.loc['r2', col] = r2_score(y_true, y_pred)
            perf_matrix.loc['rmse', col] = np.sqrt(mean_squared_error(y_true, y_pred))
            # perf_matrix.loc['AIC', col] = AICfun(len(y_true), len(est_pars),mean_squared_error(y_true, y_pred))

    y_pred = np.array([splines[col](data['time day']) for col in data_col]).flatten()
    y_true = np.array([data[col] for col in data_col]).flatten()
    valid_indices = ~np.isnan(y_true)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]
    perf_matrix.loc['r2', 'overall'] = r2_score(y_true, y_pred)
    perf_matrix.loc['rmse', 'overall'] = np.sqrt(mean_squared_error(y_true, y_pred))
    # perf_matrix.loc['AIC', 'overall'] = AICfun(len(y_true), len(est_pars),mean_squared_error(y_true, y_pred))

    temp_col = data_col.copy()
    if 'DR' not in temp_col:
        temp_col.append('DR')
    S_cols = df[temp_col]
    splines = {col: interp1d(tsim, S_cols[col], kind='linear', fill_value='extrapolate') for col in temp_col}
    y_pred1 = np.array([splines[col](data['time day']) for col in temp_col]).flatten()
    y_true1 = np.array([data[col] for col in temp_col]).flatten()
    cat = [np.repeat(col, len(data['time day'])) for col in temp_col]
    cat = np.array(cat).flatten().tolist()
    
    obstime = np.tile(data['time day'].values, len(temp_col))
    model_data = pd.DataFrame({"time":obstime,"pool":cat,'obs':y_true1,'sim':y_pred1})
    # plt.figure, plt.plot(model_data['sim'],model_data['obs'],'o'), plt.plot()
    return perf_matrix, model_data
   
    



def plot_model(tsim, fixed_param, est_pars, init_fracC, data_col=None, data=None,adapt_flag = 'N-Retention',
                protection=True, CUEflag=True,voflag=True):

    Temperature = np.ones(len(tsim))*273.15
    vh_max,vp_max,vlig,vlip, vCr = est_pars
    df = litter_decay_model(tsim,init_fracC, est_pars, fixed_param,adapt_flag, protection, CUEflag, voflag)
    a, b = fixed_param['a'], fixed_param['b']
    col = ['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']

    plt.style.use('ggplot')
    palette = sns.color_palette()
    plt.style.use('default')

    fig, ax = plt.subplots(4, 4, figsize=(14, 8))
    ax = ax.flatten()

    # Plotting on axes
    ax[0].plot(df['time'], df["totCg"], linewidth=1.5)
    if data is not None:
        ax[0].scatter(data['time day'], data["totCg"], linewidth=1.5)
    ax[0].set_xlabel('Time [d]')
    ax[0].set_ylabel('total C [g]')

    ax[1].plot(df['time'], df["totNg"], linewidth=1.5, label="model")
    if data is not None:
        ax[1].scatter(data['time day'], data["totNg"], linewidth=1.5, label="Ng")
        ax[1].scatter(data['time day'], data["protein_gN"], linewidth=1.5, label="protein_gN")
    ax[1].set_xlabel('Time [d]')
    ax[1].set_ylabel('total N [g]')
    ax[1].legend(fontsize=8, loc = "best",frameon=False)
    
    # Loop over columns and plot data for the first set of axes
    for i, column in enumerate(col):
        color = palette[i]
        ax[2].plot(df['time'], df[column]/df[column].iloc[0], label=column, linewidth=1.5, color=color)
        if data is not None:
            ax[2].scatter(data['time day'], data[column]/data[column].iloc[0], linewidth=1.5, color=color)

    ax[2].set_xlabel('Time [d]')
    ax[2].set_ylabel('gC/gC[0]')

    # Loop over columns and plot data for the second set of axes
    for i, column in enumerate(col):
        color = palette[i]
        ax[3].plot(df['time'], df[column], label=column, linewidth=1.5, color=color)
        if data is not None:
            ax[3].scatter(data['time day'], data[column], linewidth=1.5, color=color)

    ax[3].set_xlabel('Time [d]')
    ax[3].set_ylabel('gC')
    ax[3].legend(fontsize=8,  loc = "best",frameon=False)

    # Continue plotting for the third set of axes
    ax[4].plot(df['time'], df['DR'], linewidth=1.5)
    if data is not None:
        ax[4].scatter(data['time day'], data['DR'], linewidth=1.5, color=color)
    ax[4].set_xlabel('Time [d]')
    ax[4].set_ylabel('DR')

    # Continue plotting for the fourth set of axes
    ax[5].plot(df['time'], df["MNet [gN/day]"], linewidth=1.5)
    ax[5].set_xlabel('Time [d]')
    ax[5].set_ylabel("MNet [gN/day]")

    # Continue plotting for the fifth set of axes
    ax[6].plot(df['time'], df['CUE'], linewidth=1.5)
    ax[6].set_xlabel('Time [d]')
    ax[6].set_ylabel('CUE')

    # Continue plotting for the sixth set of axes
    ax[7].plot(df['time'], df["Growth rate [gC/day]"], linewidth=1.5)
    ax[7].set_xlabel('Time [d]')
    ax[7].set_ylabel("Growth rate [gC/day]")

    # Continue plotting for the seventh set of axes
    L = df["lignin_gC"]/df[col].sum(axis=1)
    ax[8].plot(df['time'], L, linewidth=1.5)
    ax[8].set_xlabel('Time [d]')
    ax[8].set_ylabel("Lignin fraction")
    

    # Continue plotting for the eighth set of axes (if protection is True)
    if protection:
        Ch_protection = vh_max*np.exp(-(L / a)**b)
        P_protection = vp_max*np.exp(-(L / a)**b)
        lig_protection = vlig*(1-np.exp(-(L / a)**b))

        # pfunc = np.exp(-(np.arange(0, 0.8, 0.01) / a)**b)
        # ax[9].plot(np.arange(0, 0.8, 0.01), pfunc, linewidth=1.5, label="protection function")
        ax[9].plot(L, Ch_protection, linewidth=3, label="Carb")
        ax[9].plot(L, P_protection, linewidth=3, label="Prot")
        if voflag:
            ax[9].plot(L, lig_protection, linewidth=3, label="Lignin")
        ax[9].set_xlabel('Lignin fraction')
        ax[9].set_ylabel("protection func.")
        ax[9].legend()
        
        ax[10].plot(np.arange(0, 0.8, 0.01),  np.exp(-(np.arange(0, 0.8, 0.01) / a)**b), linewidth=1.5, label="v_C, v_P modifier")
        ax[10].plot(L, Ch_protection/vh_max,"-", linewidth=4, label="Carb", color='grey')
        if voflag:
            ax[10].plot(np.arange(0, 0.8, 0.01),  1-np.exp(-(np.arange(0, 0.8, 0.01) / a)**b), linewidth=1.5, label="v_Lg modifier")
        ax[10].plot(L, lig_protection/vlig,"-",  linewidth=4, label="Lignin", color='grey')
        ax[10].set_xlabel('Lignin fraction'), ax[10].legend(fontsize=9)



    ax[11].plot(df['sumPool'], linewidth=1.5, label="protection function")
    ax[11].set_ylabel("mass balance check")

    ax[12].plot(df['time'], df['totCg']/df['totNg'], linewidth=1.5)
    ax[12].set_xlabel('Time [d]')
    ax[12].set_ylabel(r"$CN$")
    
    ax[13].plot(df['totCg']/df['totNg'], df['ETA'], linewidth=1.5)
    ax[13].set_xlabel(r'$CN$')
    ax[13].set_ylabel(r"$\eta$")
    
    ax[14].plot(df['time'], df['ETA'], linewidth=1.5)
    ax[14].set_xlabel('Time [d]')
    ax[14].set_ylabel(r"$\eta$")
    
    ax[15].plot(df['time'], (1-df['ETA'])*fixed_param['mP'], linewidth=1.5)
    ax[15].set_xlabel('Time [d]')
    ax[15].set_ylabel("P turnover rate costant")
    
    for axs in ax:
        axs.grid(True, color='grey', linestyle='-', linewidth=0.25)
    
    
    # Add suptitle if perf_matrix is not None
    if data is not None:
        perf_matrix,_ = cal_perf_matrix(est_pars,init_fracC, fixed_param, tsim,Temperature, data, data_col,
                                        adapt_flag, protection,CUEflag,voflag)
        
        suptitle_str = data['Study'][0]+"_"+data["Species"][0] + \
            f" [Overall rmse: {perf_matrix.loc['rmse', 'overall']:.2E}, Overall r2: {perf_matrix.loc['r2', 'overall']:.2E}]"
        # for col_name in perf_matrix.columns[1:]:
        #     suptitle_str += f"{col_name}: rmse = {perf_matrix.loc['rmse',col_name]:.2f}, r2 = {perf_matrix.loc['r2',col_name]:.2f}   "
        plt.suptitle(suptitle_str, fontsize=12, va='top')
    plt.tight_layout()
    plt.show()
    return fig, df



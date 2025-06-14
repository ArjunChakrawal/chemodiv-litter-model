# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:20:41 2024

@author: chak803
"""


# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.interpolate import interp1d,pchip_interpolate
# import time
from scipy.optimize import least_squares, minimize
from sklearn.metrics import r2_score, mean_squared_error
# import itertools
# from tqdm import tqdm

def molecular_mixing_model(NMR_data):
    NCobs = (12 / 14) / NMR_data['C:N']  # Molar
    NMR = np.array([
        NMR_data['A ALKYL 0–45 ppm'], NMR_data['B METHOX 45–60 ppm'], NMR_data['C O-ALKYL 60-95 ppm'],
        NMR_data['D DI-O-ALK 95–110ppm'], NMR_data['E AROM 110–145 ppm'], NMR_data['F PHEN 145–165 ppm'],
        NMR_data['G CARBOX 165-210 ppm']
    ])
    NMR = NMR[~np.isnan(NMR)]
    spectra = [
        NMR_data['a'], NMR_data['b'], NMR_data['c'], NMR_data['d'],
        NMR_data['e'], NMR_data['f'], NMR_data['g']
    ]
    avail_spectra = [item for item in spectra if not (isinstance(
        item, float) if not isinstance(item, np.floating) else np.isnan(item))]
    avail_spectra = [item for item in avail_spectra if item.strip()]
    # Assignment below is from Table 1 of Nelson and Baldock
    # Carbohydrate  Protein  Lignin  Lipid  Carbonyl  Char
    Mixing_terrestrial = np.array([
        [0, 39.6, 10.5, 75.6, 0, 0],    # 0-45
        [4.3, 21.9, 13.8, 4.5, 0, 0],    # 45-60
        [79.0, 2.1, 12.5, 9, 0, 0],    # 60-95
        [15.7, 0, 8.6, 0, 0, 4.3],    # 95-110
        [1, 7.5, 30.6, 3.6, 0, 73.9],    # 110-145
        [0, 2.5, 19.5, 0.7, 0, 16.1],    # 145-165
        [0, 26.4, 4.6, 6.6, 100, 5.6],    # 165-215
    ]) / 100

    # Mixing_terrestrial_df= pd.DataFrame(Mixing_terrestrial, index=['A ALKYL 0–45 ppm', 'B METHOX 45–60 ppm', 'C O-ALKYL 60-95 ppm',
    # 'D DI-O-ALK 95–110ppm', 'E AROM 110–145 ppm', 'F PHEN 145–165 ppm','G CARBOX 165-210 ppm'], columns=['Carbohydrate',
    #                                                                     'Protein' , 'Lignin',  'Lipid'  ,'Carbonyl',  'Char'])
    CNHO_terrestrial = np.array([
        [1, 1, 1, 1, 1, 1],
        [0, 0.27, 0, 0, 0, 0],
        [1.67, 1.10, 1.24, 1.94, 1, 0.45],
        [0.83, 0.16, 0.43, 0.24, 2, 0.41],
    ])
    Assignment = Mixing_terrestrial[:, :5]
    CNHO_Assignment = CNHO_terrestrial[:, :5]

    # Calculate new_mat
    str_list = ["a", "b", "c", "d", "e", "f", "g"]
    new_mat = np.zeros((len(avail_spectra), Assignment.shape[1]))
    idx = -1
    for i in range(len(avail_spectra)):
        # for i in range(0,4):
        temp = avail_spectra[i]
        idx += len(temp)
        if len(temp) == 1:
            new_mat[i, :] = Assignment[idx, :]
        else:
            id_list = [str_list.index(temp[j]) for j in range(len(temp))]
            new_mat[i, :] = np.sum(Assignment[id_list, :], axis=0)

    # Define constraints Aeq and beq
    prot_fraction = np.nan
    if np.isnan(NCobs):
        Aeq = np.ones((1, Assignment.shape[1]))  # sum of fractions = 1
        beq = 1
    else:
        prot_fraction = NCobs/0.27
        new_mat = np.delete(new_mat, 1, axis=1)
        Aeq = np.ones((1, new_mat.shape[1]))
        beq = np.array([1])-prot_fraction

    # Define optimization options
    options = {'disp': False}

    # Define the objective function
    def fun(x):
        return np.sqrt(np.mean((NMR - np.dot(new_mat, x))**2))

    # Define lower and upper bounds for x
    lb = np.zeros(new_mat.shape[1])
    ub = np.ones(new_mat.shape[1])

    # Initial guess for x
    guess_xopt = np.ones(new_mat.shape[1]) * 0.01

    # Solve the optimization problem
    result = minimize(fun, guess_xopt, method='SLSQP', bounds=list(zip(lb, ub)),
                      constraints={'type': 'eq',
                                   'fun': lambda x: Aeq @ x - beq},
                      options=options)
    x = result.x

    if not np.isnan(prot_fraction):
        frac = np.concatenate(([x[0]], [prot_fraction], x[1:]))
    else:
        frac = x

    CNHO = pd.DataFrame((CNHO_Assignment @ frac).reshape(1, -1),
                        columns=['C', 'N', 'H', 'O'])
    CNHO['molecularFormula'] = CNHO.apply(
        lambda row: f"C{row['C']:.2f}H{row['H']:.2f}N{row['N']:.2f}O{row['O']:.2f}", axis=1)
    CNHO['Cox'] = 4 - (4 * CNHO['C'] + CNHO['H'] -
                       2 * CNHO['O'] - 3 * CNHO['N'])
    r_squared = r2_score(NMR, np.dot(new_mat, x))
    rmse = np.sqrt(mean_squared_error(NMR, np.dot(new_mat, x)))
    return frac, CNHO, np.dot(new_mat, x), r_squared, rmse



# def CUE_adaptation(guess_param_val, init_fracC, fixed_param, tsim,Temperature, protection, CUEflag,voflag):

#     vh_max, vp_max, vlig_max, vlip_max, vCr_max = guess_param_val
    
#     a, b = fixed_param['a'], fixed_param['b']
    
#     EaCh, EaP, EaLig, EaLip, EaCr = fixed_param['Ea'].values()
#     R = 8.314 # J/mol K
    
    
#     if protection:
#         def vH(l, vh_max,T): return vh_max*np.exp(-(l / a)**b)*np.exp(-EaLip/(R*T))
#         def vP(l, vp_max,T): return vp_max*np.exp(-(l / a)**b)*np.exp(-EaLip/(R*T))
#     else:
#         def vH(l, vh_max,T): return vh_max*np.exp(-EaLip/(R*T))
#         def vP(l, vp_max,T): return vp_max*np.exp(-EaLip/(R*T))
        
#     if voflag:
#         def vlig(l, vlig_max,T): return vlig_max*(1-np.exp(-(l / a)**b))*np.exp(-EaLip/(R*T))
#     else:
#         def vlig(l, vlig_max,T): return vlig_max*np.exp(-EaLip/(R*T))
    
#     def vlip(T): return vlip_max*np.exp(-EaLip/(R*T))
#     def vCr(T): return vCr_max*np.exp(-EaCr/(R*T))
    
    
#     def CUE_func(fC, fP, fLg, fLp, fCr):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         frac = np.array([fC, fP, fLg, fLp, fCr])
#         frac = frac/np.sum(frac)
#         litDR = 4-np.dot(fixed_param['nosc'], frac)
#         if CUEflag:
#             return efficiency(litDR)*np.exp(-(L / a)**b)
#         else:
#             return efficiency(litDR)

#     def Ucarb(T, fC, fP, fLg, fLp, fCr, feff):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         DC = vH(L, vh_max,T) * fC
#         G = Growth(fC, fP, fLg, fLp, fCr, feff)
#         return fixed_param['mC'] * G - DC

#     def Uprot(T, fC, fP, fLg, fLp, fCr, feff):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         DP = vP(L, vp_max,T) * fP
#         G = Growth(T, fC, fP, fLg, fLp, fCr, feff)
#         return fixed_param['mP'] * G - DP

#     def Ulignin(T, fC, fP, fLg, fLp, fCr, feff):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         DLig = vlig(L, vlig_max,T) * fLg
#         G = Growth(T, fC, fP, fLg, fLp, fCr, feff)
#         return fixed_param['mLg'] * G - DLig

#     def Ulipid(T, fC, fP, fLg, fLp, fCr, feff):
#         Dlip = vlip(T) *fLp
#         G = Growth(T, fC, fP, fLg, fLp, fCr, feff)
#         return fixed_param['mLp'] * G - Dlip

#     def UCarbonyl(T, fC, fP, fLg, fLp, fCr, feff):
#         DCr = vCr(T) * fCr
#         G = Growth(T, fC, fP, fLg, fLp, fCr, feff)
#         return fixed_param['mCr']* G - DCr

#     def f_co2(T, fC, fP, fLg, fLp, fCr, feff):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         DC = vH(L, vh_max,T) * fC
#         DP = vP(L, vp_max,T) * fP
#         DLig = vlig(L, vlig_max,T) * fLg
#         Dlip = vlip(T) * fLp
#         DCr = vCr(T) * fCr
#         return (1 - feff) * (DC + DP + DLig+Dlip+DCr)

#     def Growth(T, fC, fP, fLg, fLp, fCr, feff):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         DC = vH(L, vh_max,T) * fC
#         DP = vP(L, vp_max,T) * fP
#         DLig = vlig(L, vlig_max,T) * fLg
#         Dlip = vlip(T) * fLp
#         DCr = vCr(T) * fCr
#         G = feff * (DC + DP + DLig+Dlip+DCr)
#         # Mnetf = DP / fixed_param['CNP'] - G / fixed_param['CNB']
#         return G

#     Nt = len(tsim)
#     Mnet = np.zeros((Nt, 1))
#     CUE = np.zeros((Nt, 1))
#     G = np.zeros((Nt, 1))
#     s = np.zeros((Nt, 1))
#     DR = np.zeros((Nt, 1))
#     omega = 0.25
#     theta = 0.50
#     max_iter = 1000
#     tol = 1.0e-4

#     # initial condition
#     uCarb_n, uP_n, uLig_n, uLip_n, uCr_n = init_fracC
#     uco2_n = 0
#     ue_n = CUE_func(uCarb_n, uP_n, uLig_n, uLip_n, uCr_n)
#     fC = np.array([uCarb_n, uP_n, uLig_n, uLip_n, uCr_n])
#     fC = fC/np.sum(fC)
#     DR[0] = 4-np.dot(fixed_param['nosc'], fC)

#     Lloop = uLig_n / (uCarb_n + uP_n + uLig_n + uLip_n + uCr_n)
#     DCloop = vH(Lloop, vh_max) * uCarb_n
#     DPloop = vP(Lloop, vp_max) * uP_n
#     DLigloop = vlig(Lloop, vlig_max) * uLig_n
#     Dliploop = vlip(T) * uLip_n
#     DCrloop = vCr(T) * uCr_n
#     Gloop = ue_n * (DCloop + DPloop + DLigloop + Dliploop + DCrloop)
#     temp_Mnet = DPloop / fixed_param['CNP']-Gloop / fixed_param['CNB']
#     N_lim_test = -temp_Mnet < fixed_param['Inorg']

#     if temp_Mnet > 0 or N_lim_test:
#         CUE[0] = ue_n
#         Mnet[0] = DPloop / fixed_param['CNP']-Gloop / fixed_param['CNB']
#         G[0] = Gloop
#     else:
#         CUE[0] = fixed_param['CNB'] * (fixed_param['Inorg'] + DPloop / fixed_param['CNP']
#                                        )/(DCloop + DPloop + DLigloop + Dliploop + DCrloop)
#         Mnet[0] = -fixed_param['Inorg']
#         G[0] = CUE[0]*(DCloop + DPloop + DLigloop + Dliploop + DCrloop)

#     S = np.zeros((Nt, 6))
#     totCg = np.zeros((Nt, 1))
#     totNg = np.zeros((Nt, 1))
#     s[0] = uCarb_n + uP_n + uLig_n + uLip_n + uCr_n + uco2_n
#     S[0, :] = [uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, uco2_n]
#     totCg[0] = uCarb_n + uP_n + uLig_n + uLip_n + uCr_n
#     totNg[0] = uP_n/fixed_param['CNP']  # convert protein gC in gN
#     effi = ue_n
#     for n in range(1, Nt):
#         uCarb_, uP_, uLig_, uLip_, uCr_, uco2_, ue_ = uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, uco2_n, ue_n
#         converged = False
#         r = 1
#         dt = tsim[n] - tsim[n-1]

#         while not converged:

#             uCarb_new = theta * dt * Ucarb(uCarb_, uP_, uLig_, uLip_, uCr_, ue_) + (
#                 1 - theta) * dt * Ucarb(uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, ue_n) + uCarb_n
#             Carb = omega * uCarb_new + (1 - omega) * uCarb_

#             uP_new = theta * dt * Uprot(uCarb_, uP_, uLig_, uLip_, uCr_, ue_) + (
#                 1 - theta) * dt * Uprot(uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, ue_n) + uP_n
#             prot = omega * uP_new + (1 - omega) * uP_

#             uLig_new = theta * dt * Ulignin(uCarb_, uP_, uLig_, uLip_, uCr_, ue_) + (
#                 1 - theta) * dt * Ulignin(uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, ue_n) + uLig_n
#             Lig = omega * uLig_new + (1 - omega) * uLig_

#             uLip_new = theta * dt * Ulipid(uCarb_, uP_, uLig_, uLip_, uCr_, ue_) + (
#                 1 - theta) * dt * Ulipid(uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, ue_n) + uLip_n
#             Lip = omega * uLip_new + (1 - omega) * uLip_

#             uCr_new = theta * dt * UCarbonyl(uCarb_, uP_, uLig_, uLip_, uCr_, ue_) + (
#                 1 - theta) * dt * UCarbonyl(uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, ue_n) + uCr_n
#             Carbonyl = omega * uCr_new + (1 - omega) * uCr_

#             uco2_new = theta * dt * f_co2(uCarb_, uP_, uLig_, uLip_, uCr_, ue_) + (
#                 1 - theta) * dt * f_co2(uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, ue_n) + uco2_n
#             co2 = omega * uco2_new + (1 - omega) * uco2_

#             Lloop = Lig / (Carb + prot + Lig + Lip+Carbonyl)
#             DCloop = vH(Lloop,  vh_max) * Carb
#             DPloop = vP(Lloop,  vp_max) * prot
#             DLigloop = vlig(Lloop, vlig_max) * Lig
#             Dliploop = vlip(T) * Lip
#             DCrloop = vCr(T) * Carbonyl
#             Gloop = effi * (DCloop + DPloop + DLigloop + Dliploop + DCrloop)
#             temp_Mnet = DPloop / fixed_param['CNP']-Gloop / fixed_param['CNB']
#             N_lim_test = Gloop / fixed_param['CNB'] - DPloop / fixed_param['CNP'] < fixed_param['Inorg']

#             if temp_Mnet > 0 or N_lim_test:  # true mean no N limitation
#                 ue_new = CUE_func(Carb, prot, Lig, Lip, Carbonyl)
#                 Mnet_loop = temp_Mnet
#                 G_temp = Gloop
#             else:
#                 ue_new = fixed_param['CNB'] * (fixed_param['Inorg'] + DPloop / fixed_param['CNP']
#                                                )/(DCloop + DPloop + DLigloop + Dliploop + DCrloop)
#                 Mnet_loop = -fixed_param['Inorg']
#                 G_temp = fixed_param['CNB'] * (fixed_param['Inorg'] + DPloop / fixed_param['CNP'])
                
                
#             effi = omega * ue_new + (1 - omega) * ue_
#             r = r + 1
#             # Stopping criteria
#             q1 = np.max(np.array([
#                 abs(Carb - uCarb_),
#                 abs(prot - uP_),
#                 abs(Lig - uLig_),
#                 abs(Lip - uLip_),
#                 abs(Carbonyl - uCr_),
#                 abs(co2 - uco2_),
#                 abs(effi - ue_)
#             ]))

#             uCarb_, uP_, uLig_, uLip_, uCr_, uco2_, ue_ = Carb, prot, Lig, Lip, Carbonyl,  co2, effi
#             converged = q1 < tol or r >= max_iter
#             # print(r)

#         uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, uco2_n, ue_n = Carb, prot, Lig, Lip, Carbonyl, co2, effi
#         Carb, prot, Lig, Lip, Carbonyl, co2, effi = uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, uco2_n, ue_n

#         CUE[n] = effi
#         Mnet[n] = Mnet_loop
#         S[n, :] = np.array([Carb, prot, Lig, Lip, Carbonyl, co2])
#         totCg[n] = Carb + prot + Lig + Lip + Carbonyl
#         totNg[n] = prot/fixed_param['CNP']  # convert protein gC in gN
#         s[n] = Carb + prot + Lig + Lip + Carbonyl + co2
#         G[n] = G_temp
#         fC = np.array([Carb, prot, Lig, Lip, Carbonyl])
#         fC = fC/np.sum(fC)
#         DR[n] = 4-np.dot(fixed_param['nosc'], fC)

#         out = np.hstack((tsim.reshape(len(tsim), 1), totCg, totNg, S, CUE, Mnet, G, DR, s))
#         df = pd.DataFrame(out, columns=["time", "totCg", "totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC',
#                                             'carbonyl_gC', "CO2 [gC]", "CUE", "MNet [gN/day]", "Growth rate [gC/day]", "DR", "sumPool"])

#     return df


# def N_Retention(guess_param_val, init_fracC, fixed_param, tsim,Temperature, protection, CUEflag, voflag):

#     vh_max, vp_max, vlig_max, vlip_max, vCr_max = np.array(list(guess_param_val))
    
#     a, b = fixed_param['a'], fixed_param['b']
    
#     EaCh, EaP, EaLig, EaLip, EaCr = fixed_param['Ea'].values()
#     R = 8.314 # J/mol K
        
#     if protection:
#         def vH(l, vh_max,T): return vh_max*np.exp(-(l / a)**b)*np.exp(-EaCh/(R*T))
#         def vP(l, vp_max,T): return vp_max*np.exp(-(l / a)**b)*np.exp(-EaP/(R*T))
#     else:
#         def vH(l, vh_max,T): return vh_max*np.exp(-EaCh/(R*T))
#         def vP(l, vp_max,T): return vp_max*np.exp(-EaP/(R*T))
        
#     if voflag:
#         def vlig(l, vlig_max,T): return vlig_max*(1-np.exp(-(l / a)**b))*np.exp(-EaLig/(R*T))
#     else:
#         def vlig(l, vlig_max,T): return vlig_max*np.exp(-EaLig/(R*T))
    
#     def vlip(T): return vlip_max*np.exp(-EaLip/(R*T))
#     def vCr(T): return vCr_max*np.exp(-EaCr/(R*T))

#     # plt.scatter(Temperature, vlip(Temperature))


#     def CUE_func(fC, fP, fLg, fLp, fCr):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         frac = np.array([fC, fP, fLg, fLp, fCr])
#         frac = frac/np.sum(frac)
#         litDR = 4-np.dot(fixed_param['nosc'], frac)
#         if CUEflag:
#             return efficiency(litDR)*np.exp(-(L / a)**b)
#         else:
#             return efficiency(litDR)
        
#     def derivatives(T, fC, fP, fLg, fLp, fCr,f_eta):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         DC = vH(L, vh_max,T) * fC
#         DP = vP(L, vp_max,T) * fP
#         DLig = vlig(L, vlig_max,T) * fLg
#         DLip = vlip(T) * fLp
#         DCr = vCr(T) * fCr
#         feff=CUE_func(fC, fP, fLg, fLp, fCr)
#         GC = feff * (DC + DP + DLig+DLip+DCr) # growth rate under C limitation
        
#         temp_Mnet = DP / fixed_param['CNP']-(1-f_eta)*GC / fixed_param['CNB']
#         N_lim_test = (1-f_eta)*GC / fixed_param['CNB'] - DP / fixed_param['CNP'] < fixed_param['Inorg']

#         if temp_Mnet > 0 or N_lim_test:  # true mean no N limitation
#             # eta = 1
#             G = GC # growth rate under C limitation
#         else:
#             G = (fixed_param['CNB']/(1-f_eta))*(DP / fixed_param['CNP'] +fixed_param['Inorg']) # growth rate under N limitation
            
        
#         mC = 1 - (1-f_eta)*fixed_param['mP'] - fixed_param['mLp']-fixed_param['mLg']-fixed_param['mCr']
#         dCarbdt = mC * G - DC
#         dProtdt = (1-f_eta)*fixed_param['mP']  * G - DP
#         dLigdt = fixed_param['mLg'] * G - DLig
#         # mLp = 1 - fixed_param['mC'] - (1-f_eta)*fixed_param['mP']- fixed_param['mLg'] -  fixed_param['mCr']
#         dLipdt = fixed_param['mLp'] * G - DLip
#         dCarbonyldt = fixed_param['mCr'] * G - DCr
#         dCO2dt = (DC + DP + DLig+DLip+DCr) -G
        
#         return [dCarbdt, dProtdt, dLigdt, dLipdt, dCarbonyldt, dCO2dt,G]
    
#     def eta_func(T,fC, fP, fLg, fLp, fCr):
#         totC = fC + fP + fLg + fLp + fCr
#         L = fLg / totC
#         DC = vH(L, vh_max,T) * fC
#         DP = vP(L, vp_max,T) * fP
#         DLig = vlig(L, vlig_max,T) * fLg
#         DLip = vlip(T) * fLp
#         DCr = vCr(T) * fCr
#         feff=CUE_func(fC, fP, fLg, fLp, fCr)
#         GC = feff * (DC + DP + DLig+DLip+DCr) # growth rate under C limitation
#         eta = 1 - (fixed_param['CNB']/GC)*(DP/fixed_param['CNP'] + fixed_param['Inorg'])
#         return eta

#     Nt = len(tsim)
#     Mnet = np.zeros((Nt, 1))
#     CUE = np.zeros((Nt, 1))
#     ETA = np.zeros((Nt, 1))
#     G = np.zeros((Nt, 1))
#     s = np.zeros((Nt, 1))
#     DR = np.zeros((Nt, 1))
#     omega = 0.25
#     theta = 0.50
#     max_iter = 1000
#     tol = 1.0e-6
    
#     # fC, fP, fLg, fLp, fCr = uCarb_n, uP_n, uLig_n, uLip_n, uCr_n

#     # initial condition
#     uCarb_n, uP_n, uLig_n, uLip_n, uCr_n = init_fracC
#     uco2_n = 0
#     fC = np.array([uCarb_n, uP_n, uLig_n, uLip_n, uCr_n])
#     fC = fC/np.sum(fC)
#     DR[0] = 4-np.dot(fixed_param['nosc'], fC)
#     CUE[0] = CUE_func(uCarb_n, uP_n, uLig_n, uLip_n, uCr_n)
#     ue_n = 0 # initially we assume C limited condition which will be update basedon MNet
   
#     Lloop = uLig_n / (uCarb_n + uP_n + uLig_n + uLip_n + uCr_n)
#     DCloop = vH(Lloop, vh_max,Temperature[0]) * uCarb_n
#     DPloop = vP(Lloop, vp_max,Temperature[0]) * uP_n
#     DLigloop = vlig(Lloop, vlig_max,Temperature[0]) * uLig_n
#     DLiploop = vlip(Temperature[0]) * uLip_n
#     DCrloop = vCr(Temperature[0]) * uCr_n
#     Gloop = CUE[0] * (DCloop + DPloop + DLigloop + DLiploop + DCrloop)
#     temp_Mnet = DPloop / fixed_param['CNP']- (1-ue_n)*Gloop / fixed_param['CNB']
#     N_lim_test = -temp_Mnet < fixed_param['Inorg']

#     if temp_Mnet > 0 or N_lim_test:
#         ETA[0] = 0
#         Mnet[0] = DPloop / fixed_param['CNP']- (1-ue_n)*Gloop / fixed_param['CNB']
#         G[0] = Gloop
#     else:
#         ETA[0] = eta_func(Temperature[0],uCarb_n, uP_n, uLig_n, uLip_n, uCr_n)
#         Mnet[0] = -fixed_param['Inorg']
#         G[0] = (fixed_param['CNB']/(1-ETA[0]))*(DPloop / fixed_param['CNP'] +fixed_param['Inorg'])
 
#     S = np.zeros((Nt, 6))
#     totCg = np.zeros((Nt, 1))
#     totNg = np.zeros((Nt, 1))
#     s[0] = uCarb_n + uP_n + uLig_n + uLip_n + uCr_n + uco2_n
#     S[0, :] = [uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, uco2_n]
#     totCg[0] = uCarb_n + uP_n + uLig_n + uLip_n + uCr_n
#     totNg[0] = uP_n/fixed_param['CNP']  # convert protein gC in gN
#     eta = ue_n
    
#     for n in range(1, Nt):
#         uCarb_, uP_, uLig_, uLip_, uCr_, uco2_, ue_ = uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, uco2_n, ue_n
#         converged = False
#         r = 1
#         dt = tsim[n] - tsim[n-1]

#         while not converged:
#             DC_n, DP_n, DLig_n, DLip_n, DCr_n, DCO2_n, _ = derivatives(Temperature[n],uCarb_n, uP_n, uLig_n, uLip_n, uCr_n,ue_n)
#             DC_, DP_, DLig_, DLip_, DCr_,DCO2_,_  = derivatives(Temperature[n],uCarb_, uP_, uLig_, uLip_, uCr_,ue_)
            
#             uCarb_new = (theta * DC_n + (1 - theta) * DC_)*dt + uCarb_n
#             Carb = omega * uCarb_new + (1 - omega) * uCarb_

#             uP_new = (theta * DP_n + (1 - theta) * DP_)* dt + uP_n
#             prot = omega * uP_new + (1 - omega) * uP_

#             uLig_new = (theta * DLig_n + (1 - theta) *DLig_)* dt + uLig_n
#             Lig = omega * uLig_new + (1 - omega) * uLig_

#             uLip_new = (theta *  DLip_n + (1 - theta) * DLip_)*dt + uLip_n
#             Lip = omega * uLip_new + (1 - omega) * uLip_

#             uCr_new = (theta * DCr_n + (1 - theta) * DCr_)*dt + uCr_n
#             Carbonyl = omega * uCr_new + (1 - omega) * uCr_

#             uco2_new = (theta * DCO2_n + (1 - theta) *DCO2_)*dt + uco2_n
#             co2 = omega * uco2_new + (1 - omega) * uco2_

#             Lloop = Lig / (Carb + prot + Lig + Lip+Carbonyl)
#             DCloop = vH(Lloop,  vh_max,Temperature[n]) * Carb
#             DPloop = vP(Lloop,  vp_max,Temperature[n]) * prot
#             DLigloop = vlig(Lloop, vlig_max,Temperature[n]) * Lig
#             DLiploop = vlip(Temperature[n]) * Lip
#             DCrloop = vCr(Temperature[n]) * Carbonyl
#             effi = CUE_func(Carb, prot, Lig, Lip, Carbonyl)
#             Gloop = effi * (DCloop + DPloop + DLigloop + DLiploop + DCrloop)
#             # temp_Mnet = DPloop / fixed_param['CNP']-(1-eta)*Gloop / fixed_param['CNB']
#             temp_Mnet = DPloop / fixed_param['CNP']-Gloop / fixed_param['CNB']

#             N_lim_test = -temp_Mnet < fixed_param['Inorg']

#             if temp_Mnet > 0 or N_lim_test:  # true mean no N limitation
#                 ue_new = 0
#                 Mnet_loop = temp_Mnet
#                 G_temp = Gloop
#             else:
#                 ue_new = eta_func(Temperature[n],Carb, prot, Lig, Lip, Carbonyl)
#                 Mnet_loop = -fixed_param['Inorg']
#                 G_temp = (fixed_param['CNB']/(1-ue_new))*(DPloop / fixed_param['CNP'] +fixed_param['Inorg'])
                
                
#             eta = omega * ue_new + (1 - omega) * ue_
#             r = r + 1
#             # Stopping criteria
#             q1 = np.max(np.array([
#                 abs(Carb - uCarb_),
#                 abs(prot - uP_),
#                 abs(Lig - uLig_),
#                 abs(Lip - uLip_),
#                 abs(Carbonyl - uCr_),
#                 abs(co2 - uco2_),
#                 abs(eta - ue_)
#             ]))

#             uCarb_, uP_, uLig_, uLip_, uCr_, uco2_, ue_ = Carb, prot, Lig, Lip, Carbonyl,  co2, eta
#             converged = q1 < tol or r >= max_iter
#             # print(r)

#         uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, uco2_n, ue_n = Carb, prot, Lig, Lip, Carbonyl, co2, eta
#         Carb, prot, Lig, Lip, Carbonyl, co2, eta = uCarb_n, uP_n, uLig_n, uLip_n, uCr_n, uco2_n, ue_n

#         ETA[n] = eta 
#         CUE[n] = effi
#         Mnet[n] = Mnet_loop
#         S[n, :] = np.array([Carb, prot, Lig, Lip, Carbonyl, co2])
#         totCg[n] = Carb + prot + Lig + Lip + Carbonyl
#         totNg[n] = prot/fixed_param['CNP']  # convert protein gC in gN
#         s[n] = Carb + prot + Lig + Lip + Carbonyl + co2
#         G[n] = G_temp
#         fC = np.array([Carb, prot, Lig, Lip, Carbonyl])
#         fC = fC/np.sum(fC)
#         DR[n] = 4-np.dot(fixed_param['nosc'], fC)

#         out = np.hstack((tsim.reshape(len(tsim), 1), totCg, totNg, S, CUE,ETA, Mnet, G, DR, s))
#         df = pd.DataFrame(out, columns=["time", "totCg", "totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC',
#                                             'carbonyl_gC', "CO2 [gC]", "CUE","ETA", "MNet [gN/day]", "Growth rate [gC/day]", "DR", "sumPool"])

#     return df


# def residual_fun(x,  init_fracC, fixed_param, tsim,Temperature, data,data_col,protection,CUEflag,voflag):
#     df = N_Retention(x,  init_fracC, fixed_param, tsim,Temperature, protection,CUEflag,voflag)
#     S_cols = df[data_col]
#     splines = {col: interp1d(tsim, S_cols[col], kind='linear', fill_value='extrapolate') for col in data_col}

#     # simC = np.array([splines[col](data['time day']) / splines[col](tsim)[0] if splines[col](tsim)[0] != 0
#     #                  else splines[col](data['time day']) / splines[col](tsim)[1] for col in data_col])
#     simC = np.array([splines[col](data['time day']) / data[col].max() for col in data_col])
#     # obs = np.array([data[col] / data[col].iloc[0] if data[col].iloc[0] != 0
#     #                 else data[col] / data[col].iloc[1] for col in data_col])

#     obs = np.array([data[col] / data[col].max() for col in data_col])
#     res = simC.flatten() - obs.flatten()
#     res_without_nan = res[~np.isnan(res)]
#     return res_without_nan


# def fit_data(guess_param,  init_fracC, fixed_param, tsim,Temperature, data, data_col,protection, CUEflag,voflag, loss='soft_l1'):
#     print("fitting in progress...")
#     inital_guess = list(guess_param.values())
#     lb, ub = np.ones(len(inital_guess))*1e-5, np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
#     start_time = time.time()
#     res_lsq = least_squares(residual_fun, inital_guess, loss='soft_l1', f_scale=0.1, bounds=(lb, ub),
#                             args=(init_fracC, fixed_param, tsim,Temperature, data, data_col, protection,CUEflag,voflag),
#                             verbose=0)
#     end_time = time.time()
#     print(f"Execution time: {end_time - start_time:.6f} seconds")
    
#     cov_matrix = np.linalg.inv(res_lsq.jac.T @ res_lsq.jac)
#     # Extract the standard errors (square root of the diagonal elements of the covariance matrix)
#     parameter_uncertainties = np.sqrt(np.diag(cov_matrix))
#     est_par_name = list(guess_param.keys())

#     est_pars = {est_par_name[i]: res_lsq.x[i] for i in range(len(res_lsq.x))}
#     est_pars_se = {est_par_name[i]+"_se": parameter_uncertainties[i] for i in range(len(parameter_uncertainties))}

#     print(est_pars)
#     df = N_Retention(list(est_pars.values()),  init_fracC, fixed_param, tsim,Temperature, protection,CUEflag,voflag)

#     perf_matrix,model_data = cal_perf_matrix(est_pars,init_fracC, fixed_param, tsim,Temperature, data, data_col, protection,CUEflag,voflag)
#     print("fitting completed...")
#     return df, est_pars, perf_matrix,est_pars_se,model_data


# def global_residual_fun(studyname, guess_param,fixed_param,plant_data,protection = False,CUEflag = False, voflag=False):

#     data_col = ['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']
#     res_df = pd.DataFrame(columns=["Study", "Species","Compound", "Simulated","Observed"])   
#     # study = ['Preston et al. 2009','Almendros et al. 2000','Quideau et al 2005','Mathers et al., 2007',
#     #          'Sjöberg et al 2004','Pastorelli et al 2021','De Marco et al. 2021','Bonanomi et al 2011',
#     #          'Certini et al 2023','Li et al 2020','Gao et al 2016','Wang et al 2013','Wang et al 2019']
#     # for studyname in study:
#     temp = plant_data[plant_data["Study"] == studyname]
#     SP = temp['Species'].unique()
#     sp = SP[-1]
#     for sp in SP:
#         data = temp[temp["Species"] == sp].reset_index(drop=True)
#         if pd.isna(data['carbohydrate_gC'].iloc[0]):
#             data = data.iloc[1:].reset_index(drop=True)
#             data['time day'] = data['time day'] - data['time day'][0]
            
#         data = data.dropna(subset=["carbohydrate_gC"]).reset_index(drop=True)
#         dt = data['time day'][1] - data['time day'][0]
#         if (np.sum(~np.isnan(data['totNg']))>np.sum(~np.isnan(data['protein_gN']))):
#             Imax = np.nanmax(np.gradient(data['totNg'], dt))
#         else:
#             Imax = np.nanmax(np.gradient(data['protein_gN'], dt))
        
#         if Imax > 0:
#             fixed_param['Inorg'] = Imax
            
#         if len(data['totNg'].dropna()) == len(data['protein_gC'].dropna()):
#             data_col = ["totCg",'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']
#         else:
#             data_col = ["totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']
        
#         init_fracC = data[['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']].iloc[0].values
#         tsim = np.linspace(data['time day'].iloc[0], data['time day'].iloc[-1], 50)
        
#         Temperature = pchip_interpolate(data['time day'], data['Incubtation T']+273.15, tsim)
#         df = N_Retention(guess_param,  init_fracC, fixed_param, tsim,Temperature, protection,CUEflag,voflag)
#         S_cols = df[data_col]
#         splines = {col: interp1d(tsim, S_cols[col], kind='linear') for col in data_col}
        
#         simC = np.array([splines[col](data['time day']) / data[col].max() for col in data_col])
#         obs = np.array([data[col] / data[col].max() for col in data_col])
#         simC= simC.flatten()
#         obsC = obs.flatten()
        
#         compound = [col for col in data_col for _ in range(len(data[col]))]
#         df = pd.DataFrame({"Study":studyname, "Species":sp,"Compound":compound, "Simulated":simC,"Observed":obsC})
#         res_df=pd.concat([res_df,df], ignore_index=True)
        
#         # simulated = np.hstack((simulated,simC))
#         # observation = np.hstack((observation,obsC))
   
    
#     # # fit for 'Ono et al. 2009'
#     # for studyname in ['Ono et al 2009','Ono et al 2011','Ono et al 2013', 'McKee et al., 2016']:
#     #     temp = plant_data[plant_data["Study"] == studyname]
#     #     SP = temp['Species'].unique()
#     #     # sp = SP[0]
#     #     for sp in SP:
#     #         data = temp[temp["Species"] == sp].reset_index(drop=True)

#     #         dt = data['time day'][1] - data['time day'][0]
#     #         if (np.sum(~np.isnan(data['totNg']))>np.sum(~np.isnan(data['protein_gN']))):
#     #             Imax = np.nanmax(np.gradient(data['totNg'], dt))
#     #         else:
#     #             Imax = np.nanmax(np.gradient(data['protein_gN'], dt))
                
#     #         if Imax > 0:
#     #             fixed_param['Inorg'] = Imax
#     #         else:
#     #             fixed_param['Inorg'] = 1e-8
#     #         init_fracC = data[['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']].iloc[0].values
        
#     #         if len(data['totNg'].dropna()) == len(data['protein_gC'].dropna()):
#     #             data_col = ["totCg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'DR']
#     #         else:
#     #             data_col = ["totCg", "totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'DR']
                
#     #         tsim = np.linspace(data['time day'].iloc[0], data['time day'].iloc[-1], 50)
            
#     #         Temperature = pchip_interpolate(data['time day'], data['Incubtation T']+273.15, tsim)
#     #         df = N_Retention(guess_param,  init_fracC, fixed_param, tsim,Temperature, protection,CUEflag,voflag)
#     #         S_cols = df[data_col]
#     #         splines = {col: interp1d(tsim, S_cols[col], kind='linear') for col in data_col}
#     #         simC = np.array([splines[col](data['time day']) / data[col].max() for col in data_col])
#     #         obs = np.array([data[col] / data[col].max() for col in data_col])
#     #         simC= simC.flatten()
#     #         obsC = obs.flatten()
#     #         compound = [col for col in data_col for _ in range(len(data[col]))]
#     #         df = pd.DataFrame({"Study":studyname, "Species":sp,"Compound":compound, "Simulated":simC,"Observed":obsC})
#     #         res_df=pd.concat([res_df,df], ignore_index=True)
#     res = res_df["Simulated"]-res_df["Observed"]
#     res_without_nan = res[~np.isnan(res)].values
#     return res_without_nan,res_df



# def cal_perf_matrix(est_pars,init_fracC, fixed_param, tsim,Temperature, data, data_col, protection,CUEflag,voflag):
#     df = N_Retention(list(est_pars.values()),  init_fracC, fixed_param, tsim,Temperature, protection,CUEflag,voflag)
#     S_cols = df[data_col]

#     perf_matrix = pd.DataFrame(index=['r2', 'rmse','AIC'], columns=["overall", "totCg", "totNg", 'carbohydrate_gC',
#                                                               'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC'])
#     splines = {col: interp1d(tsim, S_cols[col], kind='linear', fill_value='extrapolate') for col in data_col}
    
#     # AICfun = lambda n, p, mse: n * np.log(mse) + 2 * n * p / (n - p - 1) # n= # observation, p= # parameters, mse= mean squared error
#     AICfun = lambda n, p, mse: n * np.log(mse) + 2 *p # n= # observation, p= # parameters, mse= mean squared error

#     for col in data_col:
#         # y_pred = splines[col](data['time day'])/ data[col].max()
#         # y_true = data[col]/ data[col].max()
#         y_pred = splines[col](data['time day'])
#         y_true = data[col]

#         valid_indices = ~np.isnan(y_true)
#         # Check if all valid_indices are False
#         if not np.any(valid_indices):
#             # If all valid_indices are False, y_true is empty or contains only NaN values
#             perf_matrix.loc[['r2','rmse','AIC'], col] = np.nan
#         else:
#             # Otherwise, there are valid values in y_true
#             y_true = y_true[valid_indices]
#             y_pred = y_pred[valid_indices]
#             perf_matrix.loc['r2', col] = r2_score(y_true, y_pred)
#             perf_matrix.loc['rmse', col] = np.sqrt(mean_squared_error(y_true, y_pred))
#             perf_matrix.loc['AIC', col] = AICfun(len(y_true), len(est_pars),mean_squared_error(y_true, y_pred))
                        

#     y_pred = np.array([splines[col](data['time day'])/ data[col].max() for col in data_col]).flatten()
#     y_true = np.array([data[col]/ data[col].max() for col in data_col]).flatten()
#     valid_indices = ~np.isnan(y_true)
#     y_true = y_true[valid_indices]
#     y_pred = y_pred[valid_indices]
#     perf_matrix.loc['r2', 'overall'] = r2_score(y_true, y_pred)
#     perf_matrix.loc['rmse', 'overall'] = np.sqrt(mean_squared_error(y_true, y_pred))
#     perf_matrix.loc['AIC', 'overall'] = AICfun(len(y_true), len(est_pars),mean_squared_error(y_true, y_pred))

#     temp_col = data_col.copy()
#     if 'DR' not in temp_col:
#         temp_col.append('DR')
#     S_cols = df[temp_col]
#     splines = {col: interp1d(tsim, S_cols[col], kind='linear', fill_value='extrapolate') for col in temp_col}
#     y_pred1 = np.array([splines[col](data['time day']) for col in temp_col]).flatten()
#     y_true1 = np.array([data[col] for col in temp_col]).flatten()
#     cat = [np.repeat(col, len(data['time day'])) for col in temp_col]
#     cat = np.array(cat).flatten().tolist()
    
#     obstime = np.tile(data['time day'].values, len(temp_col))
#     model_data = pd.DataFrame({"time":obstime,"pool":cat,'obs':y_true1,'sim':y_pred1})
#     # plt.figure, plt.plot(model_data['sim'],model_data['obs'],'o'), plt.plot()
#     return perf_matrix, model_data

# def sensFun(func, params, sensvar, out_var, *args, **kwargs):
#     """
#     Calculate sensitivities of the output variable with respect to the specified parameters.

#     Parameters:
#     - func: function
#         The function that computes the model output as dataframe
#     - params: dict
#         Dictionary containing parameter values.
#     - sensvar: dict
#         Dictionary containing variable names for which sensitivities are calculated.
#     - out_var: str
#         The name of the output variable for which sensitivities are computed.
#     - *args: tuple
#         Additional arguments to be passed to the `func` function.

#     Returns:
#     - pandas.DataFrame
#         DataFrame containing sensitivities of the output variable with respect to each state variable.

#     Note:
#     - This function computes the sensitivity of the output variable with respect to each parameter specified in `sensvar`.
#     - Sensitivity is calculated using a small perturbation (`dp`) of each parameter value and finite difference approximation.
#     """
#     yRef = func(list(params.values()), *args)[out_var]
#     # return yRef

#     Sens = pd.DataFrame(columns=['Time', 'State_Variable'])
#     pp = params.copy()
#     # Iterate over state variable names
#     tiny = 1e-3
#     for spar_nam in sensvar.keys():
#         dp = params[spar_nam]*tiny
#         params[spar_nam] = params[spar_nam] + dp
#         yPert = func(list(params.values()), *args)[out_var]
#         si = (yPert - yRef) / dp * params[spar_nam]
#         simelt = pd.melt(si, var_name='State_Variable', value_name='sensitivity')
#         Sens[spar_nam] = simelt['sensitivity']
#         params[spar_nam] = pp[spar_nam]
#     Sens['State_Variable'] = simelt['State_Variable']
#     return Sens


# def collinearity(func, out_var, params, *args):
#     sensvar = params
#     par_name = list(sensvar.keys())
#     sens_par_df = pd.DataFrame(columns=['var', 'num_var'] + list(sensvar.keys())+["collinearity"])
#     for r in range(2, len(sensvar)+1):
#         # Generate combinations of length r
#         combinations = itertools.combinations(sensvar.items(), r)
#         # Convert each combination to a dictionary and add it to sens_par
#         for combo in combinations:
#             # print(combo)
#             tempdict = {'num_var': r, **dict(combo)}
#             temp_df = pd.DataFrame([tempdict])
#             sens_par_df = pd.concat([sens_par_df, temp_df], ignore_index=True)

#     gamma = np.zeros(len(sens_par_df))
#     colli_df = pd.DataFrame()
    
#     for var in out_var:
#         for i in tqdm(range(len(sens_par_df))):
#             sens_pars = dict(sens_par_df[par_name].loc[i].dropna())
#             Sens = sensFun(func, params, sens_pars, out_var, *args)
#             sens_pars_nam = list(sens_pars.keys())
#             tsens = Sens[Sens['State_Variable'] == var][sens_pars_nam].values
#             tsens = tsens[1:, :]
#             # mabs = np.mean(np.abs(tsens), axis=0)
#             # nout = tsens.shape[0]
#             # msqr = np.sum(tsens*tsens, axis=0)
#             # # root mean squared sensitivity: Below a certain threshold the parameters are considered non-influential to the outputs.
#             # msqr = np.sqrt(np.sum(tsens*tsens, axis=0)/nout)
#             sij_norm = np.sqrt(np.sum(tsens*tsens, axis=1))
#             sij_norm_reshaped = sij_norm[:, np.newaxis]
#             sij_hat = tsens/sij_norm_reshaped
#             # sij_hat = sij_hat[~np.isnan(sij_hat).any(axis=1)]
#             s_mat = np.matmul(sij_hat.T, sij_hat)
#             eigenvalues, eigenvectors = np.linalg.eig(s_mat)
#             gamma[i] = 1/(np.sqrt(np.min(eigenvalues)))
#         sens_par_df["collinearity"] = gamma
#         sens_par_df["var"] = var
#         colli_df = pd.concat([colli_df, sens_par_df]).reset_index(drop=True)
#     return colli_df


# def plot_model(tsim, fixed_param, est_pars, init_fracC, data_col=None, data=None, protection=True, CUEflag=True,voflag=True):
#     if data is not None:
#         Temperature = pchip_interpolate(data['time day'], data['Incubtation T']+273.15, tsim)
#     else:
#         Temperature = np.ones(len(tsim))*273.15


#     # df = CUE_adaptation(list(est_pars.values()),  init_fracC, fixed_param, tsim,Temperature, protection,CUEflag,voflag)
#     df = N_Retention(list(est_pars.values()),  init_fracC, fixed_param, tsim,Temperature, protection,CUEflag,voflag)

#     a, b = fixed_param['a'], fixed_param['b']

#     col = ['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']

#     plt.style.use('ggplot')
#     palette = sns.color_palette()
#     plt.style.use('default')

#     fig, ax = plt.subplots(4, 4, figsize=(14, 8))
#     ax = ax.flatten()

#     # Plotting on axes
#     ax[0].plot(df['time'], df["totCg"], linewidth=1.5)
#     if data is not None:
#         ax[0].scatter(data['time day'], data["totCg"], linewidth=1.5)
#     ax[0].set_xlabel('Time [d]')
#     ax[0].set_ylabel('total C [g]')

#     ax[1].plot(df['time'], df["totNg"], linewidth=1.5, label="model")
#     if data is not None:
#         ax[1].scatter(data['time day'], data["totNg"], linewidth=1.5, label="Ng")
#         ax[1].scatter(data['time day'], data["protein_gN"], linewidth=1.5, label="protein_gN")
#     ax[1].set_xlabel('Time [d]')
#     ax[1].set_ylabel('total N [g]')
#     ax[1].legend(fontsize=8, loc = "best",frameon=False)
    
#     # Loop over columns and plot data for the first set of axes
#     for i, column in enumerate(col):
#         color = palette[i]
#         ax[2].plot(df['time'], df[column]/df[column].iloc[0], label=column, linewidth=1.5, color=color)
#         if data is not None:
#             ax[2].scatter(data['time day'], data[column]/data[column].iloc[0], linewidth=1.5, color=color)

#     ax[2].set_xlabel('Time [d]')
#     ax[2].set_ylabel('gC/gC[0]')

#     # Loop over columns and plot data for the second set of axes
#     for i, column in enumerate(col):
#         color = palette[i]
#         ax[3].plot(df['time'], df[column], label=column, linewidth=1.5, color=color)
#         if data is not None:
#             ax[3].scatter(data['time day'], data[column], linewidth=1.5, color=color)

#     ax[3].set_xlabel('Time [d]')
#     ax[3].set_ylabel('gC')
#     ax[3].legend(fontsize=8,  loc = "best",frameon=False)

#     # Continue plotting for the third set of axes
#     ax[4].plot(df['time'], df['DR'], linewidth=1.5)
#     if data is not None:
#         ax[4].scatter(data['time day'], data['DR'], linewidth=1.5, color=color)
#     ax[4].set_xlabel('Time [d]')
#     ax[4].set_ylabel('DR')

#     # Continue plotting for the fourth set of axes
#     ax[5].plot(df['time'], df["MNet [gN/day]"], linewidth=1.5)
#     ax[5].set_xlabel('Time [d]')
#     ax[5].set_ylabel("MNet [gN/day]")

#     # Continue plotting for the fifth set of axes
#     ax[6].plot(df['time'], df['CUE'], linewidth=1.5)
#     ax[6].set_xlabel('Time [d]')
#     ax[6].set_ylabel('CUE')

#     # Continue plotting for the sixth set of axes
#     ax[7].plot(df['time'], df["Growth rate [gC/day]"], linewidth=1.5)
#     ax[7].set_xlabel('Time [d]')
#     ax[7].set_ylabel("Growth rate [gC/day]")

#     # Continue plotting for the seventh set of axes
#     L = df["lignin_gC"]/df[col].sum(axis=1)
#     ax[8].plot(df['time'], L, linewidth=1.5)
#     ax[8].set_xlabel('Time [d]')
#     ax[8].set_ylabel("Lignin fraction")
    

#     # Continue plotting for the eighth set of axes (if protection is True)
#     if protection:
#         Ch_protection = est_pars['vh_max']*np.exp(-(L / a)**b)
#         P_protection = est_pars['vp_max']*np.exp(-(L / a)**b)
#         lig_protection = est_pars['vlig']*(1-np.exp(-(L / a)**b))

#         # pfunc = np.exp(-(np.arange(0, 0.8, 0.01) / a)**b)
#         # ax[9].plot(np.arange(0, 0.8, 0.01), pfunc, linewidth=1.5, label="protection function")
#         ax[9].plot(L, Ch_protection, linewidth=3, label="Carb")
#         ax[9].plot(L, P_protection, linewidth=3, label="Prot")
#         if voflag:
#             ax[9].plot(L, lig_protection, linewidth=3, label="Lignin")
#         ax[9].set_xlabel('Lignin fraction')
#         ax[9].set_ylabel("protection func.")
#         ax[9].legend()
        
#         ax[10].plot(np.arange(0, 0.8, 0.01),  np.exp(-(np.arange(0, 0.8, 0.01) / a)**b), linewidth=1.5, label="v_C, v_P modifier")
#         ax[10].plot(L, Ch_protection/est_pars['vh_max'],"-", linewidth=4, label="Carb", color='grey')
#         if voflag:
#             ax[10].plot(np.arange(0, 0.8, 0.01),  1-np.exp(-(np.arange(0, 0.8, 0.01) / a)**b), linewidth=1.5, label="v_Lg modifier")
#         ax[10].plot(L, lig_protection/est_pars['vlig'],"-",  linewidth=4, label="Lignin", color='grey')
#         ax[10].set_xlabel('Lignin fraction'), ax[10].legend(fontsize=9)

#     # Add suptitle if perf_matrix is not None
#     if data is not None:
#         perf_matrix,_ = cal_perf_matrix(est_pars,init_fracC, fixed_param, tsim,Temperature, data, data_col, protection,CUEflag,voflag)

#         suptitle_str = data['Study'][0]+"_"+data["Species"][0] + \
#             f" [Overall rmse: {perf_matrix.loc['rmse', 'overall']:.2E}, Overall r2: {perf_matrix.loc['r2', 'overall']:.2E}]"
#         # for col_name in perf_matrix.columns[1:]:
#         #     suptitle_str += f"{col_name}: rmse = {perf_matrix.loc['rmse',col_name]:.2f}, r2 = {perf_matrix.loc['r2',col_name]:.2f}   "
#         plt.suptitle(suptitle_str, fontsize=12, va='top')


#     ax[11].plot(df['sumPool'], linewidth=1.5, label="protection function")
#     ax[11].set_ylabel("mass balance check")

#     ax[12].plot(df['time'], df['totCg']/df['totNg'], linewidth=1.5)
#     ax[12].set_xlabel('Time [d]')
#     ax[12].set_ylabel(r"$CN$")
    
#     ax[13].plot(df['totCg']/df['totNg'], df['ETA'], linewidth=1.5)
#     ax[13].set_xlabel(r'$CN$')
#     ax[13].set_ylabel(r"$\eta$")
    
#     ax[14].plot(df['time'], df['ETA'], linewidth=1.5)
#     ax[14].set_xlabel('Time [d]')
#     ax[14].set_ylabel(r"$\eta$")
    
#     ax[15].plot(df['time'], (1-df['ETA'])*fixed_param['mP'], linewidth=1.5)
#     ax[15].set_xlabel('Time [d]')
#     ax[15].set_ylabel("P turnover rate costant")
    
#     for axs in ax:
#         axs.grid(True, color='grey', linestyle='-', linewidth=0.25)
#     plt.tight_layout()
#     plt.show()

#     return fig, df

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:20:41 2024

@author: chak803
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error

def molecular_mixing_model(NMR_data):
    """
    Molecular Mixing Model: Calculates fractions of chemical components based on NMR data.

    This model analyzes Nuclear Magnetic Resonance (NMR) spectral data and performs optimization to estimate 
    the distribution of chemical components (Carbohydrate, Protein, Lignin, Lipid, Carbonyl, Char) in a 
    given sample. It also computes molecular formula and oxidation state based on Carbon, Nitrogen, Hydrogen, 
    and Oxygen ratios.

    #### Parameters
    - `NMR_data` (dict): Dictionary containing NMR spectral data and associated parameters. Keys include:
        - `C:N` - C:N ratio observed in the sample.
        - Spectral intensities for ppm ranges (e.g., `A ALKYL 0–45 ppm`, `B METHOX 45–60 ppm`, etc.).
        - Spectra identifiers (`a`, `b`, `c`, etc.).

    #### Returns
    - `frac` (ndarray): Fractional contribution of each chemical component.
    - `CNHO` (DataFrame): DataFrame containing molecular ratios (Carbon, Nitrogen, Hydrogen, Oxygen) and calculated values:
        - `molecularFormula`: Molecular formula string (e.g., CxHyNzOw).
        - `Cox`: Oxidation state of Carbon.
    - `NMR_fit` (ndarray): Fitted NMR spectrum.
    - `r_squared` (float): R-squared value indicating fit quality.
    - `rmse` (float): Root Mean Square Error indicating fit accuracy.
    """
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



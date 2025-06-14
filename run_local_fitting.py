# -*- # from my_modules import coding: utf-8 -*-
"""
Created on Sun Apr 21 14:03:13 2024

@author: Arjun Chakrawal (arjun.chakrawal@pnnl.gov)

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
# from my_modules import plot_model, fit_data, cal_perf_matrix, N-Retention
import sol_ivp_ as svp

# from sol_ivp_ import litter_decay_model, plot_model, residual_fun,cal_perf_matrix,fit_data,load_fix_par_and_data

from tqdm import tqdm
from scipy.interpolate import pchip_interpolate
plt.close('all')

#%%

def run_fitting(adapt_flag="N-Retention", protection=False, CUEflag=False, voflag=False):
    fixed_param, plant_data = svp.load_fix_par_and_data()

    study = plant_data['Study'].unique()

    df_estpar = pd.DataFrame()

    fstr = "protection_"+str(protection)+"_vo_" + str(voflag)+"_CUE_" + str(CUEflag)
    study = ['Preston et al. 2009', 'Almendros et al. 2000', 'Quideau et al 2005', 'Mathers et al., 2007',
             'Sjöberg et al 2004', 'Pastorelli et al 2021', 'De Marco et al. 2021', 'Bonanomi et al 2013',
             'Certini et al 2023', 'Li et al 2020', 'Gao et al 2016', 'Wang et al 2013', 'Wang et al 2019',
             'Ono et al 2009', 'Ono et al 2011', 'Ono et al 2013', 'McKee et al., 2016']
    # study = [ 'Certini et al 2023']
    for studyname in tqdm(study):
        plt.close('all')
        print("running          "+studyname)
        temp = plant_data[plant_data["Study"] == studyname]
        SP = temp['Species'].unique()
        # sp = SP[1]
        guess_param = {'vh_max': 0.0035, 'vp_max': 0.006, 'vlig': 0.0045,
                       'vlip': 0.006, 'vCr': 0.002}
        for sp in tqdm(SP):
            print("running          "+sp)
            data = temp[temp["Species"] == sp].reset_index(drop=True)
            
            
            if pd.isna(data['carbohydrate_gC'].iloc[0]):
                data = data.iloc[1:].reset_index(drop=True)
                data['time day'] = data['time day'] - data['time day'][0]

            dt = data['time day'][1] - data['time day'][0]
            if (np.sum(~np.isnan(data['totNg'])) > np.sum(~np.isnan(data['protein_gN']))):
                Imax = np.nanmax(np.gradient(data['totNg'], dt))
            else:
                Imax = np.nanmax(np.gradient(data['protein_gN'], dt))

            if Imax > 0:
                fixed_param['Inorg'] = Imax
            else:
                fixed_param['Inorg'] = 1e-8

            if len(data['totNg'].dropna()) == len(data['protein_gC'].dropna()):
                data_col = ['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']
            else:
                data_col = ["totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']

            if studyname == 'Quideau et al 2005':
                data_col = ["totCg", "totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']

            if studyname in ['Ono et al 2009', 'Ono et al 2011', 'Ono et al 2013', 'McKee et al., 2016']:
                if len(data['totNg'].dropna()) == len(data['protein_gC'].dropna()):
                    data_col = ["totCg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC']
                else:
                    data_col = ["totCg", "totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC']


            init_fracC = data[['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']].iloc[0].values

            tsim = np.linspace(data['time day'].iloc[0], data['time day'].iloc[-1], 25)
            Temperature = 273

            est_pars,est_pars_se = svp.fit_data(guess_param, init_fracC, fixed_param, tsim, Temperature, data, data_col,adapt_flag, protection, CUEflag, voflag, loss='soft_l1')

            fig, df = svp.plot_model(np.linspace(0, data['time day'].iloc[-1], 200), fixed_param,
                                list(est_pars.values()), init_fracC, data_col, data, adapt_flag,protection, CUEflag, voflag)

            fig.savefig("figs/model_fit/"+adapt_flag+ "/"+studyname+"_"+sp+"_"+fstr+".png", dpi=300)
          
            df_temp = data.loc[0, ["Study", "Species", "Csource", "MATC", 'C:N', 'carbohydrate_MMM', 'protein_MMM', 'lignin_MMM',
                                   'lipid_MMM', 'carbonyl_MMM']].to_frame().T
            df_temp['CUE0'] = df['CUE'][0]
            df_temp['CUE_avg'] = df['CUE'].mean()
            df_temp['DR0'] = df['DR'][0]
            df_temp['DR_avg'] = df['DR'].mean()
            est_pars_df = pd.DataFrame([est_pars])
            est_pars_se_df = pd.DataFrame([est_pars_se])

            mydf = pd.concat([df_temp, est_pars_df, est_pars_se_df], axis=1)
            df_estpar = pd.concat([df_estpar, mydf], axis=0).reset_index(drop=True)

    df_estpar.to_excel('tables/'+ adapt_flag+ '/Local_estpar_'+fstr+'.xlsx', index=False)




def run_post_process(df_estpar,adapt_flag="N-Retention", protection=False, CUEflag=False, voflag=False):
    fixed_param, plant_data = svp.load_fix_par_and_data()
    df_perf_matrix = pd.DataFrame()
    df_mode_data = pd.DataFrame()
    study = plant_data['Study'].unique()

    df_mode_data = pd.DataFrame()
    fstr = "protection_"+str(protection)+"_vo_" + str(voflag)+"_CUE_" + str(CUEflag)

    study = ['Preston et al. 2009', 'Almendros et al. 2000', 'Quideau et al 2005', 'Mathers et al., 2007',
             'Sjöberg et al 2004', 'Pastorelli et al 2021', 'De Marco et al. 2021', 'Bonanomi et al 2013',
             'Certini et al 2023', 'Li et al 2020', 'Gao et al 2016', 'Wang et al 2013', 'Wang et al 2019',
             'Ono et al 2009', 'Ono et al 2011', 'Ono et al 2013', 'McKee et al., 2016']

    studyname = study[0]
    for studyname in tqdm(study):
        plt.close('all')
        print("running          "+studyname)
        temp = plant_data[plant_data["Study"] == studyname]
        SP = temp['Species'].unique()
        # sp = SP[0]
        for sp in tqdm(SP):
            print("running          "+sp)
            data = temp[temp["Species"] == sp].reset_index(drop=True)
            if pd.isna(data['carbohydrate_gC'].iloc[0]):
                data = data.iloc[1:].reset_index(drop=True)
                data['time day'] = data['time day'] - data['time day'][0]

            dt = data['time day'][1] - data['time day'][0]
            if (np.sum(~np.isnan(data['totNg'])) > np.sum(~np.isnan(data['protein_gN']))):
                Imax = np.nanmax(np.gradient(data['totNg'], dt))
            else:
                Imax = np.nanmax(np.gradient(data['protein_gN'], dt))

            if Imax > 0:
                fixed_param['Inorg'] = Imax
            else:
                fixed_param['Inorg'] = 1e-8

            if len(data['totNg'].dropna()) == len(data['protein_gC'].dropna()):
                data_col = ['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']
            else:
                data_col = ["totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']

            if studyname == 'Quideau et al 2005':
                data_col = ["totCg", "totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']
                
            if studyname in ['Ono et al 2009', 'Ono et al 2011', 'Ono et al 2013', 'McKee et al., 2016']:
                if len(data['totNg'].dropna()) == len(data['protein_gC'].dropna()):
                    data_col = ["totCg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC']
                else:
                    data_col = ["totCg", "totNg", 'carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC']


            init_fracC = data[['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']].iloc[0].values

            tsim = np.linspace(data['time day'].iloc[0], data['time day'].iloc[-1], 50)
            Temperature =273.15

            par_nam = ['vh_max', 'vp_max', 'vlig', 'vlip', 'vCr']
            est_pars_val = df_estpar.loc[(df_estpar["Study"] == studyname) &
                                         (df_estpar["Species"] == sp), par_nam].values[0]

            df = svp.litter_decay_model(tsim, init_fracC, est_pars_val, fixed_param, adapt_flag, protection, CUEflag, voflag)

            perf_matrix, model_data = svp.cal_perf_matrix(est_pars_val, init_fracC, fixed_param, tsim, Temperature, data,
                                                      data_col,adapt_flag, protection, CUEflag, voflag)

            model_data["Study"] = studyname
            model_data["Species"] = sp
            df_mode_data = pd.concat([df_mode_data, model_data], axis=0).reset_index(drop=True)

            df_temp = data.loc[0, ["Study", "Species", "Csource", "MATC", 'C:N', 'carbohydrate_MMM', 'protein_MMM', 'lignin_MMM',
                                   'lipid_MMM', 'carbonyl_MMM']].to_frame().T
            df_temp['CUE0'] = df['CUE'][0]
            df_temp['CUE_avg'] = df['CUE'].mean()
            df_temp['eta0'] = df['ETA'][0]
            df_temp['eta_avg'] = df['ETA'].mean()
            df_temp['DR0'] = df['DR'][0]
            df_temp['DR_avg'] = df['DR'].mean()

            new_df = pd.DataFrame()
            for col in perf_matrix.columns:
                new_col_r2 = col + '_r2'
                new_col_rmse = col + '_rmse'
                new_col_AIC = col + '_AIC'
                new_df[new_col_r2] = [perf_matrix.loc['r2', col]]
                new_df[new_col_rmse] = [perf_matrix.loc['rmse', col]]
                new_df[new_col_AIC] = [perf_matrix.loc['AIC', col]]

            mydf = pd.concat([df_temp, new_df], axis=1)
            df_perf_matrix = pd.concat([df_perf_matrix, mydf], axis=0).reset_index(drop=True)

    df_perf_matrix.to_excel('tables/'+ adapt_flag+'/Local_perf_matrix_'+fstr+'.xlsx', index=False)
    df_mode_data.to_excel('tables/'+ adapt_flag+'/Local_model_data_'+fstr+'.xlsx', index=False)

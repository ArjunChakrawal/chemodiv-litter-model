
import numpy as np
import matplotlib.pyplot as plt
import sol_ivp_ as svp
import seaborn as sns

with plt.style.context('ggplot'):
    # Capture the active color cycle from the temporary style
    palette = sns.color_palette()

# sns.palplot(palette)
plt.style.use('default')
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.edgecolor'] = (0.25, 0.25, 0.25)
plt.rcParams['text.color'] = 'white'

fixed_param, plant_data = svp.load_fix_par_and_data()

adapt_flag='N-Retention'
# adapt_flag ='Flexible CUE'
init_fracC = {'carbohydrate_gC': 0.0, 'protein_gC': 0.02, 'lignin_gC': 0.3, 'lipid_gC': 0.2, 'carbonyl_gC': 0.05}
init_fracC['carbohydrate_gC'] = 1 - init_fracC['protein_gC'] - \
    init_fracC['lignin_gC'] - init_fracC['lipid_gC'] - init_fracC['carbonyl_gC']

fixed_param['Inorg']=1e-5
a, b = fixed_param['a'], fixed_param['b']
guess_param = {'vh_max': 0.01, 'vp_max': 0.01, 'vlig': 0.008, 'vlip': 0.009, 'vCr': 0.01} # taken from median of estiamted pars

tsim = np.linspace(0, 365*2, 200)
Temperature = np.ones(len(tsim))*273.15

col = ['carbohydrate_gC', 'protein_gC', 'lignin_gC', 'lipid_gC', 'carbonyl_gC']
new_col = ['Carbohydrate [gC]', 'Protein [gC]', 'Lignin [gC]', 'Lipid [gC]', 'Carbonyl [gC]']
colors = palette[:len(new_col)]

lstyle = ['-','--',':','-.']
model_name = ['NPNE', 'NPWE', 'PWOE', 'PWOV']
title_str = ['Carbohydrate', 'Protein', 'Lignin', 'Lipid', 'Carbonyl']
Lt = np.arange(0, 0.5+0.01, 0.01)
CUEmax1 = svp.efficiency(np.ones(len(Lt))*3.8)
CUEmax2 = svp.efficiency(np.ones(len(Lt))*4.8)

fig2, ax2 = plt.subplots(1,4,figsize=(10,2.5),sharex=False, sharey=False)
ax2[0].fill_between(Lt, CUEmax1, CUEmax2, color='red', alpha=0.3, edgecolor=None)
ax2[0].fill_between(Lt, CUEmax1 * np.exp(-(Lt / a) ** b), CUEmax2 * np.exp(-(Lt / a) ** b), color='red', alpha=0.3, edgecolor=None)

ax2[1].plot(Lt, np.exp(-(Lt / a) ** b),linewidth=1.0,linestyle='-', color='red', alpha=0.5)
ax2[1].plot(Lt, np.ones(len(Lt)),linewidth=1.0,linestyle='-', color='red', alpha=0.5)

ax2[2].plot(Lt, 1-np.exp(-(Lt / a) ** b),linewidth=1.0,linestyle='-',color='red', alpha=0.5)
ax2[2].plot(Lt, np.ones(len(Lt)),linewidth=1.0,linestyle='-', color='red', alpha=0.5)

ax2[3].plot(Lt, np.ones(len(Lt)), linewidth=1.0,linestyle='-', color='red', alpha=0.5)


fig, ax = plt.subplots(2,5,figsize=(11.5,4.6),sharex=True, sharey=False)
fig.subplots_adjust(top=0.925, bottom=0.15, left=0.05,right=0.99,wspace=0.3, hspace=0.6)
a, b = fixed_param['a'], fixed_param['b']


fig3, ax3= plt.subplots(2,3,figsize=(11,5),sharex=True, sharey=False)
ax3=ax3.flatten()
k=0
sttr= "ABCDEFGH"
for ls, mdnam in zip(lstyle,model_name):
    if mdnam =='NPNE':
        df = svp.litter_decay_model(tsim, np.array(list(init_fracC.values())), guess_param, fixed_param,
                                     adapt_flag=adapt_flag, protection=False, CUEflag= False,voflag=False)

        L = df['lignin_gC']/df[col].sum(axis=1)
        p_func = p_func_Ch=p_func_P=p_func_lig = np.ones(len(df))

    elif mdnam =='NPWE':
        df = svp.litter_decay_model(tsim, np.array(list(init_fracC.values())), guess_param, fixed_param,
                                adapt_flag=adapt_flag, protection=False, CUEflag= True,voflag=False)
        L = df['lignin_gC']/df[col].sum(axis=1)
        p_func_Ch=p_func_P=p_func_lig = np.ones(len(df))
        p_func = np.exp(-(L / a)**b)
    elif mdnam =='PWOE':
        df = svp.litter_decay_model(tsim, np.array(list(init_fracC.values())), guess_param, fixed_param,
                                adapt_flag=adapt_flag, protection=True, CUEflag= True,voflag=False)
        L = df['lignin_gC']/df[col].sum(axis=1)
        p_func = p_func_Ch=p_func_P=np.exp(-(L / a)**b)
        p_func_lig = np.ones(len(df))
    elif mdnam=='PWOV':
        df = svp.litter_decay_model(tsim, np.array(list(init_fracC.values())), guess_param, fixed_param,
                                adapt_flag=adapt_flag, protection=True, CUEflag= True,voflag=True)
        L = df['lignin_gC']/df[col].sum(axis=1)
        p_func= p_func_Ch=p_func_P=np.exp(-(L / a)**b)
        p_func_lig = 1-np.exp(-(L / a)**b)
    else: 
        pass
        
    df = df.rename(columns=dict(zip(col, new_col)))
    for n, column in enumerate(new_col[0:5]):
        ax[0,n].plot(df['time']/365, df[column]/df[column].iloc[0], label=mdnam, linewidth=2,linestyle=ls, color=palette[n])
        # ax[1,0].plot(df['time'], df['DR'], label=mdnam, linewidth=2.5,linestyle=ls, color=palette[n])
        ax[0,n].set_title('('+sttr[n]+') '+title_str[n],fontsize=12, color='black')
    
    
    ax[1,0].plot(df['time']/365, df['ETA'], label=mdnam, linewidth=2,linestyle=ls, color='black')
    ax[1,0].set_title(r'(F) $\eta$',fontsize=12, color='black')

    ax[1,1].plot(df['time']/365, df['CUE'], label=mdnam, linewidth=2.5,linestyle=ls, color='black')
    ax[1,1].set_title('(G) CUE',fontsize=12, color='black')
    
    vh = p_func_Ch*guess_param['vh_max']
    ax[1,2].plot(df['time']/365, vh, label=mdnam, linewidth=2.5,linestyle=ls, color='black')
    ax[1,2].set_title(r'(H) $v_{C_h}=v_{P}$',fontsize=12, color='black')
    
    vLg = p_func_lig*guess_param['vlig']
    ax[1,3].plot(df['time']/365, vLg, label=mdnam, linewidth=2.5,linestyle=ls, color='black')
    ax[1,3].set_title(r'(I) $v_{L_g}$',fontsize=12, color='black')
    vLp= np.ones(len(df))*guess_param['vlip']
    vCr= np.ones(len(df))*guess_param['vCr']
    ax[1,4].plot(df['time']/365, vLp, label=mdnam, linewidth=2.5,linestyle=ls, color='black')
    ax[1,4].plot(df['time']/365, vCr, label=mdnam, linewidth=2.5,linestyle=ls, color=[.7,.7,.7])
    ax[1,4].set_title(r'(J) $v_{L_p}, v_{C_n}$',fontsize=12, color='black')

    ax2[0].plot(L, df['CUE'], label=mdnam, linewidth=2.5,linestyle=ls, color='black')
    ax2[0].set_title('(A) CUE',fontsize=12, color='black')
    
    ax2[1].plot(L, p_func_Ch, label=mdnam, linewidth=2.5,linestyle=ls, color='black')
    ax2[1].set_title(r'(B) $p_{C_h}=p_{P}$',fontsize=12, color='black')
    
    ax2[2].plot(L, p_func_lig, label=mdnam, linewidth=2.5,linestyle=ls, color='black')
    ax2[2].set_title(r'(C) $p_{L_g}$',fontsize=12, color='black')

    ax2[3].plot(L, np.ones(len(df)), label=mdnam, linewidth=2.5,linestyle=ls, color='black')
    ax2[3].set_title(r'(D) $p_{L_p}=p_{C_n}=1$',fontsize=12, color='black')
    
    ax3[0].plot(df['time']/365, df['totCg'], label=mdnam, linewidth=2,linestyle=ls, color='black')
    ax3[1].plot(df['time']/365, df['totNg'], label=mdnam, linewidth=2,linestyle=ls, color='black')
    ax3[2].plot(df['time']/365, df['Lignin [gC]']/df['totCg'], label=mdnam, linewidth=2,linestyle=ls, color='black')
    ax3[3].plot(df['time']/365, df['DR'], label=mdnam, linewidth=2,linestyle=ls, color='black')
    
    CUE_max = [svp.efficiency(np.array([dr])) for dr in df['DR'].values]
    ax3[4].plot(df['time']/365, CUE_max, label=mdnam, linewidth=2,linestyle=ls, color='black')

    ax3[5].plot(df['time']/365, df['CUE_star']- df['CUE'], label=mdnam, linewidth=2,linestyle=ls, color='black')
    k=k+1

# fix axes for ax3
for i in range(6):
    ax3[i].set_xlabel(r'Time [Y]', fontsize=10)
    ax3[i].tick_params(axis='both', labelsize=9)

ax3[0].set_ylabel(r'total C [g]', fontsize=10)
ax3[1].set_ylabel(r'total N [g]', fontsize=10)
ax3[2].set_ylabel(r'Lignin C fraction [gC/gC]', fontsize=10)
ax3[3].set_ylabel(r'Degree of reduction', fontsize=10)
ax3[4].set_ylabel(r'CUE$_{max}$', fontsize=10)
ax3[5].set_ylabel(r'CUE$^*$ - CUE', fontsize=10)

ax3[0].legend(fontsize=9, loc="upper right", frameon=False, labelcolor='black')
abcd= ['A','B','C','D','E','F']
for i in range(6):
    ax3[i].set_title('('+abcd[i]+')',fontsize=12, color='black')
    ax3[i].grid(False)

fig3.tight_layout()
fig3.savefig('figs/FigureS4.jpg', bbox_inches='tight', dpi=600)


# fix axes for ax2
ax2[0].set_title('(A)',fontsize=12)
ax2[1].set_title(r'(B)',fontsize=12)
ax2[2].set_title(r'(C)',fontsize=12)
ax2[3].set_title(r'(D)',fontsize=12)



for i in range(4):
    ax2[i].set_xlabel(r'Lignin C fraction ($L$)', fontsize=10)
    ax2[i].set_xticks([0, 0.25,  0.5])
    ax2[i].tick_params(axis='both', labelsize=9)
    ax2[i].grid(False)
    # ax2[i].set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5'])
ax2[3].set_ylim(0.95,1.05)
ax2[3].set_yticks([0.9,1,1.1])
ax2[0].set_ylabel('CUE',fontsize=14)
ax2[1].set_ylabel(r'$p_{C_h}=p_{P}$',fontsize=14)
ax2[2].set_ylabel(r'$p_{L_g}$',fontsize=14)
ax2[3].set_ylabel(r'$p_{L_p}=p_{C_n}=1$',fontsize=14)

fig2.tight_layout()
fig2.savefig('figs/Figure2.jpg', bbox_inches='tight', dpi=600)

# fix axes for ax

# ax[0,n].set_xlabel('Time [Y]',fontsize=11)
for i in range(5):
    ax[1,i].set_xlabel('Time [Y]',fontsize=11)

ax[0,0].set_ylabel(r'C/C$_0$',fontsize=12)
ax[1,2].set_ylim(bottom =0.003)
ax[1,3].set_ylim(bottom =0.0)
ax[1,4].set_ylim(bottom=0.008,top= 0.012)
ax[1,4].set_yticks([0.008,0.009,0.01,0.011,0.012])
# Add legend handles for v_{L_p} and v_{C_n} to ax[1,4]
line1 = ax[1,4].plot(np.nan, np.nan, label=r"$v_{L_p}$", linewidth=2, linestyle='-', color='black')
line2 = ax[1,4].plot(np.nan, np.nan, label=r"$v_{C_n}$", linewidth=2, linestyle='-', color=[.7, .7, .7])
ax[1,4].legend(handles=[line1[0], line2[0]], fontsize=12,ncol=1, loc="upper right", frameon=False, labelcolor='black')
ax[1,0].set_ylim(bottom=0)
ax[1,1].set_ylim(bottom=0)
# ax[0,2].legend(fontsize=12, frameon=True,loc='upper center', bbox_to_anchor=(1.5, 1))
pts=[]
for ls, mdnam in zip(lstyle,model_name):
    pt=ax[0,0].plot(np.nan, np.nan,label=mdnam, linewidth=2,linestyle=ls, color='black')
    pts.append(pt[0])
ax[0,0].legend(pts,model_name,fontsize=9, loc="upper right", frameon=False, labelcolor='black')

for axx in ax.flatten():
    axx.tick_params(axis='both', labelsize=10,direction='out')
    axx.grid(visible=False, which='both')
    
# ax[0,2].legend(fontsize=11, frameon=False,loc='best',ncol=1)  

fig.tight_layout()
fig.savefig('figs/Figure3.jpg', dpi=600, bbox_inches='tight')

# fig.savefig('figs/Figure3.svg', bbox_inches='tight')
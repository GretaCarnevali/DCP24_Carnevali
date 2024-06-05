# Data Manipulation and Analysis
import pandas as pd
import numpy as np
import scipy
import statsmodels
import pingouin
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from scipy.stats import spearmanr
from scipy.integrate import quad

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.colors as cs
import matplotlib.cbook as cbook
from matplotlib import cm

#Clustering
from sklearn.cluster import KMeans

#
from tqdm.notebook import tqdm

#eeg routines
import eeg_routines
from eeg_routines import parcellation

coordinates, name, Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10 = eeg_routines.parcellation()
Q = {'Q1': Q1,
'Q2': Q2,
'Q3': Q3,
'Q4': Q4,
'Q5': Q5,#central channels
'Q6': Q6,
'Q7': Q7,
'Q8': Q8,
'Q9': Q9,
'Q10': Q10,#occipital channels
}

#Data Preprocessing
import data_preprocessing
from data_preprocessing import data_dictionary

erpData=data_preprocessing.data_dictionary()

#Feature extraction
import ERP_feature_extraction
from ERP_feature_extraction import p_times_windows_msec_bonf

p_GROUPF_OCC_msec_bonf,p_corrected_GROUPF_OCC_msec_bonf,effect_size_GROUPF_OCC,df_p_posthoc_GROUPF_OCC_msec_bonf,min_values_GROUPF_OCC_msec_bonf,max_values_GROUPF_OCC_msec_bonf,df_GROUPF_OCC_msec_bonf= ERP_feature_extraction.p_times_windows_msec_bonf(parcel='Q10',
                                                                                                                                                            dictionary='GROUPS-FEMALES',
                                                                                                                                                            posthoc=True,
                                                                                                                                                            n_clusters=3)


def p_plot_msec (dictionary='GROUPS-FEMALES',
               parcel='Q10',
                alpha_eb = 0.5,
                xlim=[0, 0.7],
                ylim=[-8,5],
                FONTSIZE = 16,
                min_values=min_values_GROUPF_OCC_msec_bonf,
                max_values=max_values_GROUPF_OCC_msec_bonf,
                nan_withincat=[[],[81],[19,27]],
                zscore=2):

        times = np.linspace(-300, 1000, 666)
        var = erpData[dictionary]
        data = var['data']
        colors = var['colors']
        colors_line=['lightskyblue','navy','darkorchid','lightpink','red','pink','green','yellow','grey','orange','lightgreen','lightblue']
        labels = var['name']
        alpha_ERP = [1] * len(data)
        linestyle = var['linestyle']

        plt_ERP,axERP=plt.subplots(1,1,figsize=(9,4))
        for d in tqdm(range(len(data))): 
                axERP.plot(times[times>0],
                        np.mean(np.mean(np.stack(data[d], axis=1)[Q[parcel]], axis=0), axis=0)[times>0], 
                        c=colors[d],
                        linewidth=4,
                        label=f'{labels[d]} [{len(data[d])}]',
                        alpha=alpha_ERP[d],
                        linestyle=linestyle[d])
                axERP.errorbar(times[times>0],
                                np.mean(np.mean(np.stack(data[d], axis=1)[Q[parcel]], axis=0), axis=0)[times>0],
                                scipy.stats.sem(np.mean(np.stack(data[d], axis=1)[Q[parcel]], axis=0), axis = 0, ddof = 1)[times>0],#compute the standard error across the subcategory of the mean values wrt channels
                                alpha=alpha_eb, c=colors[d])
                
        axERP.set_xlim(xlim[0], xlim[1])
        axERP.set_ylim(ylim[0], ylim[1])
        axERP.set_xlabel('Time [ms]', fontsize=FONTSIZE-4)
        axERP.axvline(200, c='k', linestyle=':', alpha=.75)
        axERP.set_ylabel(f'Voltage potential [$\mu$V] \n {parcel} electrode cluster', fontsize=FONTSIZE-3)
        if parcel == 'Q5':
                axERP.set_title(f'CENTRAL CHANNELS \n (FC1, FCz, FC2, C1, Cz, C2)', fontsize=FONTSIZE-2)
        elif parcel == 'Q10':
                axERP.set_title(f'OCCIPITAL CHANNELS  \n (PO7, PO8, O1, Oz, O2)', fontsize=FONTSIZE-2)


        if min_values!=0:
                for i in range(len(min_values)):
                        axERP.axvspan(min_values[i],max_values[i], color=colors_line[i], alpha=0.2, label=i)
                        #axERP_filt.axvspan(min_values[i],max_values[i], color=colors_line[i], alpha=0.2, label=i)
        axERP.legend(loc="upper right",fontsize="8")
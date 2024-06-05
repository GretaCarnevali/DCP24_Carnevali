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

#Feature extraction
import ERP_feature_extraction
from ERP_feature_extraction import p_times_windows_msec_bonf

p_GROUPF_OCC_msec_bonf,p_corrected_GROUPF_OCC_msec_bonf,effect_size_GROUPF_OCC,df_p_posthoc_GROUPF_OCC_msec_bonf,min_values_GROUPF_OCC_msec_bonf,max_values_GROUPF_OCC_msec_bonf,df_GROUPF_OCC_msec_bonf= ERP_feature_extraction.p_times_windows_msec_bonf(parcel='Q10',
                                                                                                                                                            dictionary='GROUPS-FEMALES',
                                                                                                                                                            posthoc=True,
                                                                                                                                                            n_clusters=3)

def post_hoc_plot(df_p_posthoc=df_p_posthoc_GROUPF_OCC_msec_bonf,
                           min_values=min_values_GROUPF_OCC_msec_bonf,
                           max_values=max_values_GROUPF_OCC_msec_bonf,
                           xlim=[0,300]):
    """ 
    This function plots the set of p-values computed during post-hoc analysis; each set corresponds to a pairwise comparison between 
    2 of the classes of the dictionary and is visualized as scatterplot.
    This function provides also a refence for the significance level by additionally displaying also 3 significance thresholds corresponding
    to alpha=0.05, 0.001, 0.0001 

        Parameters
        ----------
        df_p_posthoc : pandas dataframe 
            Dataframe computed by the function: p_times_windows_v5/p_times_windows_v8
        min_values: float array
            Array of lower boundaries of relevant time windows, computed by the function: p_times_windows_v5/p_times_windows_v8
        max_values: float array
            Array of lower boundaries of relevant time windows, computed by the function: p_times_windows_v5/p_times_windows_v8
        xlim: float array
            Boundaries for temporal x-axes, usually set to visualize just a single relevant window

        Notes
        ------
        With respect to functions: p_times_windows_v5/p_times_windows_v8, here there are no parameters related to electrode cluster or dictionary since the name of
        df_p_posthoc should already bring information about (as suggested previously); this is done also to avoid useless mismatches and incoherences in the following analysis
    """
    
    colors_line=['lightskyblue','navy','darkorchid','lightpink','red','pink','green','yellow','grey','orange','lightgreen','lightblue']
    palette=['lightgrey','dimgrey','darkslategrey' ]
    sns.set_style("white")

    fig, ax1 = plt.subplots(figsize=[5,4])
    sns.scatterplot(data=df_p_posthoc, x=df_p_posthoc["time"],y=df_p_posthoc["p12"],c=palette[0],ax=ax1)
    sns.scatterplot(data=df_p_posthoc, x=df_p_posthoc["time"],y=df_p_posthoc["p13"],c=palette[1],ax=ax1)
    sns.scatterplot(data=df_p_posthoc, x=df_p_posthoc["time"],y=df_p_posthoc["p23"],c=palette[2],ax=ax1)

    # significance thresholds
    ax1.axhline(0.05, c='lightgrey', alpha=.75,label='*')
    ax1.axhline(0.001, c='darkgrey', alpha=.50,label='**')
    ax1.axhline(0.0001, c='dimgrey', alpha=.50,label='***')
    ax1.axvline(200, c='k', linestyle=':', alpha=.75)

    for i in range(len(min_values)):
        plt.axvspan(min_values[i],max_values[i], color=colors_line[i], alpha=0.2, label=i)
    ax1.set_xlabel('times (ms)')
    ax1.set_ylabel('p_posthoc')
    ax1.set_yscale('log')
    ax1.legend(labels=["HS vs SCD","HS vs MCI","SCD vs MCI","*","**","***"],fontsize=10)

    #Add a second y-axes reporting values in microVolt
    ax2 = ax1.twinx()
    ax2.set_ylabel('ERP Voltage [$\mu$V]')  # we already handled the x-label with ax1
    #ax2.plot(df_p_posthoc["time"], df_p_posthoc["ERP1"],'--',alpha=0.5)
    plt.xlim(xlim)

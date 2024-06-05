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

import ERP_feature_extraction
from ERP_feature_extraction import p_times_windows_msec_bonf
p_GROUPF_OCC_msec_bonf,p_corrected_GROUPF_OCC_msec_bonf,effect_size_GROUPF_OCC,df_p_posthoc_GROUPF_OCC_msec_bonf,min_values_GROUPF_OCC_msec_bonf,max_values_GROUPF_OCC_msec_bonf,df_GROUPF_OCC_msec_bonf= ERP_feature_extraction.p_times_windows_msec_bonf(parcel='Q10',
                                                                                                                                                            dictionary='GROUPS',
                                                                                                                                                            posthoc=True,
                                                                                                                                                            n_clusters=3)
p_GROUPF_CEN_msec_bonf,p_corrected_GROUPF_CEN_msec_bonf,effect_size_GROUPF_CEN,df_p_posthoc_GROUPF_CEN_msec_bonf,min_values_GROUPF_CEN_msec_bonf,max_values_GROUPF_CEN_msec_bonf,df_GROUPF_CEN_msec_bonf= ERP_feature_extraction.p_times_windows_msec_bonf(parcel='Q5',
                                                                                                                                                            dictionary='GROUPS',
                                                                                                                                                            posthoc=True,
                                                                                                                                                            n_clusters=3)
def out_detection(df=df_GROUPF_OCC_msec_bonf,thr=[3,3.5],feature=['integral feat','peak feat']):
    '''
    This function performs outliers detection. 
    It performs a separate analysis for each extracted feature, for each zscore value and for each temporal window;
    moreover, the outliers are identified within each class of the dictionary.
    std_mean_v4 also plots an histogram showing on the x-axes the ID (based of the order of registration), on y-axes the sum 
    (considering all the feature selected) of NaN values; the histogram also reports a hue variable corresponding to the different 
    zscore values (foe each patient ID, there is a number of columns = number of tested zscores)

    Parameters
    ----------
    df_GROUP_OCC : pandas dataframe 
        Dataframe computed by the function: p_times_windows_v5/p_times_windows_v8
    thr: float array
        Array of zscore values wanted to be tested; So it's possible to test multiple values and so discard a higher/lower number of outliers
    feature: string array
        Array of features for which outliers are identified
    
    Returns
    ----------
    df_filtered: pandas dataframe
        Dataframe with the same columns of df_GROUP_OCC plus an additional one that reports the values of zscore; 
        Note: the length of this dataframe is equal to len(df_GROUP_OCC) x number of tested zscores
    df_nan: pandas dataframe
        Submask of df_filtered that contains only the rows with at least one NaN value 

    '''

    df_filtered=df.copy(deep=True)
    df_filtered['zscore']=np.ones(len(df))*thr[0]

    for i in range(len(thr)-1):
        df['zscore']=np.ones(len(df))*thr[i+1]
        df_filtered=pd.concat([df_filtered,df])
    df_filtered=df_filtered.set_index(np.arange(len(df_filtered)))  

    for t_wind in pd.unique(df_filtered['t_window']):
        for cat in pd.unique(df_filtered['category']):
            for feature_curr in feature:
                for thr_curr in thr:

                    df_curr=df_filtered[df_filtered['t_window']==t_wind][df_filtered['category'] == cat][df_filtered['zscore'] == thr_curr]
                    df_sub = df_filtered.loc[df_curr.index, feature_curr]
                    lim=np.abs(stats.zscore(df_sub)) < thr_curr
                    # replace outliers with nan
                    df_filtered.loc[ df_filtered[ df_filtered['t_window']==t_wind][ df_filtered['category']==cat][df_filtered['zscore'] == thr_curr].index, feature_curr + '_outlier'] = df_sub.where(lim, np.nan)
    
    df_nan=df_filtered[df_filtered.isna().any(axis=1)]

    #histogram plot
    if (len(feature)==1):
        result_nan=pd.concat([df_nan[df_nan[feature[0]+ '_outlier'].isna()]])
    if (len(feature)==2):
        result_nan=pd.concat([df_nan[df_nan[feature[0]+ '_outlier'].isna()],df_nan[df_nan[feature[1]+'_outlier'].isna()]])
    if (len(feature)==3):
        result_nan=pd.concat([df_nan[df_nan[feature[0]+ '_outlier'].isna()],df_nan[df_nan[feature[1]+'_outlier'].isna()],df_nan[df_nan[feature[2]+'_outlier'].isna()]])

    fig,ax=plt.subplots(figsize=[18,4])
    palette=['dimgrey','lightgrey','grey']
    ax.set_title('ID outliers detection considering all the features extracted')
    ax.set_xlabel('Patients ID in order of registration')
    sns.histplot(data=result_nan,hue='zscore',x="ID_reg",bins=len(pd.unique(df_filtered['ID_reg'])),multiple="stack",palette=palette)

    return df_filtered,df_nan

df_GROUPF_OCC_filt,df_nan_GROUPF_OCC= out_detection(df=df_GROUPF_OCC_msec_bonf,thr=[2],feature=['max feat','min feat','integral feat'])
df_GROUPF_CEN_filt,df_nan_GROUPF_CEN= out_detection(df=df_GROUPF_CEN_msec_bonf,thr=[2],feature=['max feat','min feat','integral feat'])

def out_feature(P1_OCC=2,N1_OCC=1,P2_CEN=2,df_OCC=df_nan_GROUPF_OCC,df_CEN=df_nan_GROUPF_CEN):

    #occipital channels
    df_OCC['min feat_outlier'][df_OCC['t_window']==P1_OCC]=df_OCC['min feat_outlier'][df_OCC['t_window']==P1_OCC].fillna(0)
    df_OCC['max feat_outlier'][df_OCC['t_window']==N1_OCC]=df_OCC['max feat_outlier'][df_OCC['t_window']==N1_OCC].fillna(0)
    df_OCC=df_OCC[df_OCC.isna().any(axis=1)]

    #central channels
    df_CEN['min feat_outlier'][df_CEN['t_window']==P2_CEN]=df_CEN['min feat_outlier'][df_CEN['t_window']==P2_CEN].fillna(0)
    df_CEN['max feat_outlier'][df_CEN['t_window']==P2_CEN]=df_CEN['max feat_outlier'][df_CEN['t_window']==P2_CEN].fillna(0)
    df_CEN=df_CEN[df_CEN.isna().any(axis=1)]
    
    #
    df_nan_GROUP=pd.concat([df_OCC[df_OCC['t_window']==P1_OCC],df_OCC[df_OCC['t_window']==N1_OCC],df_CEN[df_CEN['t_window']==P2_CEN]])
    df_nan_GROUP=df_nan_GROUP.set_index(np.arange(len(df_nan_GROUP)))
    df_nan_GROUP

    return df_nan_GROUP



def out_removal(outliers=[59,83],df_OCC_filt=df_GROUPF_OCC_filt,df_CEN_filt=df_GROUPF_CEN_filt):

    #OCCIPITAL
    for out in outliers:
        df_OCC_filt = df_OCC_filt[df_OCC_filt.apply(lambda x: x['ID_reg'] != out, axis=1)]

    #CENTRAL
    for out in outliers:
        df_CEN_filt = df_CEN_filt[df_CEN_filt.apply(lambda x: x['ID_reg'] != out, axis=1)]

    return df_OCC_filt,df_CEN_filt

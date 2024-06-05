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

import pickle

def data_dictionary():

    savePath="C:/Users/greca/Desktop/SantAnna/Lab Training/notebook/"

    file_name = savePath + 'df-gender-greta_conOutlier.pkl'

    try:
        with open(file_name, 'rb') as f:
            gretaData = pickle.load(f)
    except EOFError:
        print("Error")

    ### Load 3CVT DataSet:
    df=gretaData
    ### Creation of masks:
    CTR = df['GROUPS'] == 'CTR'
    SCD = df['GROUPS'] == 'SCD'
    MCI = df['GROUPS'] == 'MCI'
    FEMALE = df['SEX'] == 'F'
    MALE = df['SEX'] == 'M'
    for feat in ['3CVT_ACC', '3CVT_RT', '3CVT_FMEASURE']:
        df[f'{feat}_CAT'] = df[feat]>=np.median(df[feat])
        df[f'{feat}_CAT'] = df[f'{feat}_CAT'].replace(False, 'LOW')
        df[f'{feat}_CAT'] = df[f'{feat}_CAT'].replace(True, 'HIGH')
    ACCLOW = df['3CVT_ACC_CAT'] == 'LOW'
    ACCHIGH = df['3CVT_ACC_CAT'] == 'HIGH'
    RTLOW = df['3CVT_RT_CAT'] == 'LOW'
    RTHIGH = df['3CVT_RT_CAT'] == 'HIGH'
    FMLOW = df['3CVT_FMEASURE_CAT'] == 'LOW'
    FMHIGH = df['3CVT_FMEASURE_CAT'] == 'HIGH'

    ## AGE EFFECT
    for feat in ['C_AGE']:
        df[f'{feat}_CAT'] = df[feat]>=np.median(df[feat])
        df[f'{feat}_CAT'] = df[f'{feat}_CAT'].replace(False, 'OLDMINUS')
        df[f'{feat}_CAT'] = df[f'{feat}_CAT'].replace(True, 'OLDPLUS')
    OLDMINUS = df['3CVT_ACC_CAT'] == 'LOW'
    OLDPLUS = df['3CVT_ACC_CAT'] == 'HIGH'

    ### Creation of an unique dictionary:
    erpData = {}

    erpData['GROUPS'] = {'colors': ['blue', 'orange', 'red'],
            'ID_reg': [list(df['Numero (in ordine di registrazione)'][CTR]), list(df['Numero (in ordine di registrazione)'][SCD]), list(df['Numero (in ordine di registrazione)'][MCI])],
            'name': ['HS','SCD', 'MCI'],
            'title': 'GROUPS',
            'data': [list(df['ERP'][CTR]), list(df['ERP'][SCD]), list(df['ERP'][MCI])],
            'sex':[list(df['SEX'][CTR]), list(df['SEX'][SCD]), list(df['SEX'][MCI])],
            'rt': [df['3CVT_RT'][CTR],df['3CVT_RT'][SCD],df['3CVT_RT'][MCI]],
            'age_cont':[list(df['C_AGE'][CTR]), list(df['C_AGE'][SCD]), list(df['C_AGE'][MCI])],
            'age_cat':[list(df['C_AGE_CAT'][CTR]), list(df['C_AGE_CAT'][SCD]), list(df['C_AGE_CAT'][MCI])],
            'ylim_min':[-8,-1.5],
            'ylim_max':[3,3],
            'xlim_min':[0,0],
            'xlim_max':[1000,1000],
            'linestyle': ['solid','solid','solid'],
            }

    erpData['GROUPS-MALES'] = {'colors': ['blue', 'orange', 'red'],
            'ID_reg': [list(df['Numero (in ordine di registrazione)'][CTR][MALE]), list(df['Numero (in ordine di registrazione)'][SCD][MALE]), list(df['Numero (in ordine di registrazione)'][MCI][MALE])],
            'name': ['HS','SCD', 'MCI'],
            'title': 'GROUPS - MALES',
            'data': [list(df['ERP'][CTR][MALE]), list(df['ERP'][SCD][MALE]), list(df['ERP'][MCI][MALE])],
            'sex':[list(df['SEX'][CTR][MALE]), list(df['SEX'][SCD][MALE]), list(df['SEX'][MCI][MALE])],
            'rt': [df['3CVT_RT'][CTR][MALE],df['3CVT_RT'][SCD][MALE],df['3CVT_RT'][MCI][MALE]],
            'age_cont':[list(df['C_AGE'][CTR][MALE]), list(df['C_AGE'][SCD][MALE]), list(df['C_AGE'][MCI][MALE])],
            'age_cat':[list(df['C_AGE_CAT'][CTR][MALE]), list(df['C_AGE_CAT'][SCD][MALE]), list(df['C_AGE_CAT'][MCI][MALE])],
            'ylim_min':[-8,-1.5],
            'ylim_max':[3,3],
            'xlim_min':[0,0],
            'xlim_max':[1000,1000],
            'linestyle': ['solid','solid','solid'],
            }

    erpData['GROUPS-FEMALES'] = {'colors': ['blue', 'orange', 'red'],
            'ID_reg': [list(df['Numero (in ordine di registrazione)'][CTR][FEMALE]), list(df['Numero (in ordine di registrazione)'][SCD][FEMALE]), list(df['Numero (in ordine di registrazione)'][MCI][FEMALE])],
            'name': ['HS','SCD', 'MCI'],
            'title': 'GROUPS - FEMALES',
            'data': [list(df['ERP'][CTR][FEMALE]), list(df['ERP'][SCD][FEMALE]), list(df['ERP'][MCI][FEMALE])],
            'sex':[list(df['SEX'][CTR][FEMALE]), list(df['SEX'][SCD][FEMALE]), list(df['SEX'][MCI][FEMALE])],
            'rt': [df['3CVT_RT'][CTR][FEMALE],df['3CVT_RT'][SCD][FEMALE],df['3CVT_RT'][MCI][FEMALE]],
            'age_cont':[list(df['C_AGE'][CTR][FEMALE]), list(df['C_AGE'][SCD][FEMALE]), list(df['C_AGE'][MCI][FEMALE])],
            'age_cat':[list(df['C_AGE_CAT'][CTR][FEMALE]), list(df['C_AGE_CAT'][SCD][FEMALE]), list(df['C_AGE_CAT'][MCI][FEMALE])],
            'ylim_min':[-8,-1.5],
            'ylim_max':[3,3],
            'xlim_min':[0,0],
            'xlim_max':[1000,1000],
            'linestyle': ['solid','solid','solid'],
            }

    ##Additional:
    erpData['PATIENTS-MALES'] = {'colors': ['orange', 'red'],
            'name': ['SCD', 'MCI'],
            'title': 'PATIENTS-MALES',
            'data': [list(df['ERP'][SCD][MALE]), list(df['ERP'][MCI][MALE])],
            'sex':[list(df['SEX'][SCD][MALE]), list(df['SEX'][MCI][MALE])],
            'rt': [df['3CVT_RT'][SCD][MALE],df['3CVT_RT'][MCI][MALE]],
            'ylim_min':[-8,-1.5],
            'ylim_max':[3,3],
            'xlim_min':[0,0],
            'xlim_max':[1000,1000],
            'linestyle': ['solid','solid','solid'],
            }


    erpData['PATIENTS-FEMALES'] = {'colors': ['orange', 'red'],
            'name': ['SCD', 'MCI'],
            'title': 'PATIENTS-FEMALES',
            'data': [list(df['ERP'][SCD][FEMALE]), list(df['ERP'][MCI][FEMALE])],
            'sex':[list(df['SEX'][SCD][FEMALE]), list(df['SEX'][MCI][FEMALE])],
            'rt': [df['3CVT_RT'][SCD][FEMALE],df['3CVT_RT'][MCI][FEMALE]],
            'ylim_min':[-8,-1.5],
            'ylim_max':[3,3],
            'xlim_min':[0,0],
            'xlim_max':[1000,1000],
            'linestyle': ['solid','solid','solid'],
            }

    return erpData 

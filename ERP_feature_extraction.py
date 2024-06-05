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


#Data preprocessing
import data_preprocessing
from data_preprocessing import data_dictionary

erpData=data_preprocessing.data_dictionary()

#This function works for dataset processed with Band Pass Filter 1-45 Hz (not 0.1 - 45 Hz); time variable in ms (not sec)
def p_times_windows_msec_bonf(parcel='Q10',
                    dictionary='GROUPS',
                    posthoc=True,
                    n_clusters=3):

    times = np.linspace(-300, 1000, 666)
    feat = dictionary
    var = erpData[feat]
    data = var['data']
    labels = var['name']
    alpha_stat = 0.01

    ###Group analysis
    if len(data) == 3:
        p = []
        d = []
        num_comparisons = sum(times > 2)
        for i in range(num_comparisons):
            group1 = np.vstack(np.stack(data[0], axis=1)[Q[parcel]])[:, times > 2][:, i] 
            group2 = np.vstack(np.stack(data[1], axis=1)[Q[parcel]])[:, times > 2][:, i]
            group3 = np.vstack(np.stack(data[2], axis=1)[Q[parcel]])[:, times > 2][:, i]
            f, p_ = scipy.stats.kruskal(group1, group2, group3) #reminder: Kruskal-Wallis H-test, non-param version of ANOVA
            p += [p_] #The p-value for the test using the assumption that H has a chi square distribution. 
            #The p-value returned is the survival function of the chi square distribution evaluated at H.
            n1, n2, n3 = len(group1), len(group2), len(group3)
            mean1, mean2, mean3 = np.mean(group1), np.mean(group2), np.mean(group3)
            pooled_std = np.sqrt(((n1-1)*np.var(group1) + (n2-1)*np.var(group2) + (n3-1)*np.var(group3)) / (n1+n2+n3-3))
            effect_size = (mean1-mean2)/pooled_std if f < 1 else (mean1-mean3)/pooled_std
            d += [effect_size]

    if len(data) == 2:
        p = []
        d = []
        num_comparisons = sum(times > 2)
        for i in range(num_comparisons):
            group1 = np.vstack(np.stack(data[0], axis=1)[Q[parcel]])[:, times > 2][:, i]
            group2 = np.vstack(np.stack(data[1], axis=1)[Q[parcel]])[:, times > 2][:, i]
            f, p_ = scipy.stats.kruskal(group1, group2)
            p += [p_]
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            pooled_std = np.sqrt(((n1-1)*np.var(group1) + (n2-1)*np.var(group2)) / (n1+n2-2))
            effect_size = (mean1-mean2)/pooled_std
            d += [abs(effect_size)]

    p = np.array(p)
    _, Pcorr, _, _ = statsmodels.stats.multitest.multipletests(p, alpha=alpha_stat, method='bonferroni', is_sorted=False, returnsorted=False)
    p = Pcorr
    alphalevel = alpha_stat
    p_corrected = p < alphalevel

    ### HIGHLIGHT THE RELEVANT P INTERVALS and COMPUTE THE POST-HOC ANALYSIS
    p_posthoc = []

    for i in range(len(p_corrected)): 

        if p_corrected[i] > 0:
            ## POST-HOC
            if len(data) == 3:
                group1 = np.vstack(np.stack(data[0], axis=1)[Q[parcel]])[:, times > 2][:, i]
                group2 = np.vstack(np.stack(data[1], axis=1)[Q[parcel]])[:, times > 2][:, i]
                group3 = np.vstack(np.stack(data[2], axis=1)[Q[parcel]])[:, times > 2][:, i]

                s, p_12 = scipy.stats.mannwhitneyu(group1, group2) #Perform the Mann-Whitney U rank test on two independent samples.
                #The Mann-Whitney U test is a nonparametric test of the null hypothesis that the distribution underlying sample x is the same
                #as the distribution underlying sample y. It is often used as a test of difference in location between distributions.
                s, p_13 = scipy.stats.mannwhitneyu(group1, group3)
                s, p_23 = scipy.stats.mannwhitneyu(group2, group3)
                p_posthoc.append([i,
                                    np.mean(np.mean(np.stack(data[0], axis=1)[Q[parcel]], axis=0), axis=0)[times>0][i],
                                    np.mean(np.mean(np.stack(data[1], axis=1)[Q[parcel]], axis=0), axis=0)[times>0][i],
                                    np.mean(np.mean(np.stack(data[2], axis=1)[Q[parcel]], axis=0), axis=0)[times>0][i],
                                    times[times>2][i],
                                    (p_12*3*len(p_corrected)),
                                    (p_13*3*len(p_corrected)),
                                    (p_23*3*len(p_corrected)),
                                    (p_12*3*len(p_corrected)) < alpha_stat,
                                    (p_13*3*len(p_corrected)) < alpha_stat,
                                    (p_23*3*len(p_corrected)) < alpha_stat])

            df_p_posthoc = pd.DataFrame(p_posthoc, columns=['i','ERP1','ERP2','ERP3', 'time', 'p12', 'p13', 'p23', 'p12L', 'p13L', 'p23L'])

    if not p_corrected.any():
        print(f'No significative temporal windows in {feat}')
        df_p_posthoc=0
        min_values= 0
        max_values = 0
        df_feature=0
    else:
        print(f'Significative temporal windows in {feat} for the electrode cluster {parcel}')
        X = times[times>2][p_corrected] #time istants in which p_corrected is >0, so temporal information of the relevant p-values corrected with Bonf
        
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X.reshape(-1, 1))  # Reshape to make it compatible with KMeans
        labels = kmeans.labels_#labels each point, the labels are just 0,1,2
        print(f"\nFor n_clusters = {n_clusters}:")
        min_values = []
        max_values = []
        len_values = []
        summary=[]
        
        for cluster_label in range(n_clusters):
            cluster_data = X[labels == cluster_label] #get all the time points corresponding to that clusters
            min_cluster = np.min(cluster_data, axis=0)
            max_cluster = np.max(cluster_data, axis=0)
            min_values.append(min_cluster)
            max_values.append(max_cluster)
            len_values.append(len(cluster_data))
            if len(data) == 3:
                if posthoc:
                    G1 = erpData[feat]['name'][0]#name is the string associated to the 'subcategory' e.g HS, LOW ecc
                    G2 = erpData[feat]['name'][1]
                    G3 = erpData[feat]['name'][2]
                    print(f"-------------posthoc with cluster label {cluster_label}------with tmin {min_cluster} and tmax {max_cluster}------and len {len(cluster_data)}")
                    #here print the results of post-hoc analysis regarding 
                    print(f"{G1}vs{G2}: {sum(df_p_posthoc['p12L'][labels == cluster_label])} of {len(df_p_posthoc['p12L'][labels == cluster_label])} | {sum(df_p_posthoc['p12L'][labels == cluster_label])/len(df_p_posthoc['p12L'][labels == cluster_label])} | avg p={np.round(np.mean(df_p_posthoc['p12'][labels == cluster_label][df_p_posthoc['p12'][labels == cluster_label]<alpha_stat]), 5)}")
                    print(f"{G1}vs{G3}: {sum(df_p_posthoc['p13L'][labels == cluster_label])} of {len(df_p_posthoc['p13L'][labels == cluster_label])} | {sum(df_p_posthoc['p13L'][labels == cluster_label])/len(df_p_posthoc['p13L'][labels == cluster_label])} | avg p={np.round(np.mean(df_p_posthoc['p13'][labels == cluster_label][df_p_posthoc['p13'][labels == cluster_label]<alpha_stat]), 5)}")
                    print(f"{G2}vs{G3}: {sum(df_p_posthoc['p23L'][labels == cluster_label])} of {len(df_p_posthoc['p23L'][labels == cluster_label])} | {sum(df_p_posthoc['p23L'][labels == cluster_label])/len(df_p_posthoc['p23L'][labels == cluster_label])} | avg p={np.round(np.mean(df_p_posthoc['p23'][labels == cluster_label][df_p_posthoc['p23'][labels == cluster_label]<alpha_stat]), 5)}")

            if len(data) == 2:
                print(f'-------------label {cluster_label}------with tmin {min_cluster} and tmax {max_cluster}------and len {len(cluster_data)}')
            
            for i in range(len(data)):
                category=erpData[feat]['name'][i]
                ind_data=var['name'].index(category)
                n_subj=len(np.vstack(np.stack(data[ind_data], axis=1)[0]))
                
                for j in range(n_subj):
                    integrand=abs(np.median(np.stack(data[ind_data], axis=1)[Q[parcel]],axis=0)[j][np.logical_and(times>=min_cluster,times<=max_cluster)])
                    #intg_curr=scipy.integrate.trapezoid(integrand,x=times[np.logical_and(times>min_cluster,times<max_cluster)])
                    #intg_curr=np.sum(integrand)
                    intg_curr=scipy.integrate.simpson(integrand,x=times[np.logical_and(times>=min_cluster,times<=max_cluster)])
                    #intg_curr=np.trapz(integrand,x=times[np.logical_and(times>min_cluster,times<max_cluster)])
                    #intg_curr=auc(times[np.logical_and(times>min_cluster,times<max_cluster)],integrand) #trapz e auc mi danno gli stessi risultati
                    max_curr=max(np.median(np.stack(data[ind_data], axis=1)[Q[parcel]],axis=0)[j][np.logical_and(times>=min_cluster,times<=max_cluster)])
                    min_curr=min(np.median(np.stack(data[ind_data], axis=1)[Q[parcel]],axis=0)[j][np.logical_and(times>=min_cluster,times<=max_cluster)])
                    sex= var['sex'][i][j]
                    age_cont=var['age_cont'][i][j]
                    age_cat=var['age_cat'][i][j]
                    ID_reg=var['ID_reg'][i][j]
                    summary.append([ID_reg,
                                    j,#subj_ID
                                    category,
                                    sex,
                                    age_cont,
                                    age_cat,
                                    parcel,
                                    cluster_label,
                                    min_cluster,
                                    max_cluster,
                                    max_curr,
                                    min_curr,
                                    intg_curr])
                    
                print(f"Temporal window label: {cluster_label} --- Sub-group: {category} ")

        df_feature= pd.DataFrame(summary, columns=['ID_reg','ID_withincat', 'category','sex','age_cont','age_cat', 'electrode','t_window','t_min','t_max','max feat','min feat','integral feat'])

    return p,p_corrected,effect_size,df_p_posthoc,min_values,max_values,df_feature



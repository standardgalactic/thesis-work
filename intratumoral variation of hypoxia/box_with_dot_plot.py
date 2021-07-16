# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:05:50 2021
Combination boxplot and stripplot to show how hypoxia can vary, both across different tissue sections in the same depth, and the same tissue section at different depths
@author: Mark Zaidi
"""
#%% load libraries
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
plt.close('all')
#%% Read data
data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Intratumoral Variation of Hypoxia\annotation_measurements.csv')
data=data.sort_values("Class")
#%% test on sample data
#tips = sns.load_dataset("tips")
#plt.figure()
#sns.boxplot(x="day", y="total_bill", data=tips)
#sns.stripplot(x="day", y="total_bill", data=tips, size=4, jitter=True, edgecolor="gray")
#%% Generate boxplot first
PROPS = {
    'boxprops':{'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'},
    'flierprops':{'markerfacecolor':'black','markeredgecolor':'black'}

}

plt.figure()
sns.boxplot(x="Image", y="Positive %", data=data, color=(1,1,1),showfliers=False,**PROPS)
#%% generate stripplot
g=sns.stripplot(x="Image", y="Positive %", data=data, size=8, jitter=True, edgecolor="gray",hue="Class")
plt.xticks(rotation=90)
g.get_legend().remove()
g.set_xticklabels(g.get_xticklabels(), size=10)
#Optional renaming of labels and additional parameters
g.set_xticklabels({1,2,3,4,5}, size=10)
g.set_xlabel("Depth in Tissue Block")
g.set_ylabel("% PIMO Positive")
g.figure.set_size_inches(3,3)

plt.tight_layout()

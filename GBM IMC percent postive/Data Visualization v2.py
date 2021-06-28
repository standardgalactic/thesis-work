# -*- coding: utf-8 -*-
"""
V2 contains less hardcoded values throughout the code, and performs statistical tests on violin plots

To do:
    - add vessel distance analysis
    - calculate microvessel density across areas (MVD)
    - optional data filtering by value
    - support for grouping patients by supplementary data (e.g. who is IDH wt or mutant)
    - pairplots

@author: Mark Zaidi
"""

#%% load libraries
import pandas
import math
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy import stats
from statannot import add_stat_annotation
import time
#%% Read data
data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\measurements_ROI4only.csv')
annotation_data=pandas.read_csv(r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\annotation_measurements.csv')
col_names=data.columns
#%% set constants
param_Name='Name'
param_Parent='Parent'
param_UnusedClass='PathCellObject'
param_pos_kwd='pimo positive'
param_neg_kwd='pimo negative'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\Feb 2021 IMC\figures'
#measurement names for Feb 2021 batch
measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Median','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurement names for Jun 2021 old batch
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Median','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(148)_148Nd-Tau: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Nd(150)_150Nd-PD-L1: Cell: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Median','Sm(154)_154Sm-GPG95: Cell: Mean','Gd(155)_155Gd-Pimo: Cell: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-pSTAT3: Nucleus: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-NGFR: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Dy(163)_163Dy-CD163: Cell: Mean','Ho(165)_165Ho-CD45RO: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tm-Synaptophysin: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(172)_172Yb-CD57: Cell: Mean','Yb(173)_173Yb-S100: Cell: Mean','Lu(175)_175Lu-pS6: Cell: Mean','Yb(176)_176Yb-Iba1: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']

#%% count the number of cells in different regions of interest
cells_in_pimo_pos=sum(data.apply(lambda x: 1 if x[param_Parent] == param_pos_kwd else 0 , axis=1))
cells_in_pimo_neg=sum(data.apply(lambda x: 1 if x[param_Parent] == param_neg_kwd else 0 , axis=1))
#%% get unique IHC marker names
#split class name by :, listing all classes a cell belongs to
df2=data[param_Name].str.split(':',expand=True)
#Identify unique names, filtering out the unused class name and any NoneType cases
df3=pandas.unique(df2[df2.columns].values.ravel('K'))
df3=df3[(df3 != param_UnusedClass)]
marker_list = [x for x in df3 if x != None]
#remove spaces, cast to list
marker_list=[x.strip(' ') for x in marker_list]
marker_list=list(set(marker_list))
#get unique annotation class names. Should be equal to the value set by param_pos_kwd and param_neg_kwd
annotation_list=data[param_Parent].unique().tolist()
#get shortened marker name (for figure plotting)
marker_short=[i.split('-', 1)[1] for i in marker_list]
#%% iteratively get percent positive scores
#define variables
count=0
pct_name=list()
marker_count=0
pct_pos=[]
pos_in_pimo_neg=[]
pos_in_pimo_pos=[]
for marker in marker_short:
    #identify full marker name
    full_marker=marker_list[marker_count]
    marker_count=marker_count+1
    for annotation in annotation_list:
        #label current entry as the current iterators for marker and annotation
        pct_name.append(['Percent ' + marker + ' positive in ' + annotation])
        #create filters to include cells from original data that belong to current annotation and marker classification
        cond1= data[param_Name].str.contains(full_marker,regex=False)
        cond2= data[param_Parent].str.contains(annotation,regex=False)
        #If conditions are met, calculate the total number of cells that meet cond1&cond2, divide by total cells in cond2(annotation) and multiply by 100
        #pct_name corresponds to pct_pos value
        pct_pos.append((cond1&cond2).sum()/cond2.sum()*100)
        if param_pos_kwd in annotation:
            pos_in_pimo_pos.append((cond1&cond2).sum()/cond2.sum()*100)
        elif param_neg_kwd in annotation:
            pos_in_pimo_neg.append((cond1&cond2).sum()/cond2.sum()*100)

        count=count+1
#%% visualize percent positive as clustered bar chart
pct_pos_df=pandas.DataFrame([marker_short,pos_in_pimo_pos,pos_in_pimo_neg,[i / j for i, j in zip(pos_in_pimo_pos, pos_in_pimo_neg)]]).transpose().sort_values(3,ascending=False)
#labels = marker_short
#neg_bar = pos_in_pimo_pos
#pos_bar = pos_in_pimo_neg
labels=pct_pos_df[0]
pos_bar=pct_pos_df[1]
neg_bar=pct_pos_df[2]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
plt.xticks(rotation=90)
rects1 = ax.bar(x - width/2, neg_bar, width, label='PIMO Negative')
rects2 = ax.bar(x + width/2, pos_bar, width, label='PIMO Positive')

#add data labels
ratios = pct_pos_df[3]
for rect1, rect2, ratio in zip(rects1, rects2, ratios):
    ratio="{:.2f}".format(ratio)
    #height = rect2.get_height()
    height=(max(rect1.get_height(),rect2.get_height()))
    ax.text(rect2.get_x() + rect2.get_width() / 2, height + 5, ratio,
            ha='center', va='bottom',rotation='vertical')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percent Positive')
ax.set_title('Percent positive scores in PIMO +/- regions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.ylim([0,125])
fig.tight_layout()
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

#fig.tight_layout()
#plt.savefig(r'C:\Users\Mark Zaidi\Documents\Python Scripts\GBM IMC percent postive\Percent positive scores in PIMO + vs - regions.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
plt.savefig(figpath + r'\Percent positive scores in PIMO + vs - regions.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
plt.close()

# Make some labels.
#ratios = ["label%d" % i for i in xrange(len(rects))]
#%% Visualize IHC marker in PIMO positive vs negative as a violin plot
#Identify cells belonging to pos or neg Parent annotations
pos_data= data[data['Parent'].str.contains('pimo positive',regex=False)]
neg_data= data[data['Parent'].str.contains('pimo negative',regex=False)]
# Pair up marker names with measurement (not all are cell mean, some are nucleus median, etc.)
#maybe made dataframe with columns: marker, measurement, and hmmmmmmmm
col_names=data.columns
#%% create a clustered violin plot with 1 example
#violin_plt_data=data[data['Pr(141)_141Pr-aSMA: Cell: Mean']<0.9*data['Pr(141)_141Pr-aSMA: Cell: Mean'].max()]
#ax = sns.violinplot( x='Parent',y="Pr(141)_141Pr-aSMA: Cell: Mean",data=violin_plt_data, palette="muted",scale='width',cut=0)
#%% now do the above, but iteratively
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Mean','Nd(143)_143Nd-GFAP: Cell: Median','Nd(145)_145Nd-CD31: Cell: Mean','Nd(150)_150Nd-SOX2: Nucleus: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Mean','Eu(153)_153Eu-VCAM: Cell: Mean','Gd(155)_155Gd-PIMO: Cell: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-GLUT1: Cell: Mean','Dy(163)_163Dy-HK2: Cell: Mean','Dy(164)_164Dy-LDHA: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Er(170)_170Er-IBA1: Cell: Mean','Yb(173)_173Yb-TMHistone: Nucleus: Mean','Yb(174)_174Yb-ICAM: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']
#measurements_of_interest=['Pr(141)_141Pr-aSMA: Cell: Median','Nd(143)_143Nd-GFAP: Cell: Mean','Nd(145)_145Nd-CD31: Cell: Mean','Nd(146)_146Nd-Nestin: Cell: Mean','Nd(148)_148Nd-Tau: Cell: Mean','Sm(149)_149Sm-CD11b: Cell: Mean','Nd(150)_150Nd-PD-L1: Cell: Median','Eu(151)_151Eu-CA9: Cell: Mean','Sm(152)_152Sm-CD45: Cell: Median','Sm(154)_154Sm-GPG95: Cell: Mean','Gd(155)_155Gd-Pimo: Cell: Mean','Gd(156)_156Gd-CD4: Cell: Mean','Gd(158)_158Gd-pSTAT3: Nucleus: Mean','Tb(159)_159Tb-CD68: Cell: Mean','Gd(160)_160Gd-NGFR: Cell: Mean','Dy(161)_161Dy-CD20: Cell: Mean','Dy(162)_162Dy-CD8a: Cell: Mean','Dy(163)_163Dy-CD163: Cell: Mean','Ho(165)_165Ho-CD45RO: Cell: Mean','Er(167)_167Er-GranzymeB: Cell: Mean','Er(168)_168Er-Ki67: Nucleus: Mean','Tm(169)_169Tm-Synaptophysin: Cell: Mean','Er(170)_170Er-CD3: Cell: Mean','Yb(172)_172Yb-CD57: Cell: Mean','Yb(173)_173Yb-S100: Cell: Mean','Lu(175)_175Lu-pS6: Cell: Mean','Yb(176)_176Yb-Iba1: Cell: Mean','Ir(191)_191Ir-DNA191: Nucleus: Mean','Ir(193)_193Ir-DNA193: Nucleus: Mean']

#2 lines below are violin code
# fig = plt.figure(figsize=(30, 30))
# gs = fig.add_gridspec(3, 6)

#fig = plt.figure(figsize=(6, 6))
count=0
index=0,0
#figpath=r'C:\Users\Mark Zaidi\Documents\Python Scripts\GBM IMC percent postive\figures'
measure_short=[i.split('-', 1)[1] for i in measurements_of_interest]
measure_short_for_fname=[i.split(':', 1)[0] for i in measure_short]

for measure in measurements_of_interest:
    #set up conditions for filtering cells that belong to a given annotation and are less than the mean + 2 std for the measurement
    cond1_pos= data[param_Parent].str.contains(param_pos_kwd,regex=False)
    cond1_neg= data[param_Parent].str.contains(param_neg_kwd,regex=False)
    cond2=data[measure]<(data[measure].mean()+2*data[measure].std())
    pos_selection=data[cond1_pos&cond2][measure]
    neg_selection=data[cond1_neg&cond2][measure]

    ax = sns.violinplot( x='Parent',y=measure,data=data[data[measure]<(data[measure].mean()+2*data[measure].std())], palette="muted",scale='width',cut=0,inner="box")

    #Calculate stats. NOTE: NOT SURE HOW RELIABLE THESE ARE; GETTING INSANELY SMALL P VALUES
    test_results = add_stat_annotation(ax, data=data[data[measure]<(data[measure].mean()+2*data[measure].std())], x='Parent', y=measure,
                                   box_pairs=[(param_neg_kwd, param_pos_kwd)],
                                   test='t-test_ind', text_format='full',
                                   loc='outside', verbose=2,comparisons_correction=None)
    print(measure,stats.ttest_ind(neg_selection,pos_selection,equal_var=False),'\n')

    plt.savefig(os.path.join(figpath,measure_short_for_fname[count]+'.png'),dpi=800,pad_inches=0.1,bbox_inches='tight')
    count=count+1
    plt.close()
#%% create overall violinplot figure
fig = plt.figure(figsize=(30, 30))

gs = fig.add_gridspec(5, 6)
count=0
index=0,0

#figpath=r'C:\Users\Mark Zaidi\Documents\Python Scripts\GBM IMC percent postive\figures'
for measure in measurements_of_interest:
    
    #ax = sns.violinplot( x='Parent',y=measure,data=data[data[measure]<0.9*data[measure].max()], palette="muted",scale='width',cut=0)
    # ax = sns.violinplot( x='Parent',y=measure,data=data[data[measure]<(data[measure].mean()+2*data[measure].std())], palette="muted",scale='width',cut=0,inner="box")
    # plt.savefig(os.path.join(figpath,measure_short[count]+'.png'),dpi=800,pad_inches=0.1,bbox_inches='tight')
    # count=count+1
    # plt.close()
    ##Violin subplot code below
    print(count)
    if count<6:
        index=0,count
    elif (count>=6)&(count<12):
        index=1,count-6
    elif (count>=12)&(count<18):
        index=2,count-12
    elif (count>=18)&(count<24):
        index=3,count-18
    elif (count>=24)&(count<30):
        index=4,count-24
    ax = fig.add_subplot(gs[index])
    sns.set(font_scale=0.5)
    ax.set_ylabel(measure_short[count])
    ax = sns.violinplot( x='Parent',y=measure,data=data[data[measure]<(data[measure].mean()+2*data[measure].std())], palette="muted",scale='width',cut=0,inner="box")

    

    test_results = add_stat_annotation(ax, data=data[data[measure]<(data[measure].mean()+2*data[measure].std())], x='Parent', y=measure,
                                   box_pairs=[(param_neg_kwd, param_pos_kwd)],
                                   test='t-test_ind', text_format='full',
                                   loc='outside', verbose=2,comparisons_correction=None)
    count=count+1
#need to put a 1 second pause to finish plotting before applying tight layout
matplotlib.pyplot.pause(1)
fig.tight_layout()
plt.savefig(os.path.join(figpath,'OverallV5.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:02:10 2022

@author: Mark Zaidi

Goal is to adapt this "Percent positive heatmaps" script to amend a 3rd dimension to this data, which is the annotation from the pixel classifier it came from
"""
#%% Import libraries
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
#%% Read data
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\processed_annotation_measurements_v2.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\percent_multiple_positives_withPixelClassifier'
groupby='Class'
groupby_bckup=groupby
denominator='Area µm^2'#'Area µm^2'

data=pandas.read_csv(csv_path)
columns=data.columns.tolist()
total_expected_markers=27
#Drop rows containing "missing" code
data=data[data["Code"].str.contains("Missing")==False]
if (groupby=='Code'):
    #order = ['Control','ICI-AIN', 'Drug-AIN', 'ATN', 'TKI-TMA', 'ATN-TX','ATN-Other']
   order = ['Normal','Mixed', 'ABMR', 'BK','C. Pyel','ACR']
#   order.remove('cABMR')
#   Replace ABMR* and cABMR as ABMR
   data.Code=data.Code.replace({'cABMR':'ABMR','ABMR*':'ABMR'})
elif (groupby=="Code_full"):
    order=['Control 1','Control 2','Control 3','ICI-AIN','ICI-AIN 1','ICI-AIN 2','ICI-AIN 3','ICI-AIN 4','Drug-AIN 1','Drug-AIN 2','Drug-AIN 3','ATN 1','ATN 2','ATN 3','TKI-TMA','ATN-TX','ATN-Other']
elif (groupby=="Sandra_4"):
    order=['Control','ICI-AIN','ICI-Other','Drug-AIN']
else:
    order=data[groupby].unique()
#%%Optional filtering of data
data=data[~data['Class'].str.contains("Empty Space")] #Specify the classes you don't want to plot
order=data[groupby].unique()

#%% Derive IHC marker names
#drop non IHC related columns
col1=[x for x in columns if x.startswith('Num')]
col1.remove('Num Detections')
#Split list elements by :, remove leading whitespace
col2 = [word.lstrip() for line in col1 for word in line.split(':')]
#Identify unique elements. Apparently you just convert the list to a set and back
col2=list(set(col2))
#Remove leading "Num" again
col3= [word.lstrip() for line in col2 for word in line.split(' ')]
col3=list(set(col3))
col3.remove('Num')
#Remove contents before underscore
col4=[x.split('_',1)[1] for x in col3]
#Just for added safety, make sure the number of unique markers found matches up with what we're expecting
if (len(col4)!=total_expected_markers):
     raise Exception('Number of markers found does not match number expected')
#%% Calculate percent single positive heatmap values, grouped by the groupby variable
grouped_df=pd.DataFrame(columns=[groupby] + col4)     
#grouped_df=pd.DataFrame(columns=[groupby]+ ['Sandra_4'] + col4)     #uncomment this if group=Image and you want to generate raw data
for group in data[groupby].unique():
    #select data in which all samples belong to the same code
    #group=data[groupby].unique()[0]
    curr_data=data[data[groupby]==group]
    #sum some kind of denominator across all cases (e.g. total number of cells, or total area)
    denominator_sum=curr_data[denominator].sum(axis=0)
    pct_pos_list=[]
    for curr_marker in col4:
        #for each single positive marker, sum the total number of numerator (e.g. single positive) cells
        #curr_marker=col4[3]
        #Identify columns containing marker name
        cols_containing_marker=[x for x in columns if curr_marker in x]
        #Sum all columns containing cell counts relevant to curr_marker
        summed_columns=curr_data[cols_containing_marker].sum(axis=1)
        #sum all rows containing IMC ROIs relating to a particular grouping
        summed_rows=summed_columns.sum()
        if (denominator=='Num Detections'):
            pct_pos=summed_rows/denominator_sum*100
        elif (denominator=='Area µm^2'):
            pct_pos=summed_rows/denominator_sum*1000**2
        else:
            raise Exception('Unable to figure out what arithmetic operation to perform with pct_pos and denominator_sum ')
        pct_pos_list.append(pct_pos)
    #create a temporary dataframe to prepare appending to the master dataframe
    temp_df=pd.DataFrame([pct_pos_list],columns=col4)
    temp_df.insert(0,groupby,group)
    #temp_df.insert(1,'Sandra_4',curr_data.iloc[0]['Sandra_4'])      #uncomment this if group=Image and you want to generate raw data

    #append to grouped_df
    grouped_df=pd.concat([grouped_df,temp_df])
#Make the index be the groupby variable. Useful if we want to sort the values when plotting the heatmap
grouped_df.index=grouped_df[groupby]
grouped_df=grouped_df.reindex(order)
#Rename the god forsaken CD16 case to have a hyphen in it, as that always breaks things down the line
grouped_df=grouped_df.rename(columns={"146NdCD16":"146Nd-CD16","144Nd-cd14":"144Nd-CD14"})
#Write out grouped_df to .csv. Useful when grouping variable is Image and we want to do stats
#grouped_df.to_csv(os.path.join(figpath,'grouped_data.csv'),index=False)

#%% Append percent double/triple positive combinations
#This is where we'd calculate the double/triple positive combos. We'd perform
#the same script as above, but the cols_containing_marker would have to be
#selected in such a way that it selects the columns containing the triple
#positive combinations. I'd reccomend looking at the regex solution https://stackoverflow.com/a/70324300
#which shows a regex-based way of selecting columns that contain all keywords present in a list of words.
#So maybe, you can pass it a dict, where the key is the 'Activated CD4 macrophage' and the values are a
#variable length list containing the mass number and marker (see col4 item for example)

#Create a dict
immune_marker_dict={
    "CD4 Memory T":   ["156Gd-CD4","170Er-CD3","173Yb-CD45RO"],
    "CD4 Naive T":    ["156Gd-CD4","170Er-CD3","166Er-CD45RA"],
    "CD8 Memory T":   ["162Dy-CD8a","170Er-CD3","173Yb-CD45RO"],
    "CD8 Naive T":    ["162Dy-CD8a","170Er-CD3","166Er-CD45RA"],
    "T Helper":       ["170Er-CD3","156Gd-CD4"],
    "T Cytotoxic":    ["170Er-CD3","162Dy-CD8a"],
    "T Reg":          ["170Er-CD3","156Gd-CD4","155Gd-FoxP3"],
    "B":              ["142Nd-CD19","161Dy-CD20"],
    "Dendritic":      ["154Sm-CD11c","174Yb-HLA-DR"],
    "Macrophage":     ["149Sm-CD11b","144Nd-cd14","159Tb-CD68"],
    "NK":             ["167Er-GranzymeB","146NdCD16"] #Note missing hyphen between metal tag and marker
    #Omitted Activated CD4 and CD8 as Sandra's panel is missing PD-1. Also not sure if we're going forward with this for priya (stained for PD-L1 not PD-1)
    }
immune_df=pd.DataFrame(columns=[groupby] + list(immune_marker_dict.keys()))
for group in data[groupby].unique():
    #select data in which all samples belong to the same code
    #group=data[groupby].unique()[0]
    curr_data=data[data[groupby]==group]
    #sum some kind of denominator across all cases (e.g. total number of cells, or total area)
    denominator_sum=curr_data[denominator].sum(axis=0)
    pct_pos_list=[]
    for immune_key in immune_marker_dict: #For each immune cell key in the immune_marker_dict
        #immune_key=list(immune_marker_dict.keys())[0]
        curr_immune_vals=immune_marker_dict[immune_key] #Get the list of markers that a cell must be positive for to be defined as a given immune cell
        cols_containing_marker=columns #First, set cols_containing_marker to be all columns in the dataset
        for single_marker in curr_immune_vals: #Then, for each of the individual markers defining the immune cell
            cols_containing_marker=[x for x in cols_containing_marker if single_marker in x] #Only select the columns that contain the individual marker. From this list, move onto the next single marker, and the next, and so on
            #Once this loop is completed, cols_containing_markers should now ONLY be composed of columns representative of cells that contain all of the necessary positive markers
            
        #Sum all columns containing cell counts relevant to curr_marker
        summed_columns=curr_data[cols_containing_marker].sum(axis=1)
        #sum all rows containing IMC ROIs relating to a particular grouping
        summed_rows=summed_columns.sum()
        if (denominator=='Num Detections'):
            pct_pos=summed_rows/denominator_sum*100
        elif (denominator=='Area µm^2'):
            pct_pos=summed_rows/denominator_sum*1000**2
        else:
            raise Exception('Unable to figure out what arithmetic operation to perform with pct_pos and denominator_sum ')

        pct_pos_list.append(pct_pos)
    #create a temporary dataframe to prepare appending to the master dataframe
    temp_df=pd.DataFrame([pct_pos_list],columns=list(immune_marker_dict.keys()))
    temp_df.insert(0,groupby,group)
    #temp_df.insert(1,'Sandra_4',curr_data.iloc[0]['Sandra_4'])      #uncomment this if group=Image and you want to generate raw data

    #append to grouped_df
    immune_df=pd.concat([immune_df,temp_df])
#Make the index be the groupby variable. Useful if we want to sort the values when plotting the heatmap
immune_df.index=immune_df[groupby]
immune_df=immune_df.reindex(order)
immune_df=immune_df.drop(groupby,axis=1)
#Concatenate with existing group
grouped_df=pd.concat([grouped_df,immune_df],axis=1,verify_integrity=True)
grouped_df=grouped_df[grouped_df.columns[1:]].astype('float')
grouped_df.to_csv(os.path.join(figpath,'grouped_data.csv'),index=True)











#%% Generate heatmaps
plt.close('all')

#Sandra heatmap cols
#Immune markers only
#heatmap_cols=['170Er-CD3','156Gd-CD4','162Dy-CD8a','149Sm-CD11b','154Sm-CD11c','144Nd-CD14','146Nd-CD16','142Nd-CD19','161Dy-CD20','152Sm-CD45','173Yb-CD45RO','166Er-CD45RA','159Tb-CD68','155Gd-FoxP3','174Yb-HLA-DR','167Er-GranzymeB']
#Structural markers only
#heatmap_cols=['160Gd-Vista','141-SMA','143Nd-Vimentin','158Gd-E_Cadherin','148-Pan-Ker','175Lu-Beta2M','176Yb-Nak-ATPase','169Tmp-CollagenI']
#Immune cell subtypes
heatmap_cols=['CD4 Memory T', 'CD4 Naive T', 'CD8 Memory T', 'CD8 Naive T', 'T Helper', 'T Cytotoxic', 'T Reg', 'B', 'Dendritic', 'Macrophage', 'NK']

#normalize heatmap values by column (using z score normalization formula)
norm_heatmap=grouped_df.copy()
norm_heatmap[grouped_df.columns]=norm_heatmap[grouped_df.columns].apply(lambda x: (x-x.mean())/(x.std()+0.00000001), axis = 0)

#plot heatmap (labelled with original percentages but colored by column-normalized values)
#.transpose() rotates the dataframe to be row=marker and column=group
h=sns.heatmap(data=norm_heatmap[heatmap_cols].transpose(),annot=grouped_df[heatmap_cols].transpose(),square=True,cmap="viridis",cbar=False,annot_kws={"size": 6},fmt='.1f')
#plot heatmap without any normalization
#h=sns.heatmap(data=grouped_df[heatmap_cols],annot=grouped_df[heatmap_cols],square=True,cmap="viridis")

#Trim ytick labels
# trimmed_cols=[x.split('-',1)[1] for x in heatmap_cols]
# h.set_yticklabels(trimmed_cols)
h.set_xlabel('') #hide x axis label
#Shift xtick labels to top
h.xaxis.tick_top() # x axis on top
h.xaxis.set_label_position('top')
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(os.path.join(figpath,'Immune subtypes heatmap.png'),dpi=300,pad_inches=0.1,bbox_inches='tight')





     
#%% Generate raw_data csv showing percent of cells positive by ROI, with an additional column denoting the parent group
secondary_group=groupby_bckup

groupby='Image'
denominator='Num Detections'
order=data[groupby].unique()

#grouped_df=pd.DataFrame(columns=[groupby] + col4)     
grouped_df=pd.DataFrame(columns=[groupby]+ [secondary_group] + col4)     #uncomment this if group=Image and you want to generate raw data
for group in data[groupby].unique():
    #select data in which all samples belong to the same code
    #group=data[groupby].unique()[0]
    curr_data=data[data[groupby]==group]
    #sum some kind of denominator across all cases (e.g. total number of cells, or total area)
    denominator_sum=curr_data[denominator].sum(axis=0)
    pct_pos_list=[]
    for curr_marker in col4:
        #for each single positive marker, sum the total number of numerator (e.g. single positive) cells
        #curr_marker=col4[3]
        #Identify columns containing marker name
        cols_containing_marker=[x for x in columns if curr_marker in x]
        #Sum all columns containing cell counts relevant to curr_marker
        summed_columns=curr_data[cols_containing_marker].sum(axis=1)
        #sum all rows containing IMC ROIs relating to a particular grouping
        summed_rows=summed_columns.sum()
        if (denominator=='Num Detections'):
            pct_pos=summed_rows/denominator_sum*100
        elif (denominator=='Area µm^2'):
            pct_pos=summed_rows/denominator_sum*1000**2
        else:
            raise Exception('Unable to figure out what arithmetic operation to perform with pct_pos and denominator_sum ')
        pct_pos_list.append(pct_pos)
    #create a temporary dataframe to prepare appending to the master dataframe
    temp_df=pd.DataFrame([pct_pos_list],columns=col4)
    temp_df.insert(0,groupby,group)
    temp_df.insert(1,secondary_group,curr_data.iloc[0][secondary_group])      #uncomment this if group=Image and you want to generate raw data

    #append to grouped_df
    grouped_df=pd.concat([grouped_df,temp_df])
#Make the index be the groupby variable. Useful if we want to sort the values when plotting the heatmap
grouped_df.index=grouped_df[groupby]
grouped_df=grouped_df.reindex(order)
#Rename the god forsaken CD16 case to have a hyphen in it, as that always breaks things down the line
grouped_df=grouped_df.rename(columns={"146NdCD16":"146Nd-CD16","144Nd-cd14":"144Nd-CD14"})
immune_df=pd.DataFrame(columns=[groupby] + list(immune_marker_dict.keys()))
for group in data[groupby].unique():
    #select data in which all samples belong to the same code
    #group=data[groupby].unique()[0]
    curr_data=data[data[groupby]==group]
    #sum some kind of denominator across all cases (e.g. total number of cells, or total area)
    denominator_sum=curr_data[denominator].sum(axis=0)
    pct_pos_list=[]
    for immune_key in immune_marker_dict: #For each immune cell key in the immune_marker_dict
        #immune_key=list(immune_marker_dict.keys())[0]
        curr_immune_vals=immune_marker_dict[immune_key] #Get the list of markers that a cell must be positive for to be defined as a given immune cell
        cols_containing_marker=columns #First, set cols_containing_marker to be all columns in the dataset
        for single_marker in curr_immune_vals: #Then, for each of the individual markers defining the immune cell
            cols_containing_marker=[x for x in cols_containing_marker if single_marker in x] #Only select the columns that contain the individual marker. From this list, move onto the next single marker, and the next, and so on
            #Once this loop is completed, cols_containing_markers should now ONLY be composed of columns representative of cells that contain all of the necessary positive markers
            
        #Sum all columns containing cell counts relevant to curr_marker
        summed_columns=curr_data[cols_containing_marker].sum(axis=1)
        #sum all rows containing IMC ROIs relating to a particular grouping
        summed_rows=summed_columns.sum()
        if (denominator=='Num Detections'):
            pct_pos=summed_rows/denominator_sum*100
        elif (denominator=='Area µm^2'):
            pct_pos=summed_rows/denominator_sum*1000**2
        else:
            raise Exception('Unable to figure out what arithmetic operation to perform with pct_pos and denominator_sum ')
        pct_pos_list.append(pct_pos)
    #create a temporary dataframe to prepare appending to the master dataframe
    temp_df=pd.DataFrame([pct_pos_list],columns=list(immune_marker_dict.keys()))
    temp_df.insert(0,groupby,group)
    #temp_df.insert(1,'Sandra_4',curr_data.iloc[0]['Sandra_4'])      #uncomment this if group=Image and you want to generate raw data

    #append to grouped_df
    immune_df=pd.concat([immune_df,temp_df])
#Make the index be the groupby variable. Useful if we want to sort the values when plotting the heatmap
immune_df.index=immune_df[groupby]
immune_df=immune_df.reindex(order)
immune_df=immune_df.drop(groupby,axis=1)
#Concatenate with existing group
grouped_df=pd.concat([grouped_df,immune_df],axis=1,verify_integrity=True)
#grouped_df=grouped_df[grouped_df.columns[1:]].astype('float') #Cast all number containing columns as float
grouped_df.to_csv(os.path.join(figpath,'raw_data.csv'),index=False)

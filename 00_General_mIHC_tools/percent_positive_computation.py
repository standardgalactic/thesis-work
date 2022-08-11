# -*- coding: utf-8 -*-
"""
Created on August 9 2022

@author: Mark Zaidi

General purpose percent positive scoring, requiring a processed_annotation_measurements.csv exported from QuPath
Currently computes:
    Percent single positive for each marker bearing a classification
    Percent multiple positive, for cell types defined by more than one markers
    Negative gating: if a cell defined by multiple markers requires one or more markers to be negative, append _NEGATIVE to the name
    Choice of using total cells (percent positive) or total area (num positive per mm^2) as the denominator
    Alternative grouping methods, but for now, will compute a score for each ROI
    Will yield raw_data.csv, where each row is an entry (ROI, group, etc.) and each column is a percent positive score
    Data visualization should be kept as a separate script
To implement:
    Clean up comments
"""
#%% Import libraries
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
#%% Set constants
csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\20220124_Sandra\processed_annotation_measurements.csv'
#Example with multiple annotations per image (renal structures)
#csv_path=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\processed_annotation_measurements_v2.csv'
figpath=r'C:\Users\Mark Zaidi\Documents\QuPath\HistoWiz\Priya_v2\figures\percent positive heatmaps\debug'
groupby='Image'
denominator='Num Detections'
total_expected_markers=26
metadata_to_append=['KR','IMC_ROI','Code'] #Columns to include in raw_data. Must be already present in data
#%% Read data
data=pandas.read_csv(csv_path)

#%% Filter or rename data
#Drop rows containing "missing" code
#data=data[data["Code"].str.contains("Missing")==False]
# Replace any instances of improperly named CD16 or CD14 cases. If more, consider looping over a dict
data.columns = data.columns.str.replace('146NdCD16', '146Nd-CD16')
data.columns = data.columns.str.replace('144Nd-cd14', '144Nd-CD14')
columns=data.columns.tolist()
columns_nonum=[x for x in columns if not x.startswith('Num')] #Additional
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
grouped_df=grouped_df.set_index(groupby)
#Sort columns alphabetically
grouped_df = grouped_df.reindex(sorted(grouped_df.columns), axis=1)
# grouped_df.index=grouped_df[groupby]
# grouped_df=grouped_df.reindex(order)
#Rename the god forsaken CD16 case to have a hyphen in it, as that always breaks things down the line
#grouped_df=grouped_df.rename(columns={"146NdCD16":"146Nd-CD16","144Nd-cd14":"144Nd-CD14"})
#Write out grouped_df to .csv. Useful when grouping variable is Image and we want to do stats
#grouped_df.to_csv(os.path.join(figpath,'grouped_data.csv'),index=False)
#%% Generate column for total unclassified cells
cols_containing_marker=columns #First, set cols_containing_marker to be all columns in the dataset
cols_containing_marker_protein=[x for x in cols_containing_marker if x.startswith('Num')] #Only select columns that contain Num keyword
cols_containing_marker_protein=[x for x in cols_containing_marker_protein if not x.startswith('Num Detections')] #Remove Num Detections
unclassified_sum=list(data['Num Detections']-data[cols_containing_marker_protein].sum(axis=1))
data['Num Unclassified']=unclassified_sum
#update cols_containing_marker and it's derivatives to contain a Num Unclassified column, useful in negative gating where cells lack
#any classification. If a cell is negative for all markers
cols_containing_marker=data.columns.tolist() #First, set cols_containing_marker to be all columns in the dataset
cols_containing_marker_protein=[x for x in cols_containing_marker if x.startswith('Num')] #Only select columns that contain Num keyword
cols_containing_marker_protein=[x for x in cols_containing_marker_protein if not x.startswith('Num Detections')] #Remove Num Detections
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
    "Macrophage":     ["149Sm-CD11b","144Nd-CD14","159Tb-CD68"],
    "NK":             ["167Er-GranzymeB","146Nd-CD16"],
    "MARK_DEBUG":     ["146Nd-CD16","167Er-GranzymeB_NEGATIVE"], #experimental case for negative gating
    "SMA_INVERSE":    ["141-SMA_NEGATIVE"]
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
        #immune_key=list(immune_marker_dict.keys())[11]
        curr_immune_vals=immune_marker_dict[immune_key] #Get the list of markers that a cell must be positive for to be defined as a given immune cell

        list_of_list_cols_selected=[]
        for single_marker in curr_immune_vals: #Then, for each of the individual markers defining the immune cell
            #single_marker=curr_immune_vals[1]
            #If a _NEGATIVE marker is specified, omit
            #if ('_NEGATIVE' in single_marker):
            #    single_marker=single_marker.replace('_NEGATIVE','')
            #cols_containing_marker=[x for x in cols_containing_marker if single_marker in x]#Only select the columns that contain the individual marker. From this list, move onto the next single marker, and the next, and so on
            cols_containing_marker_list=[]
            if ('_NEGATIVE' not in single_marker): #positive selection case
                cols_containing_marker_list.extend([s for s in cols_containing_marker_protein if single_marker in s])
            elif ('_NEGATIVE' in single_marker): #negative selection case
                single_marker=single_marker.replace('_NEGATIVE','') #Remove the _NEGATIVE tag such that you can find matches for your negative case
                cols_containing_marker_list.extend([s for s in cols_containing_marker_protein if single_marker not in s])
            list_of_list_cols_selected.append(cols_containing_marker_list)
                    #FIND THE THINGS IN COMMON between multiple lists. So for each single_marker, find the columns that contain single_marker in the name.
                    #If _NEGATIVE, have a special case where you chop off the _NEGATIVE, select the columns that contain the marker, but then
                    #invert the selection
                
            #Once this loop is completed, cols_containing_markers should now ONLY be composed of columns representative of cells that contain all of the necessary positive markers
        intersected_list=list(set.intersection(*map(set,list_of_list_cols_selected))) #This finds the intersection of one or more lists
        #Z=[s for s in cols_containing_marker if '141-SMA' in s] testing if negative case is properly omitted
        #z=curr_data[intersected_list].sort_values(by=curr_data[intersected_list].index[0], ascending=False, axis=1) #find the nonzero values
        #Sum all columns containing cell counts relevant to curr_marker
        summed_columns=curr_data[intersected_list].sum(axis=1)
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
immune_df=immune_df.set_index(groupby)

#Concatenate with existing group
grouped_df=pd.concat([grouped_df,immune_df],axis=1,verify_integrity=True)

#%% Carry over additional metadata
#Adds metadata_to_append variables to grouped_df. If there are multiple annotations per 'Image' entry,
#only the first annotation's metadata will be appended
#Also, put a check to ensure that groupby is equal to 'Image' before executing. Can't append metadata if the groupby is a code or something else
if(groupby=='Image'):
    metadata_df=pd.DataFrame(columns=metadata_to_append,index=grouped_df.index) #Obtain index from grouped_df as another safeguard
    for meta_name in metadata_to_append:
        #Let x be individual 'Image' entries passed. For the metadata_to_append entry (meta_name), find the first element
        #of the list derived exclusively of data where data['Image']==x.
        #PLEASE DEAR GOD never error on the line below because I'll go bald pulling my hair trying to debug this
        curr_meta=[data[data[groupby]==x][meta_name].head(1).tolist() for x in grouped_df.index] #I don't know how or why this works, but it does.
        #Flatten list of list to lists
        curr_meta=[item for sublist in curr_meta for item in sublist]
        metadata_df[meta_name]=curr_meta
    #Concatenate with grouped_df
    grouped_df=pd.concat([grouped_df,metadata_df],axis=1,verify_integrity=True)
else:
    print('groupby is not set to Image. Omitting metadata')

#%% Write to csv
grouped_df.to_csv(os.path.join(figpath,'grouped_data11.csv'),index=True)

'''
To do tmrw: work on negative gating as a special "if" statement
So for a negative case, you'll need to make sure you're selecting columns that contain the positive markers but lack the negative marker
keyword in the name. So for CD16+ GranzymeB-, columns that meet the criteria include things with CD16 but not GranzymeB
'''








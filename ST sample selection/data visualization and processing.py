# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:04:25 2021

@author: Mark Zaidi
"""
#%% load libraries
import pandas
import math
import matplotlib.pyplot as plt
import os
import numpy as np
#%% load data
root_folder=r'C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\ST sample selection data analysis'
annotations=pandas.read_csv(os.path.join(root_folder,r'annotation_measurements.csv'))
cells=pandas.read_csv(os.path.join(root_folder,r'cell_measurements.csv'))
#%% create output folders for figures
output_folder_path=os.path.join(root_folder, r'figures')
# create output directory if not already present
if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)
    
#%% prune data, keeping only the columns you'll use (optional, can comment out)
##list(annotations.columns.values)
##list(cells.columns.values)

#Define table to include only select column names
annotations=annotations[['Image','Class','Area µm^2','Max diameter µm','ROI: 0.50 µm per pixel: DAB: Mean','ROI: 0.50 µm per pixel: DAB: Std.dev.','Num Detections']]
cells=cells[['Image','Parent','Centroid X µm','Centroid Y µm','Cell: DAB OD mean','Cell: DAB OD std dev','Cytoplasm: DAB OD mean','Cytoplasm: DAB OD std dev','Nucleus: DAB OD mean','Nucleus: DAB OD sum','Nucleus: DAB OD std dev']]

##To do: first, only keep columns you will be using for analysis. free up some memory. Then, check .pptx from lab meeting for analysis details
#%% Create empty summary table with columns
columns=['slide','pct_necrosis','cell_density','cells_per_spot','tissue_width','pct_area_covered','mean_tumor_pimo','std_tumor_pimo','mean_cellular_pimo','std_cellular_pimo','mean_Cytoplasm_pimo','std_Cytoplasm_pimo','mean_Nucleus_pimo','std_Nucleus_pimo']
data=[]
#%% Calculate measurements on a per-image basis
images=pandas.unique(annotations.Image)
for slide in images:
    #slide=images[8]
    #slide=images[0]
    necrosis_area=0.0;tumor_area=0.0;viable_cells=0.0; #make all as float
    if ('Necrosis' in annotations.loc[(annotations['Image'] == slide)]['Class'].unique()): #check if a necrosis annotation exists in the image
        necrosis_area=annotations.loc[(annotations['Image'] == slide) & (annotations['Class'] =='Necrosis')].iloc[0]['Area µm^2']
    if ('Tumor' in annotations.loc[(annotations['Image'] == slide)]['Class'].unique()): #check if a tumor annotation exists in the image
        tumor_area=annotations.loc[(annotations['Image'] == slide) & (annotations['Class'] =='Tumor')].iloc[0]['Area µm^2']
        viable_cells=annotations.loc[(annotations['Image'] == slide) & (annotations['Class'] =='Tumor')].iloc[0]['Num Detections']
    else:
        raise ValueError('No tumor annotation present!')
    pct_necrosis=necrosis_area/(necrosis_area+tumor_area)*100 #0-100 range
    cell_density=viable_cells/tumor_area #number of cells per square micrometer
    cells_per_spot=cell_density*(math.pi*(55/2)**2) #Average number of cells per 55-um diameter visium spot
    tissue_width=annotations.loc[(annotations['Image'] == slide) & (annotations['Class'] =='Tissue')].iloc[0]['Max diameter µm'] #longest line that can be fit into Tissue, in um
    pct_area_covered=annotations.loc[(annotations['Image'] == slide) & (annotations['Class'] =='Tissue')].iloc[0]['Area µm^2']/6500**2*100
    mean_tumor_pimo=annotations.loc[(annotations['Image'] == slide) & (annotations['Class'] =='Tumor')].iloc[0]['ROI: 0.50 µm per pixel: DAB: Mean'] #pixelwise mean pimo
    std_tumor_pimo=annotations.loc[(annotations['Image'] == slide) & (annotations['Class'] =='Tumor')].iloc[0]['ROI: 0.50 µm per pixel: DAB: Std.dev.'] #pixelwise std pimo
    #Cell-level calculations
    mean_cellular_pimo=cells.loc[(cells['Image'] == slide) & (cells['Parent'] =='Tumor')]['Cell: DAB OD mean'].mean()
    std_cellular_pimo=cells.loc[(cells['Image'] == slide) & (cells['Parent'] =='Tumor')]['Cell: DAB OD mean'].std()
    #Cytoplasm-level calculations
    mean_Cytoplasm_pimo=cells.loc[(cells['Image'] == slide) & (cells['Parent'] =='Tumor')]['Cytoplasm: DAB OD mean'].mean()
    std_Cytoplasm_pimo=cells.loc[(cells['Image'] == slide) & (cells['Parent'] =='Tumor')]['Cytoplasm: DAB OD mean'].std()
    #Nucleus-level calculations
    mean_Nucleus_pimo=cells.loc[(cells['Image'] == slide) & (cells['Parent'] =='Tumor')]['Nucleus: DAB OD mean'].mean()
    std_Nucleus_pimo=cells.loc[(cells['Image'] == slide) & (cells['Parent'] =='Tumor')]['Nucleus: DAB OD mean'].std()
    # Append data
    data.append([slide,pct_necrosis,cell_density,cells_per_spot,tissue_width,pct_area_covered,mean_tumor_pimo,std_tumor_pimo,mean_cellular_pimo,std_cellular_pimo,mean_Cytoplasm_pimo,std_Cytoplasm_pimo,mean_Nucleus_pimo,std_Nucleus_pimo])
#%% Create and write summary table
summary_table = pandas.DataFrame(data, columns=columns)
summary_table['short_name'] = summary_table['slide'].str.split(' ').str[0] #remove everything after and including the space
summary_table=summary_table.sort_values(['mean_cellular_pimo'])
summary_table.to_excel(r"C:\Users\Mark Zaidi\Documents\QuPath\PIMO GBM related projects\ST sample selection data analysis\summary_table.xlsx")
#%% Data visualization constants
#15A, 19, 33B, and 35 were the samples they used
bars = summary_table['short_name'] #x tick names, reusable
y_pos = range(len(bars)) 
x = np.arange(len(bars))  # the label locations

#%% Percent necrosis
figname='Percent Necrosis'
heights = summary_table['pct_necrosis'] #y values
plt.bar(y_pos, heights) #plot bar
plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('Percentage')
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()

#%% cells per spot
figname='Expected Cells Per Spot'
heights = summary_table['cells_per_spot'] #y values
plt.bar(y_pos, heights) #plot bar
plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('Number of Cells in 55 µm Diameter Spot')
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()

#%% Percentage of slide area covered
figname='Percentage of 6.5x6.5mm Capture Area Covered By Tissue'
heights = summary_table['pct_area_covered'] #y values
plt.bar(y_pos, heights) #plot bar
plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('Percentage')
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()

#%% Percentage of slide area covered
figname='Percentage of 6.5x6.5mm Capture Area Covered By Tissue'
heights = summary_table['pct_area_covered'] #y values
plt.bar(y_pos, heights) #plot bar
plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('Percentage')
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()



#%% mean pimo in cellular vs all areas
figname='Mean Deconvolved Pimonidazole Stain Intensity'
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, summary_table['mean_tumor_pimo'], width=width,label='Pixel Level',align='center')
rects2 = ax.bar(x + width/2, summary_table['mean_cellular_pimo'],  width=width, label='Cell Level',align='center')
plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('DAB Signal Intensity (A.U.)')
plt.legend()
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()

#%% std pimo in cellular vs all areas
figname='Standard Deviation Of Deconvolved Pimonidazole Stain Intensity'
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, summary_table['std_tumor_pimo'], width=width,label='Pixel Level',align='center')
rects2 = ax.bar(x + width/2, summary_table['std_cellular_pimo'],  width=width, label='Cell Level',align='center')
plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('DAB Signal Intensity (A.U.)')
plt.legend()
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()
#TO DO: append RIN analysis data to summary table, and plot with same slide name positions

#%% RIN analysis results
summary_with_RIN=pandas.read_excel(os.path.join(root_folder,r'summary_table_with_RIN.xlsx'))
figname='RNA Integrity Number'
heights = summary_with_RIN['RIN '] #y values
plt.bar(y_pos, heights) #plot bar
plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('RIN score')
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()

#%% mean pimo in nuclear vs cytosolic areas
figname='Cellular Subcompartment Localization of Pimonidazole (v1)'
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, summary_table['mean_Nucleus_pimo'], width=width,label='Nucleus',align='center')
rects2 = ax.bar(x + width/2, summary_table['mean_Cytoplasm_pimo'],  width=width, label='Cytoplasm',align='center')

plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('DAB Signal Intensity (A.U.)')
plt.legend()
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()
#%% mean pimo in nuclear vs cytosolic vs whole cell areas
figname='Cellular Subcompartment Localization of Pimonidazole(v2)'
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, summary_table['mean_Nucleus_pimo'], width=width,label='Nucleus',align='center')
rects2 = ax.bar(x, summary_table['mean_Cytoplasm_pimo'],  width=width, label='Cytoplasm',align='center')
rects3 = ax.bar(x + width, summary_table['mean_cellular_pimo'],  width=width, label='Whole Cell',align='center')

plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('DAB Signal Intensity (A.U.)')
plt.legend()
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()
#%% stdev pimo in nuclear vs cytosolic areas
figname='Cellular Subcompartment Standard Deviation of Pimonidazole'
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, summary_table['std_Nucleus_pimo'], width=width,label='Nucleus',align='center')
rects2 = ax.bar(x + width/2, summary_table['std_Cytoplasm_pimo'],  width=width, label='Cytoplasm',align='center')

plt.xticks(y_pos, bars, rotation=90) # Plot names and rotate
plt.title(figname)
plt.ylabel('DAB Signal Intensity (A.U.)')
plt.legend()
plt.savefig(os.path.join(output_folder_path,figname+'.png'),dpi=800)
plt.close()
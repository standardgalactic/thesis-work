# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:31:51 2022

@author: Mark Zaidi
Script featuring batch workflow for all 12 of our GBM samples. Ideally, these would be the steps required to achieve this:
    1. Create two lists: one for the root directories for each Visium dataset, and another for a library ID
    2. Iteratively read each Visium dataset, and append to a list of Visium datasets
    3. Use .var_names_make_unique() to make each dataset have unique variables
    4. Run the .concatenate function on the first of the 12, concatenating the remaining 11
    5. Proceed with relevant workflow items as outlined in data_integration_tutorial.py
"""

#%% Load packages


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
import os
import SpatialDE
import squidpy as sq

#sc.logging.print_versions() # gives errror!!
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 0
#%% Set constants
dataset_dirs={'16':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\16_1',
              '45B':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\45B',
              '47':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\47',
              '50D':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\50D',
              '15A':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\A1 - 15A\outs',
              '19':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\B1(missing currently) - 19\outs',
              '33B':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\C1 - 33B\outs',
              '35':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\D1_-_35\outs',
              '56':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\V21004-FF__56_A1\outs',
              '60':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\V21004-FF__60_B1\outs',
              '61':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\V21004-FF__61_C1\outs',
              '63':r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\V21004-FF__63_D1\outs',
              }
library_names=list(dataset_dirs.keys())

fig_dir=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\whole_GBM_dataset_analysis'
qc_subdir=os.path.join(fig_dir,'QC')
spatial_gene_subdir=os.path.join(fig_dir,'spatial_gene')
integrated_subdir=os.path.join(fig_dir,'integrated')
spatially_variable_subdir=os.path.join(fig_dir,'spatially_variable_genes')
resolution=0.8 #resolution used for any form of leiden clustering
#%% Create relevant directories
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
if not os.path.exists(qc_subdir):
    os.mkdir(qc_subdir)
if not os.path.exists(spatial_gene_subdir):
    os.mkdir(spatial_gene_subdir)
if not os.path.exists(integrated_subdir):
    os.mkdir(integrated_subdir)
if not os.path.exists(spatially_variable_subdir):
    os.mkdir(spatially_variable_subdir)
#%% Iteratively read datasets
datasets=[]
#For loop will read and append each dataset to a list called datasets
for curr_key in dataset_dirs:
#curr_key=list(dataset_dirs.keys())[0]
    curr_dataset=sc.read_visium(path=dataset_dirs[curr_key],count_file='filtered_feature_bc_matrix.h5',source_image_path=dataset_dirs[curr_key]+'\\spatial\\tissue_hires_image.png',library_id=curr_key)
    curr_dataset.var_names_make_unique()
    datasets.append(curr_dataset)
#first item in list of datasets is used to start the concatenation
first_dataset=datasets[0]
#concatenation is called on the first dataset, and applied to all but the first dataset item. Keys of the dataset_dirs serve as the library_ID
adata = first_dataset.concatenate(
    datasets[1:],
    batch_key="library_id",
    uns_merge="unique",
    batch_categories=dataset_dirs.keys()
)
#%% Quality control

# add info on mitochondrial and hemoglobin genes to the objects.
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var['hb'] = adata.var_names.str.contains(("^HB")) #Original regex was "^Hb.*-"
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','hb'], percent_top=None, log1p=False, inplace=True)
plt.close('all')

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_hb'],
             jitter=0.4, groupby = 'library_id', rotation= 45)
plt.savefig(qc_subdir + r'\QC_unfiltered.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% Filter data
#Select all spots with less than 25% mitocondrial reads, less than 20% hb-reads and 1000 detected genes. You must judge for yourself based on your knowledge of the tissue what are appropriate filtering criteria for your dataset.
#note, this is a filter applied to all libraries in adata, and removes only spots based on the below criteria
keep = (adata.obs['pct_counts_hb'] < 20) & (adata.obs['pct_counts_mt'] < 25) & (adata.obs['n_genes_by_counts'] > 1000)
#print(sum(keep))

adata = adata[keep,:]

#%%Plot remaining datapoints
# for library in dataset_dirs:
#     plt.close('all')
#     sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = ["total_counts", "n_genes_by_counts",'pct_counts_mt', 'pct_counts_hb'])
#     plt.savefig(qc_subdir + r'\Filtered'+library+'.png',dpi=400,pad_inches=0.1,bbox_inches='tight')

#%% Filter genes and show top expressed genes
#Remove hemoglobin and mitochondrial genes
mito_genes = adata.var_names.str.startswith('MT-')
hb_genes = adata.var_names.str.contains('^HB')
remove = np.add(mito_genes, hb_genes)
#remove[adata.var_names == "Bc1"] = True
keep = np.invert(remove)

adata = adata[:,keep]
plt.close('all')
#Fraction of counts assigned to each gene over all spots.
sc.pl.highest_expr_genes(adata, n_top=20)
plt.savefig(fig_dir + r'\Top_expressed_genes.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% ANALYSIS
#%%Take the union of all libraries' highly variable genes
# save the counts to a separate object for later, we need the normalized counts in raw for DEG dete
counts_adata = adata.copy()

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
# take 1500 variable genes per batch and then use the union of them.
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=1500, inplace=True, batch_key="library_id")

# subset for variable genes
adata.raw = adata
adata = adata[:,adata.var.highly_variable_nbatches > 0]

# scale data
sc.pp.scale(adata)
#%% Plot VEGFA staining for each
# genes_to_preview=["VEGFA"] #can have multiple str vals
vmin=None;vmax=None
vmin=[0] #can have multiple float vals
vmax=[4] #can have multiple float vals
# for library in dataset_dirs:
#     plt.close('all')

#     sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = genes_to_preview,vmin=vmin,vmax=vmax)
#     plt.savefig(spatial_gene_subdir+'\\'+'_'.join(genes_to_preview)+'_'+library+'.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% Alternate method of plotting VEGFA staining
plt.close('all')

fig, axs = plt.subplots(3, 4, figsize=(15, 10))
unraveled_axs=axs.ravel() #VERY IMPORTANT, TUTORIAL ASSUMES YOU ONLY HAVE 1 ROW OF PLOTS

#I don't exactly know what the below does, but my guess is that it keeps cluster colors consistent (e.g. cluster 6 will always be bright green)
gene_to_plot='LTF'

for i, library in enumerate(
    library_names
):
    ad = adata[adata.obs.library_id == library, :].copy()
    sc.pl.spatial(
        ad,
        img_key="hires",
        library_id=library,
        color=gene_to_plot,
        size=1.5,
        vmin=vmin,
        vmax=vmax,
        legend_loc='on data',
        show=False,
        ax=unraveled_axs[i],
        title=library
    )

plt.tight_layout()
plt.savefig(spatial_gene_subdir + '\\'+gene_to_plot+'_spatial.png',dpi=400,pad_inches=0.1,bbox_inches='tight')

#%% Dimensionality reduction and clustering WITHOUT integration
#Here, we perform UMAP, PCA, and leiden clustering, with adata that has only been concatenated to other adatas
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="non_integrated_clusters",resolution=resolution)
plt.close('all')
#sc.set_figure_params(scanpy=True,facecolor=(1,1,1))
sc.pl.umap(
    adata, color=["non_integrated_clusters", "library_id"], palette=sc.pl.palettes.default_20,
    legend_loc='on data'
)

plt.savefig(fig_dir + '\\non_integrated_clusters.png',dpi=400,pad_inches=0.1,bbox_inches='tight')

#%% Create spatial plot of clustering performed WITHOUT integration
# #As we are plotting the two sections separately, we need to make sure that they get the same colors by 
# #fetching cluster colors from a dict.
# clusters_colors = dict(
#     zip([str(i) for i in range(len(adata.obs.non_integrated_clusters.cat.categories))], adata.uns["non_integrated_clusters_colors"])
# )
# plt.close('all')

# fig, axs = plt.subplots(3, 4, figsize=(15, 10))
# unraveled_axs=axs.ravel() #VERY IMPORTANT, TUTORIAL ASSUMES YOU ONLY HAVE 1 ROW OF PLOTS

# #I don't exactly know what the below does, but my guess is that it keeps cluster colors consistent (e.g. cluster 6 will always be bright green)
# for i, library in enumerate(
#     library_names
# ):
#     ad = adata[adata.obs.library_id == library, :].copy()
#     sc.pl.spatial(
#         ad,
#         img_key="hires",
#         library_id=library,
#         color="non_integrated_clusters",
#         size=1.5,
#         palette=[
#             v
#             for k, v in clusters_colors.items()
#             if k in ad.obs.non_integrated_clusters.unique().tolist()
#         ],
#         legend_loc='on data',
#         show=False,
#         ax=unraveled_axs[i],
#         title=library
#     )

# plt.tight_layout()
# plt.savefig(fig_dir + '\\non_integrated_clusters_spatial.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
# #We can see that without proper integration techniques, the clusters are largely library(sample) specific
# #Meaning that despite us calculating the clusters on all spots from all datasets, clustering is largely driven
# #by batch effects. See, I told you it's not as simple as just combining all the datasets :P

#%% INTEGRATION with scanorama
#Apparently, the preprocessing done on the non-integrated data is required for this to run properly.
#Specifically, sc.pp.neighbors and sc.tl.umap, but not sc.tl.pca

#Note, if this doesn't work, another potential method is BBKNN https://scanpy-tutorials.readthedocs.io/en/latest/integrating-data-using-ingest.html#Using-BBKNN
plt.close('all')
#sc.tl.pca(adata)

#
adatas = {}
for batch in library_names:
    adatas[batch] = adata[adata.obs['library_id'] == batch,]
#convert to list of AnnData objects
adatas = list(adatas.values())

# run scanorama.integrate
#Basically runs the function on a list of anndata objects, where each object is a library(sample)
print('STARTING INTEGRATION. If this takes more than 5 minutes to run, something\'s wrong. May need to swap for scanorama.integrate()')
#I have no idea why, but you must put a delay before running scanorama.integrate_scanpy, else it'll cap at 66% CPU and run indefinitely.
#It should not take more than a minute to finish the integration.
plt.pause(10)
scanorama.integrate_scanpy(adatas, dimred = 50)
print('FINISHED INTEGRATION')

# Get all the integrated matrices.
scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]

# make into one matrix.
all_s = np.concatenate(scanorama_int)
#print(all_s.shape)

# add to the AnnData object
adata.obsm["Scanorama"] = all_s
#Run DR and clustering
print('STARTING NEIGHBOURS')
sc.pp.neighbors(adata, use_rep="Scanorama")
print('STARTING UMAP')
sc.tl.umap(adata)
print('STARTING CLUSTERING')
sc.tl.leiden(adata, key_added="integrated_clusters",resolution=resolution)

#%% Plot the UMAP
# plt.close('all')

# sc.pl.umap(
#     adata, color=["integrated_clusters", "library_id"], palette=sc.pl.palettes.default_20,
#     legend_loc='on data',
#     legend_fontsize='medium',
#     size=10
# )
# plt.savefig(fig_dir + '\\scanorama_integrated_clusters.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
# #Now, you can see that datapoints are more homogenously distributied from both samples throughout the UMAP.
# #Furthermore, the clusters that form are more likely to exist across libraries (samples)
#%% Create spatial plot of clustering performed WITH integration
# #As we are plotting the two sections separately, we need to make sure that they get the same colors by 
# #fetching cluster colors from a dict.
# clusters_colors = dict(
#     zip([str(i) for i in range(len(adata.obs.integrated_clusters.cat.categories))], adata.uns["integrated_clusters_colors"])
# )
# plt.close('all')

# fig, axs = plt.subplots(3, 4, figsize=(15, 10))
# unraveled_axs=axs.ravel() #VERY IMPORTANT, TUTORIAL ASSUMES YOU ONLY HAVE 1 ROW OF PLOTS

# #I don't exactly know what the below does, but my guess is that it keeps cluster colors consistent (e.g. cluster 6 will always be bright green)
# for i, library in enumerate(
#     library_names
# ):
#     ad = adata[adata.obs.library_id == library, :].copy()
#     sc.pl.spatial(
#         ad,
#         img_key="hires",
#         library_id=library,
#         color="integrated_clusters",
#         size=1.5,
#         palette=[
#             v
#             for k, v in clusters_colors.items()
#             if k in ad.obs.integrated_clusters.unique().tolist()
#         ],
#         legend_loc='right margin',
#         legend_fontsize='x-small',
#         show=False,
#         ax=unraveled_axs[i],
#         title=library
#     )

# plt.tight_layout()
# plt.savefig(fig_dir + '\\scanorama_integrated_clusters_spatial.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
# #Here, we see clusters in common between our two samples. I'd say scanorama seems to do a reasonable job of controlling batch effects
# print('Make sure you have less than 21 clusters, else the colors for each cluster will start to overlap')
#%% Identify cluster marker genes
# run t-test 
sc.tl.rank_genes_groups(adata, "integrated_clusters")
#%% plot as heatmap for cluster 5 genes
# for cluster in range(len(adata.obs.integrated_clusters.cat.categories)):
#     plt.close('all')
#     sc.pl.rank_genes_groups_heatmap(adata, groups=[str(cluster)], n_genes=10, groupby="integrated_clusters")
#     plt.savefig(integrated_subdir + '\\cluster_'+str(cluster)+'__marker_genes.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% Plot top marker gene for each cluster, as a spatial plot with all 12 cases
# for cluster in range(len(adata.obs.integrated_clusters.cat.categories)):
#     plt.close('all')
#     #Identifies the top differentially expressed gene for the cluster
#     top_cluster_gene=sc.get.rank_genes_groups_df(adata,group=str(cluster)).names[0]

#     print(top_cluster_gene)
    

#     fig, axs = plt.subplots(3, 4, figsize=(15, 10))
#     unraveled_axs=axs.ravel() #VERY IMPORTANT, TUTORIAL ASSUMES YOU ONLY HAVE 1 ROW OF PLOTS
    
#     for i, library in enumerate(
#         library_names
#     ):
#         ad = adata[adata.obs.library_id == library, :].copy()
#         sc.pl.spatial(
#             ad,
#             img_key="hires",
#             library_id=library,
#             color=top_cluster_gene,
#             size=1.5,
#             vmin=vmin,
#             vmax=vmax,
#             legend_loc='on data',
#             show=False,
#             ax=unraveled_axs[i],
#             title=library
#         )
    
#     plt.tight_layout()
#     plt.savefig(integrated_subdir + '\\cluster_'+str(cluster)+'_'+top_cluster_gene+'.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% Identify spatially variable genes
print('COMPUTING sq.gr.spatial_neighbors')
sq.gr.spatial_neighbors(adata,library_key='library_id')

print('IDENTIFYING SPATIALLY VARIABLE GENES')

#First, subset the list of genes to only include those that are spatially variable (as computed during the QC step)
genes = adata[:, adata.var.highly_variable].var_names.values[:1000]
#Now, we use the Moran's I test to identify spatially variable genes
sq.gr.spatial_autocorr(
    adata,
    mode="moran",
    genes=genes,
    n_perms=100,
    n_jobs=30,
    show_progress_bar=False
)
#%% Plot spatially variable genes
#The results are found in the uns (unstructured) portion of adata (adata.uns["moranI"])
#Get the top 50 spatially variable genes:
spat_var_genes=adata.uns["moranI"].head(50)
spat_var_gene_names=spat_var_genes.index.values.tolist()[:7] #Get the top 9 for plotting
spat_var_gene_names.append('integrated_clusters') #tag on the cluster as the last plot to visualize

# for library in library_names:
#     plt.close('all')
#     sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color=spat_var_gene_names,vmin=vmin,vmax=vmax)
#     manager = plt.get_current_fig_manager()
#     manager.window.showMaximized()
#     plt.savefig(spatially_variable_subdir +'\\'+ library+r'_Spatially_Variable_genes.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% Neighbourhood enrichment
# sq.gr.nhood_enrichment(adata, cluster_key="integrated_clusters",n_jobs=30,show_progress_bar=False)
# #Plot the enrichment matrix
# plt.close('all')
# sq.pl.nhood_enrichment(adata, cluster_key="integrated_clusters")
# plt.savefig(fig_dir + r'\neighbourhood_enrichment.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#%%Co-occurence across spatial dimensions
# #similar to above, but doesn't operate on a connectivity matrix, rather the original spatial coordinates.
# #Formula is https://squidpy.readthedocs.io/en/latest/auto_tutorials/tutorial_visium_hne.html#co-occurrence-across-spatial-dimensions
# #Basically, computes the probability of finding one specific cluster in increasing distances around spots of all other clusters
# #For now, lets use cluster_to_heatmap as the cluster to compute the distance to (our hypoxic cluster in 50D)
# print('COMPUTING sq.gr.co_occurrence')
# sq.gr.co_occurrence(adata, cluster_key="integrated_clusters", n_jobs=30,show_progress_bar=False)
# plt.close('all')

# sq.pl.co_occurrence(
#     adata,
#     cluster_key="integrated_clusters",
#     clusters="18",
#     figsize=(8, 4),
#     legend_kwargs={'fontsize':'x-small'}
# )
# plt.savefig(fig_dir + r'\Co-occurence_across_spatial_dimensions.png',dpi=800,pad_inches=0.1,bbox_inches='tight')


#%%Data expoort
#%% Cluster marker genes
#Goal here is to identify "marker genes", which are a set of genes used to best differentiate clusters from one another
#Rank genes for characterizing groups. Here, we rank the genes that are most differentially expressed across the leiden clusters
'''
Extract the results of rank_genes_groups into a dataframe. In sort, the following columns describe:
    group: cluster group, as defined by the rank_gene_groups groupby argument. Same genes can be present in multiple groups, but with different scores
    names: gene names, I mean, what did you expect?
    scores: the ranked score of that gene to differentiate the cluster group. For example, the score ranked_df['score'][0] represents how well ranked_df['name'][0] differentiates cluster ranked_df['group'][0]
        In Loupe, the group is directly appended into the column names. As a result, the dataframe is effectively widened for each cluster x colname combination. Please don't ask me to "make this look like how Loupe does it". I mean, I will if you ask nicely, but I really don't want to
        If ABSOLUTELY needed, here's what to do. For curr_data where ranked_df['group']=curr_group, append curr_group to the col names of the dataframe. Then, horizontally concatenate the dataframe. See, it's not that bad!
    logfoldchanges: log2 fold change for each gene for each group. Comparing the fold difference of the gene in the current group relative to all other groups
    pvals: p values for the statistical method in rank_gene_groups method argument
    pvals_adj: correction method as specified by rank_gene_groups corr_method argument. If not specified, defaults to benjamini-hochberg
In rank_genes_groups_df, you can specify additional arguments such as which group to export, and cutoffs for pvalues or fold changes
'''    
ranked_df=sc.get.rank_genes_groups_df(adata,group=None)
ranked_df.to_csv(fig_dir + r'\gene_cluster_rankings.csv',index=False)
#%% Restructure ranked_df to be in a format similar to loupe
list_of_df=[]
for curr_group in ranked_df['group'].unique():
#curr_group=ranked_df['group'].unique()[0]
    curr_data=ranked_df[ranked_df['group']==curr_group]
    del curr_data['group']
    curr_data=curr_data.sort_values('names')
    curr_data=curr_data.reset_index(drop=True)

    curr_data=curr_data.add_suffix('_'+ curr_group)
    list_of_df.append(curr_data)
ranked_df_loupe_style=pd.concat(list_of_df,axis=1)
ranked_df_loupe_style.to_csv(fig_dir + r'\gene_cluster_rankings_loupe_style.csv',index=False)
#%% Write out list of spatially variable gene scores
adata.uns["moranI"].to_csv(spatially_variable_subdir + r'\MoranI_score.csv')
















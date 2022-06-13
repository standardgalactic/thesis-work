# -*- coding: utf-8 -*-
"""
Learning about data integration from https://nbisweden.github.io/workshop-scRNAseq/labs/compiled/scanpy/scanpy_07_spatial.html


Created on Fri May 20 12:28:46 2022

@author: Mark Zaidi
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
#sc.logging.print_versions() # gives errror!!
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1
#%% Load ST data
#Here, we substitute loading of the tutorial datasets with two of our own. My hopes is that this is scalable to include all 12 of our cases


# Load two of our datasets
data_dir_anterior=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\50D'
data_dir_posterior=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\V21004-FF__56_A1\outs'
fig_dir=os.path.join(data_dir_anterior,'Scanorama_integration_figures')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

#Need to use scanpy's import function, if we intend to use some of scanpy's visualization tools
adata_anterior = sc.read_visium(path=data_dir_anterior,count_file='filtered_feature_bc_matrix.h5',source_image_path=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\50D\spatial\tissue_hires_image.png',library_id="50D")
adata_posterior = sc.read_visium(path=data_dir_posterior,count_file='filtered_feature_bc_matrix.h5',source_image_path=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\V21004-FF__56_A1\outs\spatial\tissue_hires_image.png',library_id='56')

# adata_anterior = sc.datasets.visium_sge(
#     sample_id="V1_Mouse_Brain_Sagittal_Anterior"
# )
# adata_posterior = sc.datasets.visium_sge(
#     sample_id="V1_Mouse_Brain_Sagittal_Posterior"
# )

#Ultimately, dataset reading would be done in a for loop, iterating for each user-specified directory. 
#source_image_paths could be created programmatically (e.g. appending spatial\tissue_hires_image.png to each directory)
#Library IDs would be manually specified, so what I'm thinking is that the reading will be done on a for loop that iterates on the zip
#of two variables: data_dirs and libraries



adata_anterior.var_names_make_unique()
adata_posterior.var_names_make_unique()


#%% Merge into one dataset
library_names = ["50D", "56"] #Library names would be manually set, as mentioned above in libraries
#For multiple datasets, perhaps we can iteratively concatenate each dataset? or pass a list of adatas?
adata = adata_anterior.concatenate(
    adata_posterior,
    batch_key="library_id",
    uns_merge="unique",
    batch_categories=library_names
)
#%% Quality control

# add info on mitochondrial and hemoglobin genes to the objects.
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var['hb'] = adata.var_names.str.contains(("^HB")) #Original regex was "^Hb.*-"
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','hb'], percent_top=None, log1p=False, inplace=True)
plt.close('all')

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_hb'],
             jitter=0.4, groupby = 'library_id', rotation= 45)
plt.savefig(fig_dir + r'\QC.png',dpi=400,pad_inches=0.1,bbox_inches='tight')

#%% Filter data
#Select all spots with less than 25% mitocondrial reads, less than 20% hb-reads and 1000 detected genes. You must judge for yourself based on your knowledge of the tissue what are appropriate filtering criteria for your dataset.
#note, this is a filter applied to all libraries in adata, and removes only spots based on the below criteria
keep = (adata.obs['pct_counts_hb'] < 20) & (adata.obs['pct_counts_mt'] < 25) & (adata.obs['n_genes_by_counts'] > 1000)
print(sum(keep))

adata = adata[keep,:]


for library in library_names:
    plt.close('all')
    sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = ["total_counts", "n_genes_by_counts",'pct_counts_mt', 'pct_counts_hb'])
    plt.savefig(fig_dir + r'\Filtered'+library+'.png',dpi=400,pad_inches=0.1,bbox_inches='tight')

#%% Filter genes and show top expressed genes
#Remove hemoglobin and mitochondrial genes
mito_genes = adata.var_names.str.startswith('MT-')
hb_genes = adata.var_names.str.contains('^HB')
remove = np.add(mito_genes, hb_genes)
remove[adata.var_names == "Bc1"] = True
keep = np.invert(remove)

adata = adata[:,keep]
plt.close('all')
#Fraction of counts assigned to each gene over all cells.
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

for library in library_names:
    plt.close('all')

    sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = ["VEGFA"])
    plt.savefig(fig_dir + '\\'+library+'_VEGFA_expression.png',dpi=400,pad_inches=0.1,bbox_inches='tight')

#%% Dimensionality reduction and clustering WITHOUT integration
#Here, we perform UMAP, PCA, and leiden clustering, with adata that has only been concatenated to other adatas
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="non_integrated_clusters")
plt.close('all')

sc.pl.umap(
    adata, color=["non_integrated_clusters", "library_id"], palette=sc.pl.palettes.default_20
)
plt.savefig(fig_dir + '\\non_integrated_clusters.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% Create spatial plot of clustering performed WITHOUT integration
#As we are plotting the two sections separately, we need to make sure that they get the same colors by 
#fetching cluster colors from a dict.
clusters_colors = dict(
    zip([str(i) for i in range(len(adata.obs.non_integrated_clusters.cat.categories))], adata.uns["non_integrated_clusters_colors"])
)
plt.close('all')

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
#I don't exactly know what the below does, but my guess is that it keeps cluster colors consistent (e.g. cluster 6 will always be bright green)
for i, library in enumerate(
    library_names
):
    ad = adata[adata.obs.library_id == library, :].copy()
    sc.pl.spatial(
        ad,
        img_key="hires",
        library_id=library,
        color="non_integrated_clusters",
        size=1.5,
        palette=[
            v
            for k, v in clusters_colors.items()
            if k in ad.obs.non_integrated_clusters.unique().tolist()
        ],
        #legend_loc=None,
        show=False,
        ax=axs[i],
    )

plt.tight_layout()
plt.savefig(fig_dir + '\\non_integrated_clusters_spatial.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#We can see that without proper integration techniques, the clusters are largely library(sample) specific
#Meaning that despite us calculating the clusters on all spots from all datasets, clustering is largely driven
#by batch effects. See, I told you it's not as simple as just combining all the datasets :P
#%% INTEGRATION with scanorama
#Note, if this doesn't work, another potential method is BBKNN https://scanpy-tutorials.readthedocs.io/en/latest/integrating-data-using-ingest.html#Using-BBKNN
adatas = {}
for batch in library_names:
    adatas[batch] = adata[adata.obs['library_id'] == batch,]
#convert to list of AnnData objects
adatas = list(adatas.values())

# run scanorama.integrate
#Basically runs the function on a list of anndata objects, where each object is a library(sample)
scanorama.integrate_scanpy(adatas, dimred = 50)

# Get all the integrated matrices.
scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]

# make into one matrix.
all_s = np.concatenate(scanorama_int)
print(all_s.shape)

# add to the AnnData object
adata.obsm["Scanorama"] = all_s
#Run DR and clustering
sc.pp.neighbors(adata, use_rep="Scanorama")
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="integrated_clusters")

#Plot the UMAP
plt.close('all')

sc.pl.umap(
    adata, color=["integrated_clusters", "library_id"], palette=sc.pl.palettes.default_20
)
plt.savefig(fig_dir + '\\scanorama_integrated_clusters.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#Now, you can see that datapoints are more homogenously distributied from both samples throughout the UMAP.
#Furthermore, the clusters that form are more likely to exist across libraries (samples)
#%% Create spatial plot of clustering performed WITH integration
#As we are plotting the two sections separately, we need to make sure that they get the same colors by 
#fetching cluster colors from a dict.
clusters_colors = dict(
    zip([str(i) for i in range(len(adata.obs.integrated_clusters.cat.categories))], adata.uns["integrated_clusters_colors"])
)
plt.close('all')

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
#I don't exactly know what the below does, but my guess is that it keeps cluster colors consistent (e.g. cluster 6 will always be bright green)
for i, library in enumerate(
    library_names
):
    ad = adata[adata.obs.library_id == library, :].copy()
    sc.pl.spatial(
        ad,
        img_key="hires",
        library_id=library,
        color="integrated_clusters",
        size=1.5,
        palette=[
            v
            for k, v in clusters_colors.items()
            if k in ad.obs.integrated_clusters.unique().tolist()
        ],
        #legend_loc=None,
        show=False,
        ax=axs[i],
    )

plt.tight_layout()
plt.savefig(fig_dir + '\\scanorama_integrated_clusters_spatial.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#Here, we see clusters in common between our two samples. I'd say scanorama seems to do a reasonable job of controlling batch effects
#%% Identify cluster marker genes
# run t-test 
sc.tl.rank_genes_groups(adata, "integrated_clusters")
# plot as heatmap for cluster 5 genes
plt.close('all')

sc.pl.rank_genes_groups_heatmap(adata, groups="5", n_genes=10, groupby="integrated_clusters")
plt.savefig(fig_dir + '\\scanorama_integrated_cluster_5_marker_genes.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% Spatially plot top marker genes
#plot onto spatial location
top_genes = sc.get.rank_genes_groups_df(adata, group='5',log2fc_min=0)['names'][:4]


for library in library_names:
    plt.close('all')
    sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = top_genes)
    plt.savefig(fig_dir + '\\'+library+'_cluster_5_marker_genes.png',dpi=400,pad_inches=0.1,bbox_inches='tight')
#%% Identify spatially variable genes with SpatialDE
#First, we convert normalized counts and coordinates to pandas dataframe, needed for inputs to spatialDE.
counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names) #changed from adata.X.todense()
coord = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
results = SpatialDE.run(coord, counts)

#We concatenate the results with the DataFrame of annotations of variables: `adata.var`.
results.index = results["g"]
adata.var = pd.concat([adata.var, results.loc[adata.var.index.values, :]], axis=1)

#Then we can inspect significant genes that varies in space and visualize them with `sc.pl.spatial` function.
top4=results.sort_values("qval").head(4)
#%% Spatially plot top spatial  genes
#MARK NOTE: Consider swapping with some more modern approach, e.g. the one used in Squidpy. This is slow, and the genes don't look cool
#plot onto spatial location


# for library in library_names:
#     plt.close('all')
#     sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = top4.index)
#     plt.savefig(fig_dir + '\\'+library+'_spatially_variable_genes.png',dpi=400,pad_inches=0.1,bbox_inches='tight')

'''
Left off at section 4 of https://nbisweden.github.io/workshop-scRNAseq/labs/compiled/scanpy/scanpy_07_spatial.html

Best to leave off here, as the subsequent steps rely on having a complementary and annotated scRNAseq dataset.
'''




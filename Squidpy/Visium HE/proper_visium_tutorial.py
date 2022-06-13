# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:05:29 2022

@author: Mark Zaidi

Scanpy preprocessing DO FIRST: https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
Squidpy part of tutorial:https://squidpy.readthedocs.io/en/latest/auto_tutorials/tutorial_visium_hne.html

Description of adata elements https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html#anndata.AnnData



#How to consolidate multiple datasets
#for each sample, perform clustering. Then identify, which cluster is associated with which pathway
Next, see if these pathway-clusters are present in different cases. This can be done through DGx of clusters, find key genes differentially expressed
Then see if those genes are present across multiple clusters. 
Better to make a separate script for dealing with multiple samples. See https://nbisweden.github.io/workshop-scRNAseq/labs/compiled/scanpy/scanpy_07_spatial.html
#Send sheila data in a format similar to the loupe exports, where you have columns correspond to clusters, and rows corresponding to gene measurements



"""
#%% Import libraries
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import pandas as pd
import SpatialDE #use pip install SpatialDE to install. Yes, I know we shouldn't use pip to install packages in a conda environment, but so what? Sue me :P
import time #If only I could import more time in real life...
import squidpy as sq
import anndata as ad
import numpy as np
from stardist.models import StarDist2D #This and the import below are for StarDist segmentation of cells
from csbdeep.utils import normalize
sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1
'''



SCANPY PART OF TUTORIAL



'''
#%% Read the data
# Load one of our datasets
data_dir=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\50D'
fig_dir=os.path.join(data_dir,'Squidpy_figures')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
#adata will also contain the visium image. We need to put the image into an im.ImageContainer object with the appropriate scale factor
#adata = sq.read.visium(path=data_dir,counts_file='filtered_feature_bc_matrix.h5')
#Need to use scanpy's import function, if we intend to use some of scanpy's visualization tools
adata = sc.read_visium(path=data_dir,count_file='filtered_feature_bc_matrix.h5',source_image_path=r'C:\Users\Mark Zaidi\Documents\Visium\GBM_datasets\50D\spatial\tissue_hires_image.png' )
adata.var_names_make_unique()
#I have no idea what the two lines below do
#Oh okay, I kind of get it now. Genes that start with MT are mitochondrial genes. Those are usually used for QC
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
#%% QC and preprocessing
#Create plot showing number of transcript per spot, I'm guessing
plt.close('all')

fig, axs = plt.subplots(1, 4, figsize=(15, 4))
#adata.obs relates to "row" stats, specifically relating to the multiple datapoints (observations)
#See https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.calculate_qc_metrics.html for what the different plots mean
#Total UMI counts per cell
sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
#Total counts per cell of cells with less than 10000 UMIs, binned into 40 bins
sns.distplot(adata.obs["total_counts"][adata.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
#number of genes with at least 1 count in a cell, calculated for all cells, binned into 60 bins
sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
#number of genes with at least 1 count in a cell, calculated for all cells. Honestly, this is so poorly explained, I give up
#I think it find the number of genes with at least 1 count in all cells in the dataset, and finds out how many genes are present in each cell
sns.distplot(adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])
#Filter out cells with abnormally low or high counts, and genes detected in only a few cells
#Save figure
plt.savefig(fig_dir + r'\QC.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

sc.pp.filter_cells(adata, min_counts=5000)
sc.pp.filter_cells(adata, max_counts=35000)
adata = adata[adata.obs["pct_counts_mt"] < 20] #Very high, see what others have done. Might be different for human. Check w/ PMGC
print(f"#cells after MT filter: {adata.n_obs}")
sc.pp.filter_genes(adata, min_cells=10)

#Normalize data, and detect highly variable genes
sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

#%% Manifold embedding and clustering based on transcriptional similarity
#Run a bunch of different dimensionality reduction and cluster based on this
plt.close('all')

sc.pp.pca(adata) #compute PCA coords
sc.pp.neighbors(adata) #compute neighbourhood graph (basically UMAP with extra steps). Required as a prerequisite for leiden clustering
sc.tl.umap(adata) #umap coords
sc.tl.leiden(adata, key_added="leiden_gene") #Run leiden clustering. Can specify resolution to control the number of clusters created

#Plot and see if any clusters are dependent on total_counts per cell, or number of genes with at least 1 count in a cell (shouldn't be the case after normalization)
plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "leiden_gene"], wspace=0.4)
plt.savefig(fig_dir + r'\leiden_gene_cluster.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#%% Visualization in spatial coordinates
#I think n_genes_by_counts is the number of genes with at least 1 count in all cells in the dataset, and finds out how many genes are present in each cell
plt.close('all')

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(adata, img_key="hires", color=["total_counts", "n_genes_by_counts"])
plt.savefig(fig_dir + r'\QC_spatial.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#FAQ: why are some spots on tissue missing? A: they could have been filtered out during preprocessing due to low transcript or gene counts
#Visualize clustering in gene-expression space with original image underlaid
#%% Apply nonspatial clustering to preview cluster locations on image
plt.close('all')

sc.pl.spatial(adata, img_key="hires", color="leiden_gene", size=1.5)
plt.savefig(fig_dir + r'\nonspatial_clustering.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%% Cluster marker genes
#Goal here is to identify "marker genes", which are a set of genes used to best differentiate clusters from one another
#Rank genes for characterizing groups. Here, we rank the genes that are most differentially expressed across the leiden clusters
sc.tl.rank_genes_groups(adata, groupby="leiden_gene", method="t-test")
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
#%%Preview clusters side by side with a marker of hypoxia
plt.close('all')

sc.pl.spatial(adata, img_key="hires", color=["leiden_gene", "VEGFA"])
plt.savefig(fig_dir + r'\clustering and marker preview.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%%Identify cluster marker genes
plt.close('all')
#Visualize the top 10 genes that differentiate a specific cluster. Here, we use the hypoxic cluster, cluster 6
#Note, you can specify more than one groups
cluster_to_heatmap=['6','11','7']
sc.pl.rank_genes_groups_heatmap(adata, groups=cluster_to_heatmap, n_genes=10, groupby="leiden_gene") #change ngenes to see if dendrogram changes
plt.savefig(fig_dir + r'\cluster '+','.join(cluster_to_heatmap)+' marker genes.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
plt.close('all')

#From all of this, we show that non-spatial clustering can identify regions of hypoxia, however here, this is separated into clusters 6 and 11.
#Clusters 6 and 11 both have VEGFA expression, indicating that it's a hypoxic cluster. However, they're spatially discrete. Non-spatial leiden
#clustering shows considerable gene expression similarities between the two, and if the clustering was slightly less stringent, they would be one
#cluster. We can even force them to become 1 cluster by tuning the resolution parameter of sc.tl.leiden. Now, if we perform spatial clustering,
#will these two clusters become resolved?
#%% Spatially variable genes. May want to comment all of this below out, as it's covered through the squidpy tutorial
# start_time = time.time()
# counts = pd.DataFrame(adata.X.todense(), columns=adata.var_names, index=adata.obs_names)
# coord = pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)
# results = SpatialDE.run(coord, counts)
# print('It took', time.time()-start_time, 'seconds for the spatial stuff. Tutorial shows 1hr 24mins.')
# #Bring the results back into the adata object
# results.index = results["g"]
# adata.var = pd.concat([adata.var, results.loc[adata.var.index.values, :]], axis=1)
# #Find the top spatially variable genes
# top_spatial=results.sort_values("qval").head(10)
# spatial_genes_to_plot=top_spatial['g'][0:8].tolist()
# plt.close('all')

# sc.pl.spatial(adata, img_key="hires", color=spatial_genes_to_plot, alpha=0.7,size=1.5)
# plt.savefig(fig_dir + r'\spatialDE_top8.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%%
'''



SQUIDPY PART OF TUTORIAL



'''
print('STARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY\nSTARTING SQUIDPY')

#%%Calculate image features
#Preprocessing to extract relevant data for image feature processing
#Dataset name (a.k.a. library?) will be the only key in the dict
dataset_name = list(adata.uns['spatial'].keys())[0]
#Extract the high res Visium images scale factor, ignoring the low res one
HE_res_scale=adata.uns['spatial'][dataset_name]['scalefactors']['tissue_hires_scalef']
#Scanpy labels the high resolution image as hires, whereas squidpy is hires_image
img=sq.im.ImageContainer.from_adata(adata=adata,img_key='hires',scale=HE_res_scale)
# calculate features for different scales (higher value means more context)
for scale in [1.0, 2.0]:
    feature_name = f"features_summary_scale{scale}"
    sq.im.calculate_image_features(
        adata,
        img.compute(),
        features="summary",
        key_added=feature_name,
        n_jobs=15,
        scale=scale,
        show_progress_bar=False
    )


# combine features in one dataframe
adata.obsm["image_intensity_features"] = pd.concat(
    [adata.obsm[f] for f in adata.obsm.keys() if "features_summary" in f], axis="columns"
)
# make sure that we have no duplicated feature names in the combined table
adata.obsm["image_intensity_features"].columns = ad.utils.make_index_unique(adata.obsm["image_intensity_features"].columns)
#obsm refers to multidimensional observation annotations
#%% Create a new clustering method using the extracted image features
#Note, we have renamed the keys slightly to prevent overwriting previous DR and clustering generated through gene space data
# helper function returning a clustering
def cluster_features(features: pd.DataFrame, like=None) -> pd.Series:
    """
    Calculate leiden clustering of features.

    Specify filter of features using `like`.
    """
    # filter features
    if like is not None:
        features = features.filter(like=like)
    # create temporary adata to calculate the clustering
    adata = ad.AnnData(features)
    # important - feature values are not scaled, so need to scale them before PCA
    sc.pp.scale(adata)
    # calculate leiden clustering
    sc.pp.pca(adata, n_comps=min(10, features.shape[1] - 1))
    sc.pp.neighbors(adata,key_added='neighbors_intensity')
    sc.tl.leiden(adata,key_added='leiden_intensity',neighbors_key='neighbors_intensity',resolution=0.5)

    return adata.obs["leiden_intensity"]


# calculate feature clusters
adata.obs["image_features_cluster"] = cluster_features(adata.obsm["image_intensity_features"], like="summary")

# compare feature and gene clusters
plt.close('all')

sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.pl.spatial(adata, color=["image_features_cluster", "leiden_gene"])
plt.savefig(fig_dir + r'\intensity vs gene clustering.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#%% Create a new clustering scheme, this time on cell density within spots
#First, smooth image
#from stardist.models import StarDist2D
crop = img.crop_corner(1000, 1000, size=200)
#sq.im.process(crop, layer="hires", method="smooth", sigma=2)
# plot the result
# fig, axes = plt.subplots(1, 2)
# for layer, ax in zip(["hires", "hires_smooth"], axes):
#     crop.show(layer, ax=ax)
#     ax.set_title(layer)
#Perform segmentation. Don't get your hopes up, Visium H&E images have the resolution of a Nintendo 64
sq.im.segment(img=crop, layer="hires", method="watershed", thresh=None, geq=False)

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_watershed']))}")

fig, axes = plt.subplots(1, 2)
crop.show("hires", ax=axes[0])
_ = axes[0].set_title("H&E")
crop.show("segmented_watershed", cmap=matplotlib.colors.ListedColormap (np.random.rand ( 256,3)), interpolation="none", ax=axes[1])
_ = axes[1].set_title("watershedsegmentation")

#To do: swap out segmentation stragey for StarDist, then proceed with image feature generation and PCA/Leiden
#%%Prepare StarDist model
# StarDist2D.from_pretrained()
# #sq.im.process(img, layer="hires", method="smooth", sigma=2)
# def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None):
#     #axis_norm = (0,1)   # normalize channels independently
#     axis_norm = (0,1,2) # normalize channels jointly
#     # Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.
#     # this is the default normalizer noted in StarDist examples.
#     img = normalize(img, 1, 99.8, axis=axis_norm)
#     model = StarDist2D.from_pretrained('2D_versatile_he')
#     labels, _ = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh,scale=10) #Still need to figure out what scale should be
#     return labels
# StarDist2D.from_pretrained('2D_versatile_he')
# sq.im.segment(
#     img=crop,
#     layer="hires",
#     channel=None,
#     method=stardist_2D_versatile_he,
#     layer_added='segmented_stardist_default',
#     prob_thresh=0.3,
#     nms_thresh=None,
#     chunks=1000
# )
# fig, axes = plt.subplots(1, 2)
# crop.show("hires", ax=axes[0])
# _ = axes[0].set_title("H&E")
# crop.show("segmented_stardist_default", cmap=matplotlib.colors.ListedColormap (np.random.rand ( 256,3)), interpolation="none", ax=axes[1])
# _ = axes[1].set_title("stardist segmentation")
# '''
# CONCLUSION: HIRES IMAGE HAS TOO POOR RESOLUTION FOR STARDIST, OR REALLY ANY SEGMENTATION FOR THAT MATTER.
# May need to inquire about obtaining and loading of high-res images. But for now, proceed with watershed segmentation for the rest of the tutorial
# '''
#%% Spatial statistics and graph analysis
#Compute neighbourhood enrichment. This tells us clusters that share a common neighborhood structure across tissue
print('COMPUTING sq.gr.spatial_neighbors')
sq.gr.spatial_neighbors(adata)
'''
Compute pairwise enrichment score based on the spatial proximity of clusters: if spots belonging to two different clusters are often close to each other,
then they will have a high score and can be defined as being enriched.
On the other hand, if they are far apart, and therefore are seldom a neighborhood, the score will be low and they can be defined as depleted.
This score is based on a permutation-based test, and you can set the number of permutations with the n_perms argument (default is 1000)

Mark note: my understanding is that this is done independently from any gene expression data, and only on spatial proximity of data points
'''
print('COMPUTING sq.gr.nhood_enrichment')
sq.gr.nhood_enrichment(adata, cluster_key="leiden_gene",n_jobs=15,show_progress_bar=False)
#Plot the enrichment matrix
plt.close('all')
sq.pl.nhood_enrichment(adata, cluster_key="leiden_gene")
plt.savefig(fig_dir + r'\neighbourhood_enrichment.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#%%Co-occurence across spatial dimensions
#similar to above, but doesn't operate on a connectivity matrix, rather the original spatial coordinates.
#Formula is https://squidpy.readthedocs.io/en/latest/auto_tutorials/tutorial_visium_hne.html#co-occurrence-across-spatial-dimensions
#Basically, computes the probability of finding one specific cluster in increasing distances around spots of all other clusters
#For now, lets use cluster_to_heatmap as the cluster to compute the distance to (our hypoxic cluster in 50D)
print('COMPUTING sq.gr.co_occurrence')
sq.gr.co_occurrence(adata, cluster_key="leiden_gene", n_jobs=15,show_progress_bar=False)
plt.close('all')

sq.pl.co_occurrence(
    adata,
    cluster_key="leiden_gene",
    clusters=cluster_to_heatmap,
    figsize=(8, 4)
)
plt.savefig(fig_dir + r'\Co-occurence_across_spatial_dimensions.png',dpi=800,pad_inches=0.1,bbox_inches='tight')
#%% Ligand receptor interaction analysis
#Now that we've measured cluster co-occurrence, what are the molecular drivers of this?
#CellPhoneDB, developed by Efremova et al. plus Omnipath were used to create a massive database of ligand-receptor interactions.
#This is fairly complex, so you'd need to review Efremova et al., 2020: Cellphonedb: inferring cell–cell communication from combined expression of multi-subunit ligand–receptor complexes
#Compute ligrec
sq.gr.ligrec(
    adata,
    n_perms=100,
    cluster_key="leiden_gene",
    use_raw=False,
    n_jobs=15,
    show_progress_bar=False #Will give freeze error if not set
)
#%%Plot ligrec. I think we need to tune these parameters because I have no idea what the hell it's even showing. Just garbage, garbage everywhere!
plt.close('all')

sq.pl.ligrec(
    adata,
    cluster_key="leiden_gene",
    source_groups=cluster_to_heatmap,
    #target_groups=["Pyramidal_layer", "Pyramidal_layer_dentate_gyrus"],
    target_groups='11',
    means_range=(1.7, np.inf),
    alpha=1e-4,
    swap_axes=True,
    
)
plt.savefig(fig_dir + r'\Ligand-receptor_interaction_analysis.png',dpi=800,pad_inches=0.1,bbox_inches='tight')

#%% Identify spatially variable genes
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
#The results are found in the uns (unstructured) portion of adata (adata.uns["moranI"])
#Get the top 50 spatially variable genes:
spat_var_genes=adata.uns["moranI"].head(50)
spat_var_gene_names=spat_var_genes.index.values.tolist()[:7] #Get the top 9 for plotting
spat_var_gene_names.append('leiden_gene') #tag on the cluster as the last plot to visualize
plt.close('all')

sc.pl.spatial(adata, color=spat_var_gene_names)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig(fig_dir + r'\Spatially Variable genes.png',dpi=400,pad_inches=0.1,bbox_inches='tight')

'''
This concludes the tutorial workflow. It's slightly different from what's available on scanpy/squidpy for dealing with Visium H&E data.
The modifications made typically rely on optimizing parallel computing, eliminating freeze errors, adjusting some arbitrary thresholds,
and choosing what cluster method and individual cluster to plot
'''

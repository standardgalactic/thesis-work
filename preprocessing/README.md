## batch_normalization.py
- recscale multidimensional data on a per-batch basis
- percentile based normalization, percentiles defined from cumulative histograms of all images in dataset
- input is folder containing .ome.tiffs, output is folder containing normalized .ome.tiffs
	- to work with other image formats, you may need to change how the metadata is defined
## IMC_despeckler.py
- apply a despeckle filter to select IMC channels. Current despeckle filter will likely be a median filter with a 3x3 kernel size
- write out new IMC images in subfolder, preserving relevant metadata (channel name, resolution, etc.)
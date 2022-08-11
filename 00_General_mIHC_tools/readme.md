# General purpose multiplexed immunohistochemistry analysis tools
Refinement of some previous scripts to make them robust for future analysis requests
## percent_positive_computation.py
Accepts an annotation-level export of compositely-classified cells as detailed in https://github.com/MarkZaidi/I2K-Workshop-Materials
Supports
- Percent positive scoring and/or positive density scoring for single positive cells
- Percent positive scoring and/or positive density scoring for multiple positive cells
   - Multiple positive refers to cells categorized by multiple markers
   - Appending \_NEGATIVE to the dict keys allows for negative gating
- By default, it computes the scores on a per-image basis. If multiple annotations are present, it sums all cells across all annotations for that image. You can also group by other variables (e.g. a categorical patient identifier, if you have multiple images per patient)
- Filtering to remove images or annotations not of relevance to the computation
- Data visualization will be delegated towards separate scripts

Ultimately, I want this to be a tool for any mIHC analysis that minimizes human error. No copy/pasting, creation of different composite classifiers, or manually-done Excel math. **HUMANS ARE NOT INFALLABLE**. We make mistakes. Aside from a user defining a few constants, and a dict variable of cells defined by more than one markers, this requires no additional input for stats-ready data. Going forward, the output (raw_data.csv) will be used as the core input for various statistical analysis and data visualization scripts, adapted from other code.

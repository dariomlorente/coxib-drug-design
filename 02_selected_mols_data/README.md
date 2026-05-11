# SECOND PHASE: Selection of the Most Promising Molecules from a Generated Library

### 02_hit_prioritisation.png
![2nd Phase Flowdiagram](inputs/.visuals/02_hit_prioritisation.png)
Flowchart of the Jupyter Notebook `02_HIT_PRIORITISATION.ipynb`

### LGEMV_selected.png
![Effect of the 4/5 filter on the QED](inputs/.visuals/LGEMV_selected.png)
A graph illustrating the set intersections selected throughout `01_LIBRARY_GENERATION.ipynb` and in `02_HIT_PRIORITISATION.ipynb`

### QED_selected.png
![Effect of the 4/5 filter on the QED](inputs/.visuals/QED_selected.png)
A graph created using ggplot2 that showes the distribution of QED (Quantitative Estimate of Drug-likeness) scores following application of the filter from the `02_HIT_PRIORITISATION.ipynb` section. The sample consists of the 2878135 compounds registered in ChEMBL, of which 1714702 were accepted and 1163433 were rejected.
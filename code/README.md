To reproduce the experiments in the paper, proceed with the following steps:

1. Run ```processTasteProfile.ipynb```, which will generate a few *.csv files with train/test/validation/out-of-matrix prediction data. Playing with the conditions and thresholds in function ```remove_inactive()``` will generate data with different level of sparsity (corresponding toe DEN and SPR subsets in the paper).

2. Run ```numberizeData.ipynb``` to map ID's to numbered indices.

3. ```WMF_(deepContent_)tasteProfile_binary.ipynb``` will train the model for Hu et al. (plus deep content) and save the parameters. For content model, VQ features must be pre-generated (see [@dawenl/StochasticPMF](https://github.com/dawenl/stochastic_PMF) for reference). For the shallow model, ```WMF_deepContent_tasteProfile_binary.ipynb``` can still be used, just swap the feature that is used. 

4. Run ```evalution.py``` to compute the evaluation metrics.

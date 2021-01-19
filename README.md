# Partial Code for "Brain Function in the Pre-Adolescent Brain: Results from the ABCD Study."

A note on .py vs. .sh, all scripts are written in python, but submitted with corresponding SLURM .sh scripts.

Breakdown of different files:
- resave_merged.{py, sh}: Cortical task activation maps in fsaverage5 space are resaves such that their left and right hemisphere are concatenated.
- submit_all.py: Submits SLURM job scripts for all tasks and contrasts of interest for scripts generate, run_rely and run_corr_rely.
- generate.{py, sh}: Loads a single tasks data, then residualizes it. Next, a cohen's map is saved based on the residualized map, as well as the raw residualized data.
  Depending on the task, a performance correlation is then generated and saved.
- run_rely.{py, sh}: Estimates a measure of activation map (cohen's map) reliability for a contrast. Designed to be run in parralell. Based on the public package
  https://github.com/sahahn/Rely
- run_corr_rely.{py, sh}: Simmilar to run_rely, and also based on the same package, but computes reliability on performance correlations instead of between cohen's maps.
- resample.py : Requires Freesurfer installation. This script re-samples all saved cortical cohen's and performance correlation maps in fsaverage5 space to fsaverage space for plotting.
- plot_collage.{py, sh}: Generate and plot the final consortium paper figures.
- plot_funcs.py: Contains the majority of plotting and loading code needed to create the final collages.
- supplemental_analysis.py: This script contains the code to generate a number of supplemental analysis, including making figures for how much different co-variates alone explain variance in different contrasts, if there are any suitable performance variables for the MID task, and more.
- info.py : Contains a number of key utility functions for the above scripts as well as path definitions and other key information like the order of how task contrasts were originally stacked.
- helpers.py: Contains some general helper functions used across different scripts.
- resamp.sh : Fixed bash script for resampling from fsaverage5 to fsaverage, not to be used alone, but by via a function in helpers.py.



A note on the csv's with co-variates referenced within these scripts. Due to a combination of factors, we reference pre-generated csv's with the covairates and subsetting 'valid' subjects for each task. This process of generating "valid" subjects included a number of different steps, e.g., missing datat, qc on imaging data, problems with data from some scanners, etc... and likewise different steps were conducted by multiple analysist's (which makes collecting all of the code used here difficult). These csv's include additionally the performance variables if any for that task. Note also that any categorical variables within these csv's have been already dummy coded, but NOT yet de-meaned. De-meaning for all variables is taken care of in different steps of the provided scripts here. Lastly, we apologize for not being able to make all of the steps from raw data to referenced csv's with co-variates avaliable at this time.
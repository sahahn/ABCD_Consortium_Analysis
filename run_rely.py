import sys
from Rely import run_rely
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from info import (load_covars_df, proc_covars_func, get_mask, get_template_path, get_strat_series)

# Process arguments
task = str(sys.argv[1])
contrast = int(str(sys.argv[2]))
is_cortical = bool(int(str(sys.argv[3])))
run = str(sys.argv[4])

print('sys.argv=', sys.argv, flush=True)

# Load the covars df
covars_df = load_covars_df(task, return_perf=False)

# Get the strat series
strat_series = get_strat_series(covars_df)

# Generate the proper mask based off contrast + cortical vs. subcortical
mask = get_mask(task, contrast, is_cortical)

# Generate template path
template_path = get_template_path(task, is_cortical)

print('is_cortical=', is_cortical)
print('mask.shape=', mask.shape)
print('contrast=', contrast)
print('template_path=', template_path)

# Proc covars func de-means every column!
# Run rely
x_labels, corr_means, _, _, _ =\
    run_rely(covars_df, contrast, template_path,
             mask=mask, proc_covars_func=proc_covars_func,
             min_size=2, max_size=2500,
             every=10, n_repeats=50,
             thresh=None, n_jobs=8,
             stratify=strat_series,
             verbose=1)

# Save under rely output
output_dr = 'Rely_Output/'
save_loc = output_dr + task + '_' + str(contrast) + '_' + str(is_cortical) + '_' + run + '.npy'
np.save(save_loc, corr_means)
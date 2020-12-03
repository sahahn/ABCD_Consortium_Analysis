import sys
from Rely import load_resid_data
import numpy as np
import os

from info import (load_covars_df, proc_covars_func,
                  get_mask, get_template_path, contrasts)
from helpers import get_cohens, fast_corr


# Process arguments
task = str(sys.argv[1])
contrast = int(str(sys.argv[2]))
is_cortical = bool(int(str(sys.argv[3])))

# Load the covars df
covars_df, perf_df = load_covars_df(task, return_perf=True)

# De-MEAN!
covars_df = proc_covars_func(covars_df)

# Generate the proper mask based off contrast + cortical vs. subcortical
mask = get_mask(task, contrast, is_cortical)

# Generate template path
template_path = get_template_path(task, is_cortical)

# Load resid data
subjects, resid_data =\
    load_resid_data(covars_df, contrast, template_path, mask=mask,
                    n_jobs=8, verbose=1)
print('Final Shape:', resid_data.shape, flush=True)

# Save copy of each subject's resid data
os.makedirs('Resid_Data/', exist_ok=True)
for s, d in zip(subjects, resid_data):

    if is_cortical:
        c_dr = 'Resid_Data/cortical/'
    else:
        c_dr = 'Resid_Data/subcortical/'

    c_name = contrasts[task][contrast]
    save_dr = c_dr + task + '_' + c_name
    os.makedirs(save_dr, exist_ok=True)

    save_loc = os.path.join(save_dr, s)
    np.save(save_loc, d)

# Get cohens
resid_cohens = get_cohens(resid_data)
save_loc = 'Base_Output/' + task + '_' + str(contrast) + '_' + str(is_cortical) + '.npy'
np.save(save_loc, resid_cohens)

# Generate performance correlations
for col in perf_df:
    corr = fast_corr(resid_data, np.array(perf_df[col].loc[subjects]))
    save_loc = 'Corr_Output/' + task + '_' + str(contrast) + '_' + str(is_cortical) + '_' + col + '.npy'
    np.save(save_loc, corr)




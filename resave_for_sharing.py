from plot_funcs import get_data, get_corr_data
from info import contrasts, perf_labels, performance_vars
import os
import numpy as np
import nibabel as nib

scales = {'nBack': .98,
          'MID': .98,
          'SST': .99}

dr = 'Nifti_Maps/'
os.makedirs(dr, exist_ok=True)

for task in contrasts:
    
    task_dr = os.path.join(dr, task)
    os.makedirs(task_dr, exist_ok=True)

    cs = contrasts[task]

    # Save activation maps
    activation_dr = os.path.join(task_dr, 'Activations')
    os.makedirs(activation_dr, exist_ok=True)

    scale = scales[task]
    cortical_data, subcortical_data, _, _ = get_data(task=task,
                                                     cs=cs,
                                                     scale=scale)

    for cort, subcort, contrast in zip(cortical_data, subcortical_data, cs):

        # Cast to nifti & save cort
        cort_lh = nib.Nifti1Image(cort[0].reshape((163842, 1, 1)), affine=np.eye(4))
        cort_rh = nib.Nifti1Image(cort[1].reshape((163842, 1, 1)), affine=np.eye(4))
        nib.save(cort_lh, os.path.join(activation_dr, contrast + '_lh.nii.gz'))
        nib.save(cort_rh, os.path.join(activation_dr, contrast + '_rh.nii.gz'))

        # Save subcort
        nib.save(subcort, os.path.join(activation_dr, contrast + '_subcort.nii.gz'))

    # Save performance correlations
    perf_dr = os.path.join(task_dr, 'Performance_Corrs')
    os.makedirs(perf_dr, exist_ok=True)

    for p_corr, p_label in zip(performance_vars[task], perf_labels[task]):
        cortical_data, subcortical_data, _, _ = get_corr_data(task=task,
                                                              cs=cs,
                                                              corr_with=p_corr)

        for cort, subcort, contrast in zip(cortical_data, subcortical_data, cs):

            # Cast to nifti & save cort
            cort_lh = nib.Nifti1Image(cort[0].reshape((163842, 1, 1)), affine=np.eye(4))
            cort_rh = nib.Nifti1Image(cort[1].reshape((163842, 1, 1)), affine=np.eye(4))
            nib.save(cort_lh, os.path.join(perf_dr, contrast + '_' + p_label + '_lh.nii.gz'))
            nib.save(cort_rh, os.path.join(perf_dr, contrast + '_' + p_label + '_rh.nii.gz'))

            # Save subcort
            nib.save(subcort, os.path.join(perf_dr, contrast + '_' + p_label + '_subcort.nii.gz'))

        
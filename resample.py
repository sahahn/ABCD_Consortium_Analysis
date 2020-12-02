import os
import nibabel as nib
import numpy as np

os.makedirs('Base_Resampled', exist_ok=True)
os.makedirs('Corr_Resampled', exist_ok=True)

def resample_cortical(loc):

    # Load as np array
    as_np = np.load(loc)

    # Re-add medial wall as 0's
    medial_wall =\
        np.load('/users/s/a/sahahn/ABCD_Data/fsaverage5_medial_wall.npy').astype('bool')
    
    # Fill in
    to_fill = np.zeros(medial_wall.shape)
    to_fill[medial_wall] = as_np

    # Split by hemi and save as mgz
    lh = to_fill[:len(to_fill)//2]
    nib.save(nib.Nifti1Image(np.reshape(lh, (len(lh), 1, 1)), np.eye(4)), 'temp_lh.mgz')
    rh = to_fill[len(to_fill)//2:]
    nib.save(nib.Nifti1Image(np.reshape(rh, (len(rh), 1, 1)), np.eye(4)), 'temp_rh.mgz')

    # Resample
    os.system('bash resamp.sh')

    # Load and re-save, remove intermediate files
    lh = np.squeeze(nib.load('resamp_lh.mgz').get_fdata())
    rh = np.squeeze(nib.load('resamp_rh.mgz').get_fdata())
    os.remove('resamp_lh.mgz')
    os.remove('resamp_rh.mgz')

    np.save(loc.replace('_Output', '_Resampled'), np.array([lh, rh]))


for file in os.listdir('Base_Output'):
    if 'True' in file.split('_')[2]:
        loc = os.path.join('Base_Output', file)
        resample_cortical(loc)

for file in os.listdir('Corr_Output'):
    if 'True' in file.split('_')[2]:
        loc = os.path.join('Corr_Output', file)
        resample_cortical(loc)




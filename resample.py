import os
import nibabel as nib
import numpy as np
from helpers import sample_to_fsaverage

os.makedirs('Base_Resampled', exist_ok=True)
os.makedirs('Corr_Resampled', exist_ok=True)

def resample_cortical(loc):

    # Load as np array
    as_np = np.load(loc)

    # Sample to fsaverage from fsaverage5
    lh, rh = sample_to_fsaverage(as_np)

    np.save(loc.replace('_Output', '_Resampled'), np.array([lh, rh]))


for file in os.listdir('Base_Output'):
    if 'True' in file.split('_')[2]:
        loc = os.path.join('Base_Output', file)
        resample_cortical(loc)

for file in os.listdir('Corr_Output'):
    if 'True' in file.split('_')[2]:
        loc = os.path.join('Corr_Output', file)
        resample_cortical(loc)




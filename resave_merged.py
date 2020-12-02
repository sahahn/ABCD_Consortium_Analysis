import numpy as np
import nibabel as nib
import os
import random

dr = '/users/s/a/sahahn/ABCD_Data/All_Merged_Vertex_2.0.1/Stacked/'
files = os.listdir(dr)
stubs = list(set([f.replace('_lh.mgz', '').replace('_rh.mgz', '') for f in files]))
random.shuffle(stubs)

for stub in stubs:
    base = os.path.join(dr, stub)

    save_loc = '/users/s/a/sahahn/scratch/Temp_Cortical/' + stub + '.mgz'
    if not os.path.exists(save_loc):

        try:
            lh = nib.load(base + '_lh.mgz').get_fdata()
            rh = nib.load(base + '_rh.mgz').get_fdata()
            merged = np.concatenate([lh, rh])
            nib.save(nib.Nifti1Image(merged, np.eye(4)), save_loc)
        except:
            pass
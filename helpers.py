import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression
from info import subcort_loc
import os

def get_cohens(data):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    cohen = mean / std

    return cohen

def fast_corr(O, P):
    
    n = P.size
    DO = O - (np.sum(O, 0) / np.double(n))
    DP = P - (np.sum(P) / np.double(n))
    return np.dot(DP, DO) / np.sqrt(np.sum(DO ** 2, 0) * np.sum(DP ** 2))

def get_resid(covars, data):

    # Go from pandas df to numpy array
    covars = np.array(covars)

    # Make sure data is numpy array
    data = np.array(data)

    # Fit a linear regression with covars as predictors, each voxel as target variable
    model = LinearRegression().fit(covars, data)

    # The difference is the real value of the voxel, minus the predicted value
    dif = data - model.predict(covars)

    # Set the residualized data to be, the intercept of the model + the difference
    resid = model.intercept_ + dif

    return resid

def get_r2(covars, data):

    # Make sure numpy arrays
    covars = np.array(covars)
    data = np.array(data)
    
    r2s = []
    for i in range(np.shape(data)[1]):
        model = LinearRegression().fit(covars, data[:,i])
        score = model.score(covars, data[:,i])
        r2s.append(score)
        
    return np.array(r2s)

def to_subcort(data):
    
    submask_affine = nib.load(subcort_loc).affine
    submask = np.squeeze(nib.load(subcort_loc).get_fdata())
    
    new = np.copy(submask)
    new[submask == 1] = data
    
    return nib.Nifti1Image(new, submask_affine)

def sample_to_fsaverage(as_np):

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

    return lh, rh
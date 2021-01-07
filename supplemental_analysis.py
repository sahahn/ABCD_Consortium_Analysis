import os
import numpy as np
import pandas as pd
from Neuro_Plotting.Ref import SurfRef, VolRef
from Neuro_Plotting.Plot import Plot_Surf, Plot_Surf_Collage
import matplotlib.pyplot as plt
from nilearn.plotting import plot_glass_brain, plot_stat_map
from Rely import load_resid_data
import scipy.io

from info import (contrasts, load_covars_df,
                  proc_covars_func, get_mask, get_template_path)
from helpers import fast_corr, get_resid, to_subcort, sample_to_fsaverage, get_r2, get_cohens
import nibabel as nib

# Create Neuro Plotting surface reference
surf_ref = SurfRef('/users/s/a/sahahn/Neuro_Plotting/data/', space='fsaverage')

# Location of nda folder
nda_folder = '/users/s/a/sahahn/ABCD_Data/ABCD2p0NDA/'

# Dof's dr
dofs_by_subj_dr = '/users/s/a/sahahn/ABCD_Data/dofs_by_subject'


def q_stand(d):
    
    d -= np.mean(d)
    d /= np.std(d)
    
    return d

def na_check(val):
    
    if pd.isna(val):
        return True
    if val == 777 or val == '777' or val == '999' or val == 999:
        return True
    
    return False

def plot_corrs(task, corrs, to_corr):
    
    for contrast in contrasts[task]:
        
        dr = 'Extra_Figures/' + to_corr
        os.makedirs(dr, exist_ok=True)

        dr += '/' + contrast
        os.makedirs(dr, exist_ok=True)

        # Re-sample the cortical correlations to fsaverage to plot
        lh, rh = sample_to_fsaverage(corrs[contrast + '.cortical'])
        both = [lh, rh]
        
        # Plot cortical collage
        figure, _, _ = Plot_Surf_Collage(both,
                                         surf_ref,
                                         surf_mesh='inflated',
                                         cmap='seismic',
                                         vmax=None,
                                         vmin=None,
                                         threshold=None,
                                         wspace=0,
                                         hspace=0,
                                         title=contrast + ' Corr. w/ ' + to_corr, 
                                         title_sz=22,
                                         avg_method='mean',
                                         bg_on_data=True,
                                         alpha=1,
                                         colorbar=True)
        
        # Save
        figure.savefig(dr + '/cortical_corr_' + to_corr + '.png', dpi=200)
        figure.clear()
        plt.close(figure)

        # Conv subcortical to nifti for plotting
        subcort = to_subcort(corrs[contrast + '.subcortical'])
        
        # Plot
        figure = plt.figure()
        plot_glass_brain(subcort,
                         cmap='seismic', 
                         symmetric_cbar=True,
                         plot_abs=False,
                         colorbar=True,
                         title=contrast + ' Corr. w/ ' + to_corr,
                         figure=figure)
        
        # Save
        figure.savefig(dr + '/subcortical_glass_corr_' + to_corr + '.png', dpi=200)
        
        # Close
        plt.close()
        figure.clear()
        plt.close(figure)
        
        # Plot alternate
        figure = plt.figure()
        plot_stat_map(subcort,
                      cmap='seismic', 
                      symmetric_cbar=True,
                      colorbar=True,
                      title=contrast + ' Corr. w/ ' + to_corr,
                      figure=figure)
        
        # Save and close
        figure.savefig(dr + '/subcortical_slices_corr_' + to_corr + '.png', dpi=200)
        figure.clear()
        plt.close(figure)

def load_raw_data(task, resid=False, overlap_with=None):

    all_data = {}

    # Subjects will be same for all
    # overwrite w/ latest
    subjects = None

    # Get covars df + de-mean
    covars = load_covars_df(task, return_perf=False)
    covars = proc_covars_func(covars)

    # Overlap with one or more series
    if overlap_with is not None:

        if not isinstance(overlap_with, list):
            overlap_with = [overlap_with]

        for ow in overlap_with:
            to_keep = np.intersect1d(covars.index, ow.index)
            covars = covars.loc[to_keep]

    # For each contrast
    for contrast in contrasts[task]:
        for is_cortical in [True, False]:

            # Get mask
            mask = get_mask(task, contrast, is_cortical=is_cortical)

            # Generate template path
            tp = get_template_path(task, is_cortical=is_cortical)

            # Load raw data
            subjects, data =\
                load_resid_data(covars, contrast, tp, mask=mask,
                                resid=resid, n_jobs=16, verbose=1)
            
            # Add to all_data dict
            app_name = '.subcortical'
            if is_cortical:
                app_name = '.cortical'

            all_data[contrast + app_name] = data

    return all_data, covars.loc[subjects]

def get_col_corr(data, covars, to_corr):
    '''Residualize data for all but the variable to correlate.'''

    col = np.array(covars[to_corr])
    to_resid = np.array(covars.drop([to_corr], axis=1))
    
    corrs = {}
    
    for d in data:
        resid_data = get_resid(to_resid, data[d])
        corrs[d] = fast_corr(resid_data, col)
        
    return corrs

def run_correlations():

    for task in contrasts:
        
        # Load the raw data + proc'ed covars df
        data, covars = load_raw_data(task)

        # Generate the correlations for each of the following
        for to_corr in ['interview_age', 'averaged_puberty', 'sex', 'education']:
            corrs = get_col_corr(data, covars, to_corr)
            plot_corrs(task, corrs, to_corr)

        # Go through and generate the variance explained by site
        site_cols = [col for col in covars if 'mri_info_deviceserialnumber' in col]
        site = np.array(covars[site_cols])

        covars_no_site = covars.drop(site_cols, axis=1)
        
        for d in data:

            # Residualize for all but site
            resid_data = get_resid(covars_no_site, data[d])

            # Get the r2 explained by site
            r2 = get_r2(site, resid_data)
            
            # Save as mgz in Site_Var folder
            as_nii = nib.Nifti1Image(r2, np.eye(4))
            os.makedirs('Site_Var/', exist_ok=True)
            nib.save(as_nii, 'Site_Var/' + task + '.' + d + '.mgz')

        # Clear mem
        del data
        del covars

def test_mid_extra_vars():

    txt_loc = os.path.join(nda_folder, 'abcd_bisbas01.txt')

    bis = pd.read_csv(txt_loc, sep='\t', skiprows=[1], index_col='src_subject_id')
    bis.index = [b.replace('NDAR_', '') for b in bis.index]

    to_compute = {'bis_sum': ['1', '2', '3', '4', '5', '6', '7'],
                  'bas_reward_responsiveness': ['8', '9', '10', '11', '12'],
                  'bas_drive': ['13', '14', '15', '16'],
                  'bas_fun_seeking': ['17', '18', '19', '20'],
                  'bis_sum_modified': ['2', '3', '4', '6'],
                  'bas_reward_responsiveness_modified': ['8', '9', '11', '12']}

    for sbj in bis.index:
        for c in to_compute:
            to_sum = [bis.loc[sbj, 'bisbas' + i + '_y'] for i in to_compute[c]]
            
            if any([na_check(v) for v in to_sum]):
                bis.loc[sbj, c] = np.nan
            else:
                bis.loc[sbj, c] = np.sum(to_sum)
                
    bis_sum_modified = bis['bis_sum_modified'].dropna()
    bis_sum_modified = q_stand(bis_sum_modified)

    bas_reward_responsiveness = bis['bas_reward_responsiveness'].dropna()
    bas_reward_responsiveness = q_stand(bas_reward_responsiveness)

    bas_reward_responsiveness_modified = bis['bas_reward_responsiveness_modified'].dropna()
    bas_reward_responsiveness_modified = q_stand(bas_reward_responsiveness_modified)

    bas_drive = bis['bas_drive'].dropna()
    bas_drive = q_stand(bas_drive)

    bas_fun_seeking = bis['bas_fun_seeking'].dropna()
    bas_fun_seeking = q_stand(bas_fun_seeking)

    noise = bas_fun_seeking.copy()
    noise.name = 'noise'
    noise.loc[noise.index] = np.random.normal(size=noise.shape)

    for to_corr in [bis_sum_modified, bas_reward_responsiveness,
                    bas_reward_responsiveness_modified,
                    bas_drive, bas_fun_seeking, noise]:
        
        # Load the resid MID data as overlapping with to_corr
        resid_data, covars_df = load_raw_data('MID', resid=True, overlap_with=to_corr)
        
        # Set to matching index
        to_c = to_corr.loc[covars_df.index]

        # Generate the correlations
        corrs = {}
        for d in resid_data:
            corrs[d] = fast_corr(resid_data[d], to_c)
        
        # Plot
        plot_corrs('MID', corrs, to_corr.name)
        
        # Save memory
        del corrs, resid_data, covars_df

def load_dofs():

    def load_dof(loc):

        dof = scipy.io.loadmat(loc)['dof'][0][0]
        dof = float(dof)
        
        return dof

    subject_dofs = pd.DataFrame()
    subject_dofs.index.name = 'src_subject_id'

    subjects = os.listdir(dofs_by_subj_dr)
    for subject in subjects:
        
        sbj_id = subject.split('_')[2]
        sbj_dr = os.path.join(dofs_by_subj_dr, subject)
        
        files = os.listdir(sbj_dr)
        task_summed_dfs = {}
        
        for file in files:
            
            tsk = file.split('_')[1]
            dof = load_dof(sbj_dr + '/' + file)
            
            try:
                task_summed_dfs[tsk] += dof
            except KeyError:
                task_summed_dfs[tsk] = dof
                
        for tsk in task_summed_dfs:
            subject_dofs.loc[sbj_id, tsk] = task_summed_dfs[tsk]
            
    task_dofs = {}

    for task in contrasts:
        qc_subset = subject_dofs[task]
        qc_subset = qc_subset.dropna()
        task_dofs[task] = qc_subset

    return task_dofs

def load_handedness():

    hand_loc = os.path.join(nda_folder, 'abcd_ehis01.txt')
    handed = pd.read_csv(hand_loc, sep='\t',
                         skiprows=[1], index_col='src_subject_id')
    handed.index = [h.replace('NDAR_', '') for h in handed.index]

    handed['l_hand'] = 0
    handed['m_hand'] = 0

    handed['l_hand'].loc[handed['ehi_y_ss_scoreb']==2] = 1
    handed['m_hand'].loc[handed['ehi_y_ss_scoreb']==3] = 1

    # De-mean
    l_hand = handed['l_hand'] - handed['l_hand'].mean()
    m_hand = handed['m_hand'] - handed['m_hand'].mean()

    return l_hand, m_hand

def load_motion():
    
    mean_motions = {}

    qc_loc = os.path.join(nda_folder, 'mriqcrp202.txt')
    qc = pd.read_csv(qc_loc, sep='\t', skiprows = [1], index_col='src_subject_id',
                    usecols=['src_subject_id', 'iqc_mid_all_mean_motion', 'iqc_sst_all_mean_motion', 'iqc_nback_all_mean_motion'])
    qc.index = [h.replace('NDAR_', '') for h in qc.index]

    for task in contrasts:
        qc_subset = qc['iqc_' + task.lower() + '_all_mean_motion']
        qc_subset = qc_subset.dropna()
        mean_motions[task] = qc_subset

    return mean_motions

def make_histograms():

    os.makedirs('Histograms', exist_ok=True)

    task_dofs = load_dofs()
    for task in contrasts:

        subjs = load_covars_df(task, return_perf=False).index
        plt.hist(task_dofs[task].loc[subjs], bins=50)
        plt.title(task + ' DOF Histogram')
        plt.ylabel('Counts')
        plt.xlabel('Value')
        plt.savefig('Histograms/'+task+'_dof_histogram.png', dpi=750)
        plt.show()
        plt.clf()
        plt.close()

    mean_motions = load_motion()
    for task in contrasts:

        subjs = load_covars_df(task, return_perf=False).index
        plt.hist(mean_motions[task].loc[subjs], bins=50)
        plt.title(task + ' FD Histogram')
        plt.ylabel('Counts')
        plt.xlabel('Value')
        
        if task == 'nBack':
            plt.xlim(0, 3)
            
        plt.savefig('Histograms/'+task+'_fd_histogram.png', dpi=750)  
        plt.show()
        plt.clf()
        plt.close()

def get_resid_cohens(data, covars):
    
    to_resid = np.array(covars)
    cohens = {}

    for d in data:
        resid_data = get_resid(to_resid, data[d])
        cohens[d] = get_cohens(resid_data)
        
    return cohens

def run_combos():

    task_dofs = load_dofs()
    l_hand, m_hand =  load_handedness()
    mean_motions = load_motion()

    corr_with_dofs = {'nBack': [],
                      'SST': [],
                      'MID': []}
    corr_with_hands = {'nBack': [],
                        'SST': [],
                        'MID': []}
    corr_with_mots = {'nBack': [],
                        'SST': [],
                        'MID': []}
    corr_with_alls = {'nBack': [],
                        'SST': [],
                        'MID': []}

    names = {'nBack': [],
             'SST': [],
             'MID': []}

    for task in contrasts:
        print(task)
        data, covars = load_raw_data(task, overlap_with=[l_hand, m_hand,
                                                         task_dofs[task],
                                                         mean_motions[task]])
       
        # Get dof's but de-mean
        covars_dof = covars.copy()
        covars_dof['dof'] = task_dofs[task] - task_dofs[task].mean()

        covars_hand = covars.copy()
        covars_hand['l_hand'] = l_hand
        covars_hand['m_hand'] = m_hand

        # De-meaned motion
        covars_mot = covars.copy()
        covars_mot['mot'] = mean_motions[task] - mean_motions[task].mean()

        # All
        covars_all = covars.copy()
        covars_all['dof'] = task_dofs[task] - task_dofs[task].mean()
        covars_all['l_hand'] = l_hand
        covars_all['m_hand'] = m_hand
        covars_all['mot'] = mean_motions[task] - mean_motions[task].mean()

        # Get cohens
        cohens = get_resid_cohens(data, covars)
        cohens_dof = get_resid_cohens(data, covars_dof)
        cohens_hand = get_resid_cohens(data, covars_hand)
        cohens_mot = get_resid_cohens(data, covars_mot)
        cohens_all = get_resid_cohens(data, covars_all)

        # Compute corr
        for c in cohens:
            corr_with_dofs[task].append(np.corrcoef(cohens[c], cohens_dof[c])[0][1])
            corr_with_hands[task].append(np.corrcoef(cohens[c], cohens_hand[c])[0][1])
            corr_with_mots[task].append(np.corrcoef(cohens[c], cohens_mot[c])[0][1])
            corr_with_alls[task].append(np.corrcoef(cohens[c], cohens_all[c])[0][1])
            names[task].append(c)

        print('Corr with dofs:', np.mean(corr_with_dofs[task]))
        print('Corr with hand:', np.mean(corr_with_hands[task]))
        print('Corr with motion:', np.mean(corr_with_mots[task]))
        print('Corr with all:', np.mean(corr_with_alls[task]))
        
        del data

    for task in contrasts:
        
        print(task)
        
        print('Dofs')
        for n, c in zip(names[task], corr_with_dofs[task]):
            print(n,c)

        print('Hand')
        for n, c in zip(names[task], corr_with_hands[task]):
            print(n,c)

        print('Motion')
        for n, c in zip(names[task], corr_with_mots[task]):
            print(n,c)

        print('All')
        for n, c in zip(names[task], corr_with_alls[task]):
            print(n,c)
    

#run_correlations()
#test_mid_extra_vars()
#make_histograms()
run_combos()

import pandas as pd
import numpy as np

# Names of contrasts used
nback_contrasts1 = ['0-back vs. fixation', '2-back vs. fixation', '2-back_vs._0-back']
nback_contrasts2 = ['Face_vs._places','Negface_vs._neutface', 'Posface_vs._neutface']
nback_contrasts = nback_contrasts1 + nback_contrasts2
sst_contrasts = ['Correct_stop_vs._correct_go', 'Incorrect_stop_vs._correct_go',
                 'Correct_stop_vs._incorrect_stop']
mid_contrasts = ['Antic._large_reward_vs._neutral', 'Antic._large_loss_vs._neutral', 
                 'Reward_pos._vs._neg._feedback', 'Loss_pos._vs._neg._feedback',
                 'Antic._large_vs._small_reward', 'Antic._large_vs._small_loss']
contrasts = {'MID': mid_contrasts,
             'nBack': nback_contrasts,
             'SST': sst_contrasts}

# Performance labels for plotting
perf_labels = {'nBack': ['dprime_2back', 'dprime_0back'],
               'SST': ['SSRT'],
               'MID': []}

# Covars Locs
mid_covars_loc = '/users/s/a/sahahn/Process_ABCD_Data/Palm/Covar_Files/new_covars_MID.csv'
nback_covars_loc = '/users/s/a/sahahn/Process_ABCD_Data/Palm/Covar_Files/new_covars_nBack.csv'
sst_covars_loc = '/users/s/a/sahahn/Process_ABCD_Data/Palm/Covar_Files/new_covars_SST.csv'
covars_locs = {'MID': mid_covars_loc,
               'nBack': nback_covars_loc,
               'SST': sst_covars_loc}

# Name of performance vars in saved covars csv
nback_performance_vars = ['dprime_2back', 'dprime_0back']
sst_performance_vars = ['tfmri_sst_all_beh_total_meanrt']
performance_vars = {'MID': [],
                    'nBack': nback_performance_vars,
                    'SST': sst_performance_vars}

all_labels = {
    'MID':
    ['Large_reward_antic',
    'Large_reward_pos._feedback',
    'Large_reward_neg._feedback',
    'Small_reward_antic.',
    'Small_reward_pos._feedback',
    'Small_reward_neg._feedback',
    'Large_loss_antic.',
    'Large_loss_pos._feedback',
    'Large_loss_neg._feedback',
    'Small_loss_antic.',
    'Small_loss_pos._feedback',
    'Small_loss_neg._feedback',
    'Neutral_antic.',
    'Neutral_pos._feedback',
    'Neutral_neg._feedback',""
    'Antic_reward_vs._neutral',
    'Antic_loss_vs._neutral',
    'Reward_pos._vs._neg._feedback',
    'Loss_pos._vs._neg._feedback',
    'Antic._large_reward_vs._neutral',
    'Antic._small_reward_vs._neutral',
    'Antic._large_vs._small_reward',
    'Antic._large_loss_vs._neutral',
    'Antic._small_loss_vs._neutral',
    'Antic._large_vs._small_loss'],
    'SST' : [
    'Correct_go',
    'Incorrect_go',
    'Correctlate_go',
    'Noresp_go',
    'Incorrectlate_go',
    'Correct_stop',
    'Incorrect_stop',
    'Ssd_stop',
    'Correct_go_vs._fixation',
    'Correct_stop_vs._correct_go',
    'Incorrect_stop_vs._correct_go',
    'Any_stop_vs._correct_go',
    'Correct_stop_vs._incorrect_stop',
    'Incorrect_go_vs._correct_go',
    'Incorrect_go_vs._incorrect_stop'],
    'nBack' : [
    '2-back_posface',
    '2-back_neutface',
    '2-back_negface',
    '2-back_place',
    '0-back_posface',
    '0-back_neutface',
    '0-back_negface',
    '0-back_place',
    'Cue',
    '0-back vs. fixation',
    '2-back vs. fixation',
    'Place',
    'Emotion',
    '2-back_vs._0-back',
    'Face_vs._places',
    'Emotion_vs._neutface',
    'Negface_vs._neutface',
    'Posface_vs._neutface']}

# Subcort mask
subcort_loc = '/users/s/a/sahahn/ABCD_Data/sub_mask.nii'

# Subcort affine
affine = np.array([[-2.,  0.,    0.,   90.],
                   [0.,    2.,    0., -126.],
                   [0.,    0.,    2.,  -72.],
                   [0.,    0.,    0.,    1.]])

# Medial wall locs
lh_medial_loc = '/users/s/a/sahahn/ABCD_Data/fsaverage_lh_medial.npy'
rh_medial_loc = '/users/s/a/sahahn/ABCD_Data/fsaverage_rh_medial.npy'

# Plotting conts
NAME_SIZE = 20
TITLE_SIZE = 17
C_NAME_SIZE = 16
LABEL_SIZE = 12

def load_covars_df(task, return_perf=False, add_stratify=False):

    df = pd.read_csv(covars_locs[task])

    # Set index to subject id w/o NDAR preprend
    df['src_subject_id'] = [c.replace('NDAR_', '') for c in df['src_subject_id']]
    df = df.set_index('src_subject_id')

    # Extract perf variables, and drop
    perf_df = df[performance_vars[task]]
    df = df.drop(performance_vars[task], axis=1)

    # Add stratify if requested
    if add_stratify:
        strat_by = ['sex'] + [col for col in df if col.startswith('mri_info_deviceserial')]
        df['stratify'] = df[strat_by].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # If task is sst
    if task == 'SST':
        sst_extra = pd.read_csv('Release_3.0_SST_newcols.csv')
        sst_extra['src_subject_id'] = [c.replace('NDAR_', '') for c in sst_extra['src_subject_id']]
        sst_extra = sst_extra.set_index('src_subject_id')
        sst_extra = sst_extra.loc[sst_extra['eventname'] == 'baseline_year_1_arm_1']
        sst_extra = sst_extra.drop('eventname', axis=1)

        # Calculate subjects to drop
        to_drop = sst_extra.loc[(sst_extra['tfmri_sst_beh_glitchflag'] == 1) |
                                (sst_extra['tfmri_sst_beh_violatorflag'] == 1) |
                                (sst_extra['tfmri_sst_all_beh_total_issrt'] < 50) |
                                (sst_extra['tfmri_sst_beh_0SSDcount'] > 20)].index
        to_drop = np.intersect1d(to_drop, df.index)

        # Add subjects that are not in sst_extra to to_drop
        dif = np.setdiff1d(df.index, sst_extra.index)
        to_drop = np.union1d(to_drop, dif)
 
        # Drop from both df's
        df = df.drop(to_drop)
        perf_df = perf_df.drop(to_drop)

        # Re-save the performance df, with updated SSRTs
        perf_df['tfmri_sst_all_beh_total_meanrt'] = sst_extra['tfmri_sst_all_beh_total_issrt']

    if return_perf:
        return df, perf_df
    else:
        return df

def proc_covars_func(df):

    # Just de-mean all columns
    for col in df:
        df[col] = df[col] - df[col].mean()

    return df

def get_mask(task, contrast, is_cortical):

    if is_cortical:
        n = 20484
    else:
        n = 31870
        
    n_contrasts = len(all_labels[task])
    contrast_name = contrasts[task][contrast]
    contrast_ind = all_labels[task].index(contrast_name)

    mask = np.zeros((n, 1, 1, n_contrasts))

    # If cortical, use only the non-medial wall for that contrast
    if is_cortical:
        medial_wall =\
            np.reshape(np.load('/users/s/a/sahahn/ABCD_Data/fsaverage5_medial_wall.npy'),
                      (n, 1, 1))
        mask[:,:,:,contrast_ind] = medial_wall

    # If subcortical, use full, as already subcortically masked
    else:
        mask[:,:,:,contrast_ind] = 1

    return mask

def get_template_path(task, is_cortical):

    if is_cortical:
        return '/users/s/a/sahahn/scratch/Temp_Cortical/SUBJECT_' + task + '.mgz'
    else:
        return '/users/s/a/sahahn/ABCD_Data/All_Merged_Voxel_2.0.1/Stacked/SUBJECT_' + task + '.mgz'
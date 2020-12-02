source /gpfs1/arch/x86_64-rhel7/freesurfer-6.0.0/SetUpFreeSurfer.sh
mri_surf2surf --hemi lh --srcsubject fsaverage5 --srcsurfval temp_lh.mgz --trgsubject fsaverage --trgsurfval resamp_lh.mgz
mri_surf2surf --hemi rh --srcsubject fsaverage5 --srcsurfval temp_rh.mgz --trgsubject fsaverage --trgsurfval resamp_rh.mgz
rm temp_lh.mgz
rm temp_rh.mgz
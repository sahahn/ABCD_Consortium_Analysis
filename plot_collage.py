from plot_funcs import (get_data, get_corr_data, load_rely,  load_perf_rely,
                        fill_spot, add_collage_colorbar, make_plot, plot_surf)
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from info import NAME_SIZE, TITLE_SIZE, LABEL_SIZE

import numpy as np
import nilearn
import nilearn.datasets

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_nback1(plot=True):

    scale = .98
    task = 'nBack'

    figure = plt.figure(figsize=(20, 10))

    # Top grid
    grid = gridspec.GridSpec(nrows=2, ncols=6,
                            wspace=None, hspace=.1,
                            width_ratios=[3.5,10,10,10,1,10],
                            height_ratios=None)

    # Add side text
    text_ax = figure.add_subplot(grid[0, 0])
    text_ax.text(0, 1, 'N-Back', fontsize=NAME_SIZE, weight="bold")
    text_ax.text(0, .75, 'a)\nCortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.text(0, .12, 'b)\nSubcortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.set_axis_off()

    # Contrasts to plot
    contrasts = ['0-back vs. fixation', '2-back vs. fixation', '2-back_vs._0-back']
    cortical_data, subcortical_data, vmin, vmax =\
        get_data(task, contrasts, scale=scale)

    # Plot the base activations
    smfs, cnt = [], 1
    for i, contrast in enumerate(contrasts):
        
        # Replace name
        contrast = contrast.replace('_', ' ')

        smfs.append(fill_spot(figure,
                              contrast,
                              grid,
                              0, cnt,
                              cortical_data[i],
                              subcortical_data[i],
                              cmap='cold_hot',
                              vmin=vmin, vmax=vmax,
                              threshold=.2,
                              plot=plot))
        cnt += 1

    # Add colorbar axis
    colorbar_ax = figure.add_subplot(grid[0, 4])
    colorbar_ax.set_axis_off()
    
    add_collage_colorbar(figure=figure,
                         ax=colorbar_ax,
                         smfs=smfs,
                         vmax=vmax, vmin=vmin,
                         multicollage=True,
                         cbar_shrink=1,
                         cbar_aspect=30,
                         cbar_pad=0,
                         anchor='W',
                         cmap='cold_hot',
                         threshold=.2,
                         nb_ticks=5,
                         round_dec=2)

    # Add side text
    bot_text_ax = figure.add_subplot(grid[1, 0])
    bot_text_ax.text(0, .75, 'c)\nCortical\ncorrelation', fontsize=TITLE_SIZE)
    bot_text_ax.text(0, .12, 'd)\nSubcortical\ncorrelation', fontsize=TITLE_SIZE)
    bot_text_ax.set_axis_off()

    # Load the performance correlation data

    # First with just one 0 back
    d0_cortical_data, d0_subcortical_data, d0_vmin, d0_vmax =\
        get_corr_data(task, [contrasts[0]], 'dprime_0back')

    # Last 2 with 2back
    d2_cortical_data, d2_subcortical_data, d2_vmin, d2_vmax =\
        get_corr_data(task, contrasts[1:], 'dprime_2back')

    # Get performance vmin / vmax
    vmin = np.min([d0_vmin, d2_vmin])
    vmax = np.max([d0_vmax, d2_vmax])
    if np.abs(vmin) > vmax:
        vmax = np.abs(vmin)
    else:
        vmin = -vmax

    # Plot the performance contrasts
    smfs = []

    # 0back w/ 0-back d'
    smfs.append(fill_spot(figure,
                          '',
                          grid,
                          1, 1,
                          d0_cortical_data[0],
                          d0_subcortical_data[0],
                          cmap='seismic',
                          vmin=vmin, vmax=vmax,
                          threshold=.000001,
                          label = "      Correlation with 0-back D'",
                          plot = plot))

    # 2'back with 2-back d' and 2back - 0 back
    for i in range(2):
        smfs.append(fill_spot(figure,
                              '',
                              grid,
                              1, 2+i,
                              d2_cortical_data[i],
                              d2_subcortical_data[i],
                              cmap='seismic',
                              vmin=vmin, vmax=vmax,
                              threshold=.000001,
                              label = "      Correlation with 2-back D'",
                              plot = plot))

    # Add the performance contrast color bar
    bot_colorbar_ax = figure.add_subplot(grid[1,4])
    bot_colorbar_ax.set_axis_off()

    add_collage_colorbar(figure=figure,
                        ax=bot_colorbar_ax,
                        smfs=smfs,
                        vmax=vmax, vmin=vmin,
                        multicollage=True,
                        cbar_shrink=1,
                        cbar_aspect=30,
                        cbar_pad=0,
                        anchor='W',
                        cmap='seismic',
                        threshold=None,
                        nb_ticks=5,
                        round_dec=2)

    # Add Reliability Plots
    rely_grid =\
        gridspec.GridSpecFromSubplotSpec(nrows=5, ncols=2,
                                         subplot_spec = grid[:,5],
                                         hspace=0,
                                         height_ratios=[.05, 1, .5, 1, .13],
                                         width_ratios=[0.175, 1])
    
    # Load rely
    cort_rely, subcort_rely = load_rely(task, contrasts)

    # Load perf rely
    cort_p_rely1, subcort_p_rely1 =\
        load_perf_rely(task, [contrasts[0]], 'dprime_0back')
    cort_p_rely2, subcort_p_rely2 =\
        load_perf_rely(task, contrasts[1:], 'dprime_2back')
    
    cort_p_rely = cort_p_rely1 + cort_p_rely2
    subcort_p_rely = subcort_p_rely1 + subcort_p_rely2
    
    # Plot base rely + legend
    rely1_ax = figure.add_subplot(rely_grid[1,1])
    make_plot(rely1_ax,
              'e) Reproducibility',
              list(zip(cort_rely, subcort_rely)),
              contrasts,
              sz=6, legend_alpha=.5,
              legend_loc=(-.5,-.43),
              xlabel=None, x_lim=(-10, 2500),
              cspaces=5, sspaces=2,
              yticks=[0, .1, .2, .3, .4, .5, .6, .7, 0.8, 0.9, 1.0],
              ylabel = 'Correlation for activation maps')

    # Plot performance rely
    rely2_ax = figure.add_subplot(rely_grid[3,1])
    make_plot(rely2_ax, '',
              list(zip(cort_p_rely, subcort_p_rely)),
              contrasts,
              sz=6,
              ylabel='Correlation for performance maps',
              low_val= 0,
              yticks=[0, .1, .2, .3, .4, .5, .6, .7, 0.8, 0.9, 1.0],
              x_lim=(-10, 2500),
              legend=False)

    #figure.savefig('Figures/nback1.png', bbox_inches='tight', dpi=500)
    figure.savefig('Figures/nback1.pdf', bbox_inches='tight', dpi=500, transparent=True)
    plt.close(figure)

def plot_sst(plot=True):

    scale = .99
    task = 'SST'

    figure = plt.figure(figsize=(20, 10))

    # Top grid
    grid = gridspec.GridSpec(nrows=2, ncols=6,
                            wspace=None, hspace=.1,
                            width_ratios=[3.5,10,10,10,1,10],
                            height_ratios=None)

    text_ax = figure.add_subplot(grid[0, 0])
    text_ax.text(0, 1, 'SST', fontsize=NAME_SIZE, weight="bold")
    text_ax.text(0, .75, 'a)\nCortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.text(0, .12, 'b)\nSubcortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.set_axis_off()

    bot_text_ax = figure.add_subplot(grid[1, 0])
    bot_text_ax.text(0, .75, 'c)\nCortical\ncorrelation', fontsize=TITLE_SIZE)
    bot_text_ax.text(0, .12, 'd)\nSubcortical\ncorrelation', fontsize=TITLE_SIZE)
    bot_text_ax.set_axis_off()

    # Plot these contrasts
    contrasts = ['Correct_stop_vs._correct_go', 'Incorrect_stop_vs._correct_go',
                 'Correct_stop_vs._incorrect_stop']

    # Load the base activation maps
    cortical_data, subcortical_data, vmin, vmax =\
        get_data(task, contrasts, scale=scale)

    # Plot the base activations
    smfs, cnt = [], 1
    for i, contrast in enumerate(contrasts):
        
        # Replace name
        contrast = contrast.replace('_', ' ')

        smfs.append(fill_spot(figure,
                              contrast,
                              grid,
                              0, cnt,
                              cortical_data[i],
                              subcortical_data[i],
                              cmap='cold_hot',
                              vmin=vmin, vmax=vmax,
                              threshold=.2,
                              plot=plot))
        cnt += 1
    
    # Add colorbar
    colorbar_ax = figure.add_subplot(grid[0, 4])
    colorbar_ax.set_axis_off()

    add_collage_colorbar(figure=figure,
                         ax=colorbar_ax,
                         smfs=smfs,
                         vmax=vmax, vmin=vmin,
                         multicollage=True,
                         cbar_shrink=1,
                         cbar_aspect=30,
                         cbar_pad=0,
                         cmap='cold_hot',
                         threshold=.2)

    # Load corr data
    cortical_perf, subcortical_perf, vmin, vmax =\
        get_corr_data(task, contrasts, 'tfmri_sst_all_beh_total_meanrt')

    # Plot performance corrs
    smfs, cnt = [], 1
    for i, contrast in enumerate(contrasts):
    
        # Clean name
        contrast = contrast.replace('_', ' ')
    
        smfs.append(fill_spot(figure,
                              '',
                              grid,
                              1, cnt,
                              cortical_perf[i],
                              subcortical_perf[i],
                              cmap='seismic',
                              vmin=vmin, vmax=vmax,
                              threshold=.000001,
                              label = "         Correlation with SSRT",
                              plot=plot))
        cnt += 1


    # Add the performance contrast color bar
    bot_colorbar_ax = figure.add_subplot(grid[1, 4])
    bot_colorbar_ax.set_axis_off()

    add_collage_colorbar(figure=figure,
                         ax=bot_colorbar_ax,
                         smfs=smfs,
                         vmax=vmax, vmin=vmin,
                         multicollage=True,
                         cbar_shrink=1,
                         cbar_aspect=30,
                         cbar_pad=0,
                         cmap='seismic',
                         threshold=None)

    # Plot reliability
    rely_grid =\
        gridspec.GridSpecFromSubplotSpec(nrows=5, ncols=2,
                                         subplot_spec = grid[:,5],
                                         hspace=0,
                                         height_ratios=[.05, 1, .5, 1, .13],
                                         width_ratios=[0.175, 1])

    # Load rely + perf rely
    cort_rely, subcort_rely = load_rely(task, contrasts)
    cort_p_rely, subcort_p_rely =\
        load_perf_rely(task, contrasts, 'tfmri_sst_all_beh_total_meanrt')

    # Plot base rely + legend
    rely1_ax = figure.add_subplot(rely_grid[1, 1])
    make_plot(rely1_ax,
              'e) Reproducibility',
              list(zip(cort_rely, subcort_rely)),
              contrasts,
              sz=7, legend_alpha=.5,
              legend_loc=(-.66,-.4),
              xlabel=None, x_lim=(-10, 2500),
              cspaces=10, sspaces=8,
              special=True,
              yticks=[0, .1, .2, .3, .4, .5, .6, .7, 0.8, 0.9, 1.0],
              ylabel = 'Correlation for activation maps')

    # Plot performance rely
    rely2_ax = figure.add_subplot(rely_grid[3,1])
    make_plot(rely2_ax, '',
              list(zip(cort_p_rely, subcort_p_rely)),
              contrasts,
              sz=6,
              ylabel='Correlation for performance maps',
              low_val= 0,
              yticks=[0, .1, .2, .3, .4, .5, .6, .7, 0.8, 0.9, 1.0],
              x_lim=(-10, 2500),
              legend=False)

    # Save 
    #figure.savefig('Figures/sst.png', bbox_inches='tight', dpi=500)
    figure.savefig('Figures/sst.pdf', bbox_inches='tight', dpi=500, transparent=True)
    plt.close(figure)

def plot_nback2(plot=True):

    scale = .98
    task = 'nBack'

    figure = plt.figure(figsize=(20, 10))

    outer_grid = gridspec.GridSpec(nrows=2, ncols=8,
                                   wspace=None, hspace=None,
                                   width_ratios=[.75, 1, .1, .3, 1, 1, .1, .1],
                                   height_ratios=None)

    text_ax = figure.add_subplot(outer_grid[0, 0])
    text_ax.text(0, 1, 'N-Back', fontsize=NAME_SIZE, weight="bold")
    text_ax.text(0, .75, 'a)\nCortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.text(0, .12, 'b)\nSubcortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.text(0, -.5, 'Cortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.set_axis_off()

    # The contrasts to plot
    contrasts = ['Face_vs._places', 'Negface_vs._neutface', 'Posface_vs._neutface']

    # Loads faces vs. places seperate
    cortical_data1, subcortical_data1, vmin1, vmax1 =\
        get_data(task, [contrasts[0]], scale=scale)
    cortical_data2, subcortical_data2, vmin2, vmax2 =\
        get_data(task, contrasts[1:], scale=scale)

    # Plot faces vs places
    smfs = []
    smfs.append(fill_spot(figure,
                          contrasts[0].replace('_', ' '),
                          outer_grid,
                          0, 1,
                          cortical_data1[0],
                          subcortical_data1[0],
                          cmap='cold_hot',
                          vmin=vmin1, vmax=vmax1,
                          threshold=.2,
                          plot=plot))

    # Add colorbar
    colorbar_ax = figure.add_subplot(outer_grid[0, 2])
    colorbar_ax.set_axis_off()
        
    add_collage_colorbar(figure=figure,
                         ax=colorbar_ax, smfs=smfs,
                         vmax=vmax1, vmin=vmin1,
                         multicollage=True,
                         cbar_shrink=1,
                         cbar_aspect=30,
                         cbar_pad=0,
                         cmap='cold_hot',
                         threshold=.2)

    # Plot rest of contrasts
    smfs = []
    cnt = 4
    for i, contrast in enumerate(contrasts[1:]):
        contrast = contrast.replace('_', ' ')
        
        smfs.append(fill_spot(figure,
                              contrast,
                              outer_grid,
                              0, cnt,
                              cortical_data2[i],
                              subcortical_data2[i],
                              cmap='seismic',
                              vmin=vmin2, vmax=vmax2,
                              threshold=.00000001,
                              plot=plot))
        cnt += 1

    # Add seperate color bar
    colorbar_ax2 = figure.add_subplot(outer_grid[0, 6])
    colorbar_ax2.set_axis_off()
        
    add_collage_colorbar(figure=figure, ax=colorbar_ax2, smfs=smfs,
                         vmax=vmax2, vmin=vmin2,
                         multicollage=True,
                         cbar_shrink=1,
                         cbar_aspect=30,
                         cbar_pad=0,
                         cmap='seismic',
                         threshold=.00000001)

    # Add reliability plots
    bot_outer_grid = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=4,
                                                      subplot_spec = outer_grid[1,:],
                                                      width_ratios=[0.0675, .95, 1.05, .05])

    # Load rely
    cort_rely, subcort_rely = load_rely(task, contrasts)

    rely1_ax = figure.add_subplot(bot_outer_grid[2])
    make_plot(rely1_ax, 'c) Reproducibility',
              list(zip(cort_rely, subcort_rely)),
              contrasts,
              sz=6, legend_alpha=.5, low_val=-.1,
              yticks = [0, .2, .4,  .6,  .8,  1.0],
              x_lim=(-10, 2500), cspaces=3, sspaces=1,
              ylabel='Correlation for activation maps')


    # Add extra cortical view
    sml_grid = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2,
                                                subplot_spec = bot_outer_grid[1],
                                                wspace=-.5)

    fvp =  cortical_data1[0]
    fs_avg = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

    # Lh view
    lh_ax = figure.add_subplot(sml_grid[0], projection = '3d')
    plot_surf(fs_avg.infl_left,
              fvp[0],
              threshold = .2,
              view = 'posterior',
              cmap='cold_hot',
              vmin=vmin1, vmax=vmax1,
              bg_map = fs_avg.sulc_left,
              dist=6,
              alpha=1,
              bg_on_data=True,
              axes=lh_ax)

    # Rh view
    rh_ax = figure.add_subplot(sml_grid[1], projection = '3d')
    plot_surf(fs_avg.infl_right,
              fvp[1],
              threshold = .2,
              view = 'posterior',
              cmap='cold_hot',
              vmin=vmin1, vmax=vmax1,
              bg_map = fs_avg.sulc_right,
              dist=6,
              alpha=1,
              bg_on_data=True,
              axes=rh_ax)
    
    # Set alpha to 0
    lh_ax.patch.set_alpha(0)
    rh_ax.patch.set_alpha(0)
    
    # Add text
    t_ax = figure.add_subplot(sml_grid[0])
    t_ax.text(.585, -.12, "Cohen's d map", fontsize=LABEL_SIZE)
    t_ax.set_axis_off()

    # Save
    #figure.savefig('Figures/nback2.png', dpi=500, bbox_inches='tight')
    figure.savefig('Figures/nback2.pdf', bbox_inches='tight', dpi=500, transparent=True)
    plt.close(figure)

def plot_mid(plot=True):

    task = 'MID'
    plot = True
    scale = .98

    figure = plt.figure(figsize=(20, 10))

    outer_grid =\
        gridspec.GridSpec(nrows=2, ncols=6,
                          wspace=None, hspace=None,
                          width_ratios=[.35, 1, 1, 1, 1, .1],
                          height_ratios=None)

    text_ax = figure.add_subplot(outer_grid[0, 0])
    text_ax.text(0, 1, 'MID', fontsize=NAME_SIZE, weight="bold")
    text_ax.text(0, .75, 'a)\nCortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.text(0, .12, 'b)\nSubcortical\nactivation', fontsize=TITLE_SIZE)
    text_ax.set_axis_off()

    text_ax2 = figure.add_subplot(outer_grid[1, 0])
    text_ax2.text(0, .75, '\nCortical\nactivation', fontsize=TITLE_SIZE)
    text_ax2.text(0, .12, '\nSubcortical\nactivation', fontsize=TITLE_SIZE)
    text_ax2.set_axis_off()

    # Contrasts to plot
    contrasts = ['Antic._large_reward_vs._neutral', 'Antic._large_loss_vs._neutral', 
                 'Reward_pos._vs._neg._feedback', 'Loss_pos._vs._neg._feedback',
                 'Antic._large_vs._small_reward', 'Antic._large_vs._small_loss']
    
    # Load data
    cortical_data, subcortical_data, vmin, vmax =\
        get_data(task, contrasts, scale=scale)

    # Plot data
    smfs, cnt, cnt2 = [], 0, 0
    for i, contrast in enumerate(contrasts):
        
        # For plotting in columns
        if cnt == 4:
            cnt2 += 1
            cnt = 0
        
        # Clean contrast name
        contrast = contrast.replace('_', ' ')
        
        # Plot
        smfs.append(fill_spot(figure,
                              contrast,
                              outer_grid,
                              cnt2, cnt+1,
                              cortical_data[i],
                              subcortical_data[i],
                              cmap='cold_hot',
                              vmin=vmin, vmax=vmax,
                              threshold=.2,
                              plot=plot))
        cnt += 1
        
    # Add color bar
    colorbar_ax = figure.add_subplot(outer_grid[0, 5])
    colorbar_ax.set_axis_off()
        
    add_collage_colorbar(figure=figure, ax=colorbar_ax, smfs=smfs,
                         vmax=vmax, vmin=vmin,
                         multicollage=True,
                         cbar_shrink=1,
                         cbar_aspect=30,
                         cbar_pad=0,
                         cmap='cold_hot',
                         threshold=.2,
                         nb_ticks=5,
                         round_dec=2)

    # Add reliability
    bot_outer_grid = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2,
                                                      subplot_spec = outer_grid[1:,3:],
                                                      width_ratios=[0.01, 2],
                                                      height_ratios=[1, .025])

    # Load reliability
    cort_rely, subcort_rely = load_rely(task, contrasts)
    
    # Add plot
    rely1_ax = figure.add_subplot(bot_outer_grid[0, 1])
    make_plot(rely1_ax, 'c) Reproducibility',
              list(zip(cort_rely, subcort_rely)),
              contrasts,
              sz=6, legend_alpha=.5, low_val=-.2,
              yticks = [0, .2, .4, .6, .8, 1.0],
              x_lim=(-10, 2500), cspaces=13, sspaces=11)

    #figure.savefig('Figures/mid.png', dpi=500, bbox_inches='tight')
    figure.savefig('Figures/mid.pdf', bbox_inches='tight', dpi=500, transparent=True)
    plt.close(figure)


plot_nback1(plot=True)
plot_nback2(plot=True)
plot_mid(plot=True)
plot_sst(plot=True)



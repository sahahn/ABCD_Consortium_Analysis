import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colorbar import make_axes
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap

import nilearn
import nilearn.datasets
from nilearn.surface import load_surf_mesh, load_surf_data
from nilearn._utils.compat import _basestring
from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges

import nibabel as nib
import numpy as np
import os

from info import subcort_loc, affine, contrasts, lh_medial_loc, rh_medial_loc
from info import NAME_SIZE, TITLE_SIZE, C_NAME_SIZE, LABEL_SIZE


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def _crop_colorbar(cbar, cbar_vmin, cbar_vmax):
    """
    crop a colorbar to show from cbar_vmin to cbar_vmax
    Used when symmetric_cbar=False is used.
    """
    if (cbar_vmin is None) and (cbar_vmax is None):
        return
    cbar_tick_locs = cbar.locator.locs
    if cbar_vmax is None:
        cbar_vmax = cbar_tick_locs.max()
    if cbar_vmin is None:
        cbar_vmin = cbar_tick_locs.min()
    new_tick_locs = np.linspace(cbar_vmin, cbar_vmax,
                                len(cbar_tick_locs))
    cbar.ax.set_ylim(cbar.norm(cbar_vmin), cbar.norm(cbar_vmax))
    outline = cbar.outline.get_xy()
    outline[:2, 1] += cbar.norm(cbar_vmin)
    outline[2:6, 1] -= (1. - cbar.norm(cbar_vmax))
    outline[6:, 1] += cbar.norm(cbar_vmin)
    cbar.outline.set_xy(outline)
    cbar.set_ticks(new_tick_locs, update_ticks=True)

def plot_surf(surf_mesh, surf_map=None, bg_map=None,
              hemi='left', view='lateral', cmap=None, colorbar=False,
              avg_method='mean', threshold=None, alpha='auto',
              bg_on_data=False, darkness=1, vmin=None, vmax=None,
              cbar_vmin=None, cbar_vmax=None,
              title=None, output_file=None, axes=None, figure=None,
              midpoint=None, dist=6, elev1=None, azim1=None, **kwargs):
    """ Plotting of surfaces with optional background and data
    .. versionadded:: 0.3
    Parameters
    ----------
    surf_mesh: str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces.
    surf_map: str or numpy.ndarray, optional.
        Data to be displayed on the surface mesh. Can be a file (valid formats
        are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each vertex of the surf_mesh.
    bg_map: Surface data object (to be defined), optional,
        Background image to be plotted on the mesh underneath the
        surf_data in greyscale, most likely a sulcal depth map for
        realistic shading.
    hemi : {'left', 'right'}, default is 'left'
        Hemisphere to display.
    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'},
        default is 'lateral'
        View of the surface that is rendered.
    cmap: matplotlib colormap, str or colormap object, default is None
        To use for plotting of the stat_map. Either a string
        which is a name of a matplotlib colormap, or a matplotlib
        colormap object. If None, matplotlib default will be chosen
    colorbar : bool, optional, default is False
        If True, a colorbar of surf_map is displayed.
    avg_method: {'mean', 'median'}, default is 'mean'
        How to average vertex values to derive the face value, mean results
        in smooth, median in sharp boundaries.
    threshold : a number or None, default is None.
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image, values
        below the threshold (in absolute value) are plotted as transparent.
    alpha: float, alpha level of the mesh (not surf_data), default 'auto'
        If 'auto' is chosen, alpha will default to .5 when no bg_map
        is passed and to 1 if a bg_map is passed.
    bg_on_data: bool, default is False
        If True, and a bg_map is specified, the surf_data data is multiplied
        by the background image, so that e.g. sulcal depth is visible beneath
        the surf_data.
        NOTE: that this non-uniformly changes the surf_data values according
        to e.g the sulcal depth.
    darkness: float, between 0 and 1, default is 1
        Specifying the darkness of the background image.
        1 indicates that the original values of the background are used.
        .5 indicates the background values are reduced by half before being
        applied.
    vmin, vmax: lower / upper bound to plot surf_data values
        If None , the values will be set to min/max of the data
    title : str, optional
        Figure title.
    output_file: str, or None, optional
        The name of an image file to export plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
    axes: instance of matplotlib axes, None, optional
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, a new axes is created.
    figure: instance of matplotlib figure, None, optional
        The figure instance to plot to. If None, a new figure is created.
    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.
    nilearn.plotting.plot_surf_roi : For plotting statistical maps on brain
        surfaces.
    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.
    """

    # load mesh and derive axes limits
    mesh = load_surf_mesh(surf_mesh)
    coords, faces = mesh[0], mesh[1]
    limits = [coords.min(), coords.max()]

    # set view
    if hemi == 'right':
        if view == 'lateral':
            elev, azim = 0, 0
        elif view == 'medial':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        elif view == 'anterior':
            elev, azim = 0, 90
        elif view == 'posterior':
            elev, azim = 0, 270
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal, ventral, anterior, or posterior')
    elif hemi == 'left':
        if view == 'medial':
            elev, azim = 0, 0
        elif view == 'lateral':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        elif view == 'anterior':
            elev, azim = 0, 90
        elif view == 'posterior':
            elev, azim = 0, 270
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal, ventral, anterior, or posterior')
    else:
        raise ValueError('hemi must be one of right or left')

    if elev1 is not None:
        elev = elev1
    if azim1 is not None:
        azim = azim1

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if no cmap is given, set to matplotlib default
    if cmap is None:
        cmap = plt.cm.get_cmap(plt.rcParamsDefault['image.cmap'])
    else:
        # if cmap is given as string, translate to matplotlib cmap
        if isinstance(cmap, _basestring):
            cmap = plt.cm.get_cmap(cmap)

    # initiate figure and 3d axes
    if axes is None:
        if figure is None:
            figure = plt.figure()
        axes = Axes3D(figure, rect=[0, 0, 1, 1],
                      xlim=limits, ylim=limits)
    else:
        if figure is None:
            figure = axes.get_figure()
        axes.set_xlim(*limits)
        axes.set_ylim(*limits)
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()

    # plot mesh without data
    p3dcollec = axes.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                  triangles=faces, linewidth=0.,
                                  antialiased=False,
                                  color='white')

    # reduce viewing distance to remove space around mesh
    axes.dist = dist

    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    face_colors = np.ones((faces.shape[0], 4))

    if bg_map is None:
        bg_data = np.ones(coords.shape[0]) * 0.5

    else:
        bg_data = load_surf_data(bg_map)
        if bg_data.shape[0] != coords.shape[0]:
            raise ValueError('The bg_map does not have the same number '
                             'of vertices as the mesh.')

    bg_faces = np.mean(bg_data[faces], axis=1)
    if bg_faces.min() != bg_faces.max():
        bg_faces = bg_faces - bg_faces.min()
        bg_faces = bg_faces / bg_faces.max()
    # control background darkness
    bg_faces *= darkness
    face_colors = plt.cm.gray_r(bg_faces)

    # modify alpha values of background
    face_colors[:, 3] = alpha * face_colors[:, 3]
    # should it be possible to modify alpha of surf data as well?

    if surf_map is not None:
        surf_map_data = load_surf_data(surf_map)
        if len(surf_map_data.shape) is not 1:
            raise ValueError('surf_map can only have one dimension but has'
                             '%i dimensions' % len(surf_map_data.shape))
        if surf_map_data.shape[0] != coords.shape[0]:
            raise ValueError('The surf_map does not have the same number '
                             'of vertices as the mesh.')

        # create face values from vertex values by selected avg methods
        if avg_method == 'mean':
            surf_map_faces = np.mean(surf_map_data[faces], axis=1)
        elif avg_method == 'median':
            surf_map_faces = np.median(surf_map_data[faces], axis=1)

        # if no vmin/vmax are passed figure them out from data
        if vmin is None:
            vmin = np.nanmin(surf_map_faces)
        if vmax is None:
            vmax = np.nanmax(surf_map_faces)

        # treshold if inidcated
        if threshold is None:
            kept_indices = np.arange(surf_map_faces.shape[0])
        else:
            kept_indices = np.where(np.abs(surf_map_faces) >= threshold)[0]

        if midpoint is None:
            surf_map_faces = surf_map_faces - vmin
            surf_map_faces = surf_map_faces / (vmax - vmin)
            norm_object = None

        else:
            norm_object = MidpointNormalize(midpoint=midpoint, vmin=vmin,
                                            vmax=vmax)
            surf_map_faces = norm_object.__call__(surf_map_faces).data

        # multiply data with background if indicated
        if bg_on_data:
            face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])\
                * face_colors[kept_indices]
        else:
            face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])

        if colorbar:
            our_cmap = get_cmap(cmap)

            if midpoint is None:
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = norm_object

            nb_ticks = 5
            ticks = np.linspace(vmin, vmax, nb_ticks)
            bounds = np.linspace(vmin, vmax, our_cmap.N)

            if threshold is not None:
                cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
                # set colors to grey for absolute values < threshold
                istart = int(norm(-threshold, clip=True) *
                             (our_cmap.N - 1))
                istop = int(norm(threshold, clip=True) *
                            (our_cmap.N - 1))
                for i in range(istart, istop):
                    cmaplist[i] = (0.5, 0.5, 0.5, 1.)
                our_cmap = LinearSegmentedColormap.from_list(
                    'Custom cmap', cmaplist, our_cmap.N)

            # we need to create a proxy mappable
            proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(surf_map_faces)
            cax, kw = make_axes(axes, location='right', fraction=.1,
                                shrink=.6, pad=.0)
            cbar = figure.colorbar(
                proxy_mappable, cax=cax, ticks=ticks,
                boundaries=bounds, spacing='proportional',
                format='%.2g', orientation='vertical')
            _crop_colorbar(cbar, cbar_vmin, cbar_vmax)

        p3dcollec.set_facecolors(face_colors)

    if title is not None:
        axes.set_title(title, position=(.5, .95))

    # save figure if output file is given
    if output_file is not None:
        figure.savefig(output_file)
        plt.close(figure)
    else:
        return figure, surf_map_faces

def trunc(num, decimals):

    base = (10 ** decimals)

    if num > 0:
        num = np.floor(num * base) / base
    else:
        num = -num
        num = np.floor(num * base) / base
        num = -num
    return num

def add_collage_colorbar(figure, ax, smfs, vmax, vmin, midpoint=None,
                         multicollage=False,
                         cbar_fraction=.25,
                         cbar_shrink=.25,
                         cbar_aspect=20,
                         cbar_pad=.1,
                         anchor='C',
                         nb_ticks=5,
                         round_dec=None,
                         **kwargs):

    if 'cmap' not in kwargs:
        cmap = None
    else:
        cmap = kwargs.pop('cmap')

    if cmap is None:
        cmap = get_cmap(plt.rcParamsDefault['image.cmap'])
    else:
        if isinstance(cmap, _basestring):
            cmap = get_cmap(cmap)

    if 'threshold' in kwargs:
        threshold = kwargs['threshold']
    else:
        threshold = None

    if 'cbar_vmin' in kwargs:
        cbar_vmin = kwargs['cbar_vmin']
    else:
        cbar_vmin = None

    if 'cbar_vmax' in kwargs:
        cbar_vmax = kwargs['cbar_vmax']
    else:
        cbar_vmax = None

    # Color bar
    our_cmap = get_cmap(cmap)

    if midpoint is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = MidpointNormalize(midpoint=midpoint, vmin=vmin,
                                 vmax=vmax)

    nb_ticks = nb_ticks
    ticks = np.linspace(vmin, vmax, nb_ticks)

    if round_dec is not None:
        ticks = np.array([trunc(t, round_dec) for t in ticks])

    bounds = np.linspace(vmin, vmax, our_cmap.N)

    if threshold is not None:
        cmaplist = [our_cmap(i) for i in range(our_cmap.N)]

        # set colors to grey for absolute values < threshold
        istart = int(norm(-threshold, clip=True) *
                     (our_cmap.N - 1))
        istop = int(norm(threshold, clip=True) *
                    (our_cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)
        our_cmap = LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, our_cmap.N)

    # we need to create a proxy mappable
    proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
    proxy_mappable.set_array(np.concatenate(smfs))

    if multicollage:

        cbar = plt.colorbar(
            proxy_mappable, ax=ax, ticks=ticks, spacing='proportional',
            format='%.2g', orientation='vertical', anchor=anchor,
            fraction=cbar_fraction,
            shrink=cbar_shrink,
            aspect=cbar_aspect,
            pad=cbar_pad)

    else:

        left = (ax[0][0].get_position().x0 + ax[0][0].get_position().x1) / 2
        right = (ax[0][1].get_position().x0 + ax[0][1].get_position().x1) / 2
        bot = ax[1][0].get_position().y1
        width = right-left

        # [left, bottom, width, height]
        cbaxes = figure.add_axes([left, bot - (.05 / 3), width, .05])

        cbar = plt.colorbar(
            proxy_mappable, cax=cbaxes, ticks=ticks, spacing='proportional',
            format='%.2g', orientation='horizontal', shrink=1, anchor=anchor)

    _crop_colorbar(cbar, cbar_vmin, cbar_vmax)

    return figure

def base_surf_plot(data, hemi, inflate, fs_avg, dist=6, **kwargs):

    if hemi == 'lh':

        hemi = 'left'
        if inflate:
            surf_mesh = fs_avg.infl_left
        else:
            surf_mesh = fs_avg.pial_left
        bg_map = fs_avg.sulc_left

    else:
        hemi = 'right'
        if inflate:
            surf_mesh = fs_avg.infl_right
        else:
            surf_mesh = fs_avg.pial_right
        bg_map = fs_avg.sulc_right

    figure, surf_map_faces = plot_surf(surf_mesh,
                                       surf_map=data,
                                       bg_map=bg_map,
                                       hemi=hemi,
                                       dist=dist,
                                       **kwargs)

    return figure, surf_map_faces

def get_min_max(cortical_data, subcortical_data):
    
    subcort_min = np.nanmin([s.get_fdata() for s in subcortical_data])
    subcort_max = np.nanmax([s.get_fdata() for s in subcortical_data])
    cort_min = np.nanmin([c for c in cortical_data])
    cort_max = np.nanmax([c for c in cortical_data])
    vmin = np.nanmin([subcort_min, cort_min])
    vmax = np.nanmax([subcort_max, cort_max])

    if np.abs(vmin) > vmax:
        vmax = np.abs(vmin)
    else:
        vmin = -vmax
        
    return vmin, vmax

def get_data(task, cs, scale=1):

    # Get cortical + subcortical
    cortical_data = [get_cort_data(task, c, scale=scale) for c in cs]
    subcortical_data = [get_subcort_data(task, c, scale=scale) for c in cs]

    # Determine vmin vmax
    vmin, vmax = get_min_max(cortical_data, subcortical_data)

    return cortical_data, subcortical_data, vmin, vmax

def get_cort_data(task, contrast, scale=1):

    # Determine saved ind
    ind = contrasts[task].index(contrast)

    # Load data
    data = load_cort('Base_Resampled/' + task + '_' + str(ind) + '_True.npy', scale=scale)

    return data

def get_subcort_data(task, contrast, scale=1):

    # Determine saved ind
    ind = contrasts[task].index(contrast)

    # Load data
    data = load_subcort('Base_Output/' + task + '_' + str(ind) + '_False.npy', scale=scale)

    return data

def get_corr_data(task, cs, corr_with):

     # Get cortical + subcortical
    cortical_data = [get_cort_corr_data(task, c, corr_with) for c in cs]
    subcortical_data = [get_subcort_corr_data(task, c, corr_with) for c in cs]

    # Determine vmin vmax
    vmin, vmax = get_min_max(cortical_data, subcortical_data)

    return cortical_data, subcortical_data, vmin, vmax

def get_cort_corr_data(task, contrast, corr_with):

    # Determine saved ind
    ind = contrasts[task].index(contrast)
    
    # Load data
    data = load_cort('Corr_Resampled/' + task + '_' + str(ind) + '_True_' + corr_with + '.npy')

    return data

def get_subcort_corr_data(task, contrast, corr_with):
    
    # Determine saved ind
    ind = contrasts[task].index(contrast)

    # Load data
    data = load_subcort('Corr_Output/' + task + '_' + str(ind) + '_False_' + corr_with + '.npy')

    return data

def load_subcort(loc, scale=1):

    new = nib.load(subcort_loc).get_fdata()
    data = np.load(loc)
    new[new == 1] = data
    new = new * scale

    return nib.Nifti1Image(new, affine)

def load_cort(loc, scale=1):

    # Load + apply scale
    data = np.load(loc) * scale

    # Apply medial walls
    lh_medial = np.load(lh_medial_loc)
    rh_medial = np.load(rh_medial_loc)
    
    lh = data[0]
    lh[lh_medial] = 0
    rh = data[1]
    rh[rh_medial] = 0

    # Return as list of lh, rh
    return [lh, rh]

def fill_spot(figure, title, outer_grid, i, j, cortical_data, subcortical_data, fs_avg=None,
              inflate=True, vmin=None, vmax=None, dist=5.75, hspace=0, brains_hspace=.035,
              brains_wspace=.02, plot=True, label="Cohen's d map", **kwargs):
    
    if fs_avg == None:
        fs_avg = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
    
    # Add title
    title_ax = figure.add_subplot(outer_grid[i, j])
    title_ax.set_title(title, fontsize=C_NAME_SIZE)
    title_ax.set_axis_off()
    
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1,
                                                  subplot_spec = outer_grid[i, j],
                                                  hspace=hspace,
                                                  height_ratios=[1.5, 1, .01])
    
    top_grid = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                subplot_spec = inner_grid[0],
                                                hspace=brains_hspace,
                                                wspace=brains_wspace)
    
    smfs = []
    hemis = [['lh', 'rh'], ['lh', 'rh']]
    views = ['lateral', 'medial']
    
    for row in range(2):
        for col in range(2):
            
            ax = figure.add_subplot(top_grid[row, col], projection='3d')
            
            if plot:
                figure, smf = base_surf_plot(cortical_data[col], hemis[row][col],
                                             inflate=inflate,
                                             fs_avg=fs_avg,
                                             figure=figure,
                                             axes=ax, view=views[row],
                                             vmin=vmin, vmax=vmax,
                                             dist=dist, alpha=1,
                                             bg_on_data=True,
                                             **kwargs)
                
                ax.patch.set_alpha(0)
                smfs.append(smf)
                
            else:
                smfs.append(cortical_data[col])
            
    
    lower_ax = figure.add_subplot(inner_grid[1])
    
    if plot:
    
        nilearn.plotting.plot_glass_brain(subcortical_data,
                                          symmetric_cbar=True,
                                          plot_abs=False,
                                          colorbar=False,
                                          vmin=vmin,
                                          vmax=vmax,
                                          figure=figure,
                                          axes=lower_ax,
                                          **kwargs)
    
    text_ax = figure.add_subplot(inner_grid[2])
    
    if label == "Cohen's d map":
        text_ax.text(.3, 0, label, fontsize=LABEL_SIZE)
    else:
        text_ax.text(.05, -1, label, fontsize=LABEL_SIZE)
    
    text_ax.set_axis_off()
    
    #return None
    return np.concatenate(smfs)

def get_color(i):

     if i == 0:
        return 'red'
     elif i == 1:
        return 'blue'
     elif i == 2:
        return 'green'
     elif i == 3:
        return 'orange'
     elif i == 4:
        return 'black'
     else:
        return 'purple'

def load_rely(task, cs):

    cort = [load_cort_rely(task, c) for c in cs]
    subcort = [load_subcort_rely(task, c) for c in cs]

    return cort, subcort

def load_perf_rely(task, cs, corr_with):

    cort = [load_cort_perf_rely(task, c, corr_with) for c in cs]
    subcort = [load_subcort_perf_rely(task, c, corr_with) for c in cs]

    return cort, subcort

def load_cort_rely(task, contrast):

    ind = contrasts[task].index(contrast)
    cort_files = [f for f in os.listdir('Rely_Output')
                  if f.startswith(task + '_' + str(ind) + '_True_')]
    cort = np.mean([np.load(os.path.join('Rely_Output', f)) for f in cort_files], axis=0)
    
    return cort

def load_cort_perf_rely(task, contrast, corr_with):

    ind = contrasts[task].index(contrast)
    cort_files = [f for f in os.listdir('Corr_Rely_Output')
                  if f.startswith(task + '_' + str(ind) + '_True_' + corr_with)]
    cort = np.mean([np.load(os.path.join('Corr_Rely_Output', f)) for f in cort_files], axis=0)
    
    return cort

def load_subcort_rely(task, contrast):

    ind = contrasts[task].index(contrast)
    subcort_files = [f for f in os.listdir('Rely_Output')
                     if f.startswith(task + '_' + str(ind) + '_False_')]
    subcort = np.mean([np.load(os.path.join('Rely_Output', f)) for f in subcort_files], axis=0)

    return subcort

def load_subcort_perf_rely(task, contrast, corr_with):

    ind = contrasts[task].index(contrast)
    subcort_files = [f for f in os.listdir('Corr_Rely_Output')
                     if f.startswith(task + '_' + str(ind) + '_False_' + corr_with)]
    subcort = np.mean([np.load(os.path.join('Corr_Rely_Output', f)) for f in subcort_files], axis=0)

    return subcort
   
def make_plot(ax, title, means, labels,
              sz=0, legend_loc='lower right',
              legend_alpha=.8, low_val=.2,
              legend=True, x_lim=None, xlabel=True, yticks=None,
              cspaces=0, sspaces=0, dotted=False,
              ylabel=True, stack=False, special=False):
    
    # Clean labels
    labels = [l.replace('_', ' ') for l in labels]
    labels = [l.replace('vs.', '-') for l in labels]
    
    # Linewidth
    lw = 2

    # This is the base_x that the relability was generated with
    # so e.g, [2, 12, 22, 32, 42,...]
    x = list(range(2, 2501, 10))

    if len(means[0]) == 2:
        for j in range(2):
            if j == 0:
                plt.plot(0, 0, color='white',
                         label= ''.join([' ' for i in range(cspaces)]) + '$\\bf{Cortical}$')
            else:
                s_cort_name = ''.join([' ' for i in range(sspaces)]) + '$\\bf{Subcortical}$'
                if stack:
                    s_cort_name = '\n' + s_cort_name
                plt.plot(0, 0, color='white', label=s_cort_name)
        
            for i in range(len(means)):
                color =  get_color(i)
                y = means[i][j]
                    
                if j == 0:
                    ax.plot(x, y, label=labels[i], lw=lw, alpha=.8, color=color)
                elif j == 1:
                    ax.plot(x, y, label=labels[i], lw=lw, alpha=.8, color=color, dashes=[2,2])
                
    else:

        # Do non split scales plotting
        for i in range(len(means)):
            color =  get_color(i)
            y = means[i]
            
            if dotted:
                ax.plot(x, y, label=labels[i], lw=lw, color=color)
            else:
                ax.plot(x, y, label=labels[i], lw=lw, color=color, dashes=[2, 2])
    
    # Legend font size + loc params, change loc to dif. ints to get different default 
    # legend spots
    if legend:
        
        # Set n cols if stack
        ncol = 2
        if stack:
            ncol = 1
        
        # Plot legend in passed loc
        if len(means[0]) == 2:

            if special:
                ax.legend(fontsize=C_NAME_SIZE-sz, loc=legend_loc, ncol=ncol,
                          framealpha=legend_alpha, columnspacing=0, numpoints=2,
                          handletextpad=0, borderpad=.05)

            else:
                ax.legend(fontsize=C_NAME_SIZE-sz, loc=legend_loc, ncol=ncol,
                          framealpha=legend_alpha)
        
        else:
            ax.legend(fontsize=C_NAME_SIZE-sz, loc=legend_loc, framealpha=legend_alpha)
    
    # Plot title
    ax.set_title(title, fontsize=TITLE_SIZE)
    
    # X and y label
    if xlabel:
        ax.set_xlabel('Random Group Size', fontsize=LABEL_SIZE)
        
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    elif ylabel:
        ax.set_ylabel('Correlation for activation maps', fontsize=LABEL_SIZE)
    
    # Force the y axis to be between first number and 1
    ax.set_ylim(low_val, 1.00)
    
    if yticks is not None:
        ax.set_yticks(yticks)
    
    if x_lim is not None:
        ax.set_xlim(x_lim)
        
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=.25)
    
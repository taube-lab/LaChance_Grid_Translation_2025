# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:42:35 2025

run to generate all data-based main figure panels

@author: plachanc
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem, pearsonr
from scipy.stats.mstats import pearsonr as mapearsonr
from scipy import ndimage
from skimage.transform import rotate
import seaborn as sns
import pickle
import copy
import warnings

sns.set_style("white")
rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size':12})
rc('xtick',**{'bottom':True})
rc('ytick',**{'left':True})
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


def plot_spikes(tracking_data, destination):
    
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
        
    pos_x = tracking_data['pos_x']
    pos_y = tracking_data['pos_y']
    spike_train = tracking_data['spikes']
    
    savedir = os.path.dirname(destination)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    spike_x = pos_x[spike_train>0]
    spike_y = pos_y[spike_train>0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(pos_x,pos_y,color='gray',alpha=0.6,zorder=0,clip_on=False)
    ax.scatter(spike_x,spike_y,c='red',zorder=1,clip_on=False)
    ax.axis('off')
    ax.axis('equal')
    
    fig.savefig(destination)
    
    plt.close()


def plot_heatmap(heatmap, destination, max_fr = None):
    
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    
    if max_fr is None:
        max_fr = np.ceil(np.nanmax(heatmap))
    
    im = ax.imshow(heatmap, vmin=0, vmax = max_fr,cmap='viridis') 
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([0,max_fr])
    cbar.set_ticklabels(['0 Hz','%i Hz' % max_fr])
    
    ax.axis('off')
    ax.axis('equal')
        
    fig.savefig(destination)
    
    plt.close()
    
    
def strip_plot(group_data, group_ids, cell_ids, ylims, ylabel, destination, shuffle_vals = None):
    
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    
    vals = list(sum(group_data, []))
    labels = list(sum(group_ids, []))
    
    data_dict = {'val':vals,'label':labels,'cell':cell_ids*len(group_data)}
    data_df = pd.DataFrame(data_dict)
    
    for cell in cell_ids:
        if np.sum(np.isnan(data_df.loc[data_df['cell']==cell,'val'])) > 0:
            data_df = data_df[~(data_df['cell']==cell)]
    
    if len(group_data) == 2:
        palette = ['violet','gray']
        aspect = 1.3
    elif len(group_data) == 3:
        palette = ['gray','violet','gray']
        aspect = 1.3
    elif len(group_data) == 5:
        palette = ['violet','gray','violet','gray','purple']
        aspect = 1.1
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.stripplot(x='label',y='val',palette = palette, data=data_df, jitter=True)
    ax.plot((-1,len(np.unique(data_df['label']))),(0,0),'k-',alpha=.8)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel('')
    
    mean_width = 0.5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        var = text.get_text()
        mean_val = data_df[data_df['label']==var].val.mean()
        ax.plot([tick-mean_width/2, tick+mean_width/2], [mean_val, mean_val],
                lw=4, color='k',zorder=10,alpha=0.8)

    if shuffle_vals is not None:
        for i in range(len(shuffle_vals)):
            ax.plot([i-mean_width/2, i+mean_width/2], [shuffle_vals[i], shuffle_vals[i]],
                    lw=4, color='red',zorder=10,alpha=0.7)
    
    ax.set_ylim(ylims)
    ax.set_xlim([-.5,len(group_data)-.5])
    ax.set_ylabel(ylabel)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(aspect*(x1-x0)/(y1-y0))
    plt.tight_layout()
    fig.savefig(destination,dpi=400)
    plt.close()
    
    
def spatial_crosscorr(heatmap1,heatmap2):
    
    x_gr = 36
    y_gr = 40
    
    #make a matrix of zeros 2x the length and width of the smoothed heatmap (in bins)
    corr_matrix = np.zeros((2*x_gr,2*y_gr))
    
    #for every possible overlap between the smoothed heatmap and its copy, 
    #correlate those overlapping bins and assign them to the corresponding index
    #in the corr_matrix
    for i in range(-len(heatmap1),len(heatmap1)):
        for j in range(-len(heatmap2[0]),len(heatmap2[0])):
            if i < 0:
                if j < 0:
                    array1 = heatmap1[(-i):(x_gr),(-j):(y_gr)]
                    array2 = heatmap2[0:(x_gr+i),0:(y_gr+j)]
                elif j >= 0:
                    array1 = heatmap1[(-i):(x_gr),0:(y_gr-j)]
                    array2 = heatmap2[0:(x_gr+i),(j):y_gr]
            elif i >= 0:
                if j < 0:
                    array1 = heatmap1[0:(x_gr-i),(-j):(y_gr)]
                    array2 = heatmap2[(i):x_gr,0:(y_gr+j)]
                elif j >= 0:
                    array1 = heatmap1[0:(x_gr-i),0:(y_gr-j)]
                    array2 = heatmap2[(i):x_gr,(j):y_gr]
            
            #this will give us annoying warnings for issues that don't matter --
            #we'll just ignore them
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                try:
                    #get the pearson r for the overlapping arrays
                    corr,p = pearsonr(np.ndarray.flatten(array1),np.ndarray.flatten(array2))
                    #assign the value to the appropriate spot in the autocorr matrix
                    corr_matrix[x_gr+i][y_gr+j] = corr
                except:
                    corr_matrix[x_gr+i][y_gr+j] = np.nan
                
                
    corr_matrix = np.rot90(corr_matrix,-1)
    
    return corr_matrix
    
    
def plot_crosscorr_translation(crosscorr, closest_cm, destination):
    
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(crosscorr,cmap='viridis',origin='lower')
    ax.plot([0,len(crosscorr[0])-1],[40,40],'k--')
    ax.plot([36,36],[0,len(crosscorr)],'k--')
    ax.plot([36,closest_cm[1]],[40,closest_cm[0]],'r-')
    plt.axis('off')
    fig.savefig(destination, dpi=400)
    plt.close()
            
    
def create_figures():
    
    data_dir = os.getcwd() + '/data'

    active_grid_values = pd.read_csv(data_dir + '/active_grid_values.csv')
    active_grid_heatmaps = pickle.load(open(data_dir + '/active_grid_heatmaps.pickle','rb'))
    active_grid_tracking = pickle.load(open(data_dir + '/active_grid_tracking.pickle','rb'))
    
    active_grid_values['unique_id'] = active_grid_values['animal'] + active_grid_values['day'] + active_grid_values['cell']
    
    all_floor1_heatmaps = []
    all_raised_heatmaps = []
    all_floor2_heatmaps = []
    for cell in active_grid_heatmaps.keys():
        all_floor1_heatmaps.append(active_grid_heatmaps[cell]['pre'])
        all_raised_heatmaps.append(active_grid_heatmaps[cell]['exp'])
        all_floor2_heatmaps.append(active_grid_heatmaps[cell]['post'])
    

    ''' computing shuffle distribution for ratemap correlations '''

    exp_bootstrapped_corr_means = []
    post_bootstrapped_corr_means = []
    
    for shuffle in range(1000):
        shuffle_exp = []
        shuffle_post = []
        
        shuffled_inds = np.arange(len(all_floor1_heatmaps))
        np.random.shuffle(shuffled_inds)
        
        for i in range(len(shuffled_inds)):
            pre = all_floor1_heatmaps[i]
            exp = all_raised_heatmaps[shuffled_inds[i]]
            shuffle_exp.append(pearsonr(pre.flatten(),exp.flatten())[0])

            post = all_floor2_heatmaps[shuffled_inds[i]]
            shuffle_post.append(pearsonr(pre.flatten(),post.flatten())[0])
            
        exp_bootstrapped_corr_means.append(np.mean(shuffle_exp))
        post_bootstrapped_corr_means.append(np.mean(shuffle_post))
        
        
    ''' computing ratemap correlations and spatial autocorrelations '''
    
    onetwo_corrs = []
    onethree_corrs = []

    crosscorrs = []
    cellnames = []
    trialnames = []
    animalnames = []
    
    pre_autocorrs = []
    exp_autocorrs = []
    post_autocorrs = []
    
    for cell in active_grid_heatmaps.keys():
        onetwo_corrs.append(pearsonr(active_grid_heatmaps[cell]['pre'].flatten(),active_grid_heatmaps[cell]['exp'].flatten())[0])
        onethree_corrs.append(pearsonr(active_grid_heatmaps[cell]['pre'].flatten(),active_grid_heatmaps[cell]['post'].flatten())[0])
        crosscorrs.append(spatial_crosscorr(active_grid_heatmaps[cell]['exp'].T,active_grid_heatmaps[cell]['pre'].T))
        
        cellnames.append(cell)
        trialnames.append(active_grid_values[active_grid_values['unique_id']==cell].iloc[0]['day'] + active_grid_values[active_grid_values['unique_id']==cell].iloc[0]['animal'])
        animalnames.append(active_grid_values[active_grid_values['unique_id']==cell].iloc[0]['animal'])

        pre_autocorrs.append(spatial_crosscorr(active_grid_heatmaps[cell]['pre'].T,active_grid_heatmaps[cell]['pre'].T))
        exp_autocorrs.append(spatial_crosscorr(active_grid_heatmaps[cell]['exp'].T,active_grid_heatmaps[cell]['exp'].T))
        post_autocorrs.append(spatial_crosscorr(active_grid_heatmaps[cell]['post'].T,active_grid_heatmaps[cell]['post'].T))

    ''' make figures '''
    fig_dir = os.getcwd() + '/figures'
    
    ''' fig 1 panel C (example cells) '''
    dest_dir = fig_dir + '/Fig1/panel_C/PL76_12-10_TT2_C1'
    cellname = 'PL762019-12-10TT2_SS_01.txt'
    max_fr = 11
    plot_heatmap(active_grid_heatmaps[cellname]['baseline'],dest_dir + '/baseline_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['post'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_spikes(active_grid_tracking[cellname]['baseline'],dest_dir + '/baseline_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['exp'],dest_dir + '/raised_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['post'],dest_dir + '/floor2_path_spikes.png')
    
    dest_dir = fig_dir + '/Fig1/panel_C/SSW103_3-16_TT4_C1'
    cellname = 'SSW1032016-3-16TT4_SS_01.txt'
    max_fr = 27
    plot_heatmap(active_grid_heatmaps[cellname]['baseline'],dest_dir + '/baseline_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['post'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_spikes(active_grid_tracking[cellname]['baseline'],dest_dir + '/baseline_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['exp'],dest_dir + '/raised_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['post'],dest_dir + '/floor2_path_spikes.png')
    
    dest_dir = fig_dir + '/Fig1/panel_C/PL76_1-13_TT2_C2'
    cellname = 'PL762020-1-13TT2_SS_02.txt'
    max_fr = 13
    plot_heatmap(active_grid_heatmaps[cellname]['baseline'],dest_dir + '/baseline_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['post'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_spikes(active_grid_tracking[cellname]['baseline'],dest_dir + '/baseline_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['exp'],dest_dir + '/raised_path_spikes.png')
    plot_spikes(active_grid_tracking[cellname]['post'],dest_dir + '/floor2_path_spikes.png')

    
    ''' fig 1 panel D '''
    corr_data = [onetwo_corrs, onethree_corrs]
    corr_ids = [['Floor1 v. Raised'] * len(onetwo_corrs), ['Floor1 v. Floor2'] * len(onethree_corrs)]
    shuffle_vals = [np.percentile(exp_bootstrapped_corr_means,95), np.percentile(post_bootstrapped_corr_means,95)]
    dest = fig_dir + '/Fig1/panel_D.png'
    strip_plot(corr_data, corr_ids, cellnames, [-.5,1],'Rate map correlation',dest, shuffle_vals=shuffle_vals)
    
    ''' fig 1 panel E '''
    
    exp_pre = np.array(active_grid_values[active_grid_values['session']=='1m active raised s3']['field spacing']) - np.array(active_grid_values[active_grid_values['session']=='1m floor s2']['field spacing'])
    post_pre = np.array(active_grid_values[active_grid_values['session']=='1m floor s4']['field spacing']) - np.array(active_grid_values[active_grid_values['session']=='1m floor s2']['field spacing'])

    spacing_data = [list(exp_pre), list(post_pre)]
    spacing_ids = [['Raised-Floor1'] * len(exp_pre), ['Floor2-Floor1'] * len(post_pre)]
    dest = fig_dir + '/Fig1/panel_E.png'
    strip_plot(spacing_data, spacing_ids, cellnames, [-45,50], r'$\Delta$' + ' Field spacing (cm)', dest)
    
    ''' fig 1 panel F '''
    
    exp_pre = np.array(active_grid_values[active_grid_values['session']=='1m active raised s3']['field radius']) - np.array(active_grid_values[active_grid_values['session']=='1m floor s2']['field radius'])
    post_pre = np.array(active_grid_values[active_grid_values['session']=='1m floor s4']['field radius']) - np.array(active_grid_values[active_grid_values['session']=='1m floor s2']['field radius'])

    size_data = [list(exp_pre), list(post_pre)]
    size_ids = [['Raised-Floor1'] * len(exp_pre), ['Floor2-Floor1'] * len(post_pre)]
    dest = fig_dir + '/Fig1/panel_F.png'
    strip_plot(size_data, size_ids, cellnames, [-12,10], r'$\Delta$' + ' Field radius (cm)', dest)
    
    ''' fig 1 panel G '''
    
    exp_pre = np.array(active_grid_values[active_grid_values['session']=='1m active raised s3']['coherence']) - np.array(active_grid_values[active_grid_values['session']=='1m floor s2']['coherence'])
    post_pre = np.array(active_grid_values[active_grid_values['session']=='1m floor s4']['coherence']) - np.array(active_grid_values[active_grid_values['session']=='1m floor s2']['coherence'])

    coherence_data = [list(exp_pre), list(post_pre)]
    coherence_ids = [['Raised-Floor1'] * len(exp_pre), ['Floor2-Floor1'] * len(post_pre)]
    dest = fig_dir + '/Fig1/panel_G.png'
    strip_plot(coherence_data, coherence_ids, cellnames, [-.4, .4], r'$\Delta$' + ' Rate map coherence', dest)
    
    ''' fig 1 panel H '''
    
    exp_pre = np.array(active_grid_values[active_grid_values['session']=='1m active raised s3']['gridness']) - np.array(active_grid_values[active_grid_values['session']=='1m floor s2']['gridness'])
    post_pre = np.array(active_grid_values[active_grid_values['session']=='1m floor s4']['gridness']) - np.array(active_grid_values[active_grid_values['session']=='1m floor s2']['gridness'])

    gridness_data = [list(exp_pre), list(post_pre)]
    gridness_ids = [['Raised-Floor1'] * len(exp_pre), ['Floor2-Floor1'] * len(post_pre)]
    dest = fig_dir + '/Fig1/panel_H.png'
    strip_plot(gridness_data, gridness_ids, cellnames, [-2.3, 2], r'$\Delta$' + ' Grid score', dest)
    
    ''' fig 1 panel I '''
    
    pre_peaks = np.nanmax(np.array(all_floor1_heatmaps).reshape((len(all_floor1_heatmaps),1440)),axis=1)
    exp_peaks = np.nanmax(np.array(all_raised_heatmaps).reshape((len(all_raised_heatmaps),1440)),axis=1)
    post_peaks = np.nanmax(np.array(all_floor2_heatmaps).reshape((len(all_floor2_heatmaps),1440)),axis=1)
    
    exp_pre = exp_peaks - pre_peaks
    post_pre = post_peaks - pre_peaks

    peak_data = [list(exp_pre), list(post_pre)]
    peak_ids = [['Raised-Floor1'] * len(exp_pre), ['Floor2-Floor1'] * len(post_pre)]
    dest = fig_dir + '/Fig1/panel_I.png'
    strip_plot(peak_data, peak_ids, cellnames, [-20, 15], r'$\Delta$' + ' Peak firing rate (Hz)', dest)
    

    ''' rotating and correlating autocorrelations across sessions to determine grid orientation differences '''
    all_exp_corrs = []
    for i in range(len(pre_autocorrs)):
        
        pre = pre_autocorrs[i]
        exp = exp_autocorrs[i]
        
        exp_corrs = []
        
        for rot in range(-90,90):
        
            rot_pre = rotate(pre,rot,cval=np.nan,preserve_range=True)
            rot_pre = np.ma.masked_invalid(rot_pre)
            exp = np.ma.masked_invalid(exp)
            
            corr,p = mapearsonr(rot_pre.flatten(),exp.flatten())
            exp_corrs.append(corr)
            
        all_exp_corrs.append(exp_corrs)
            
    
    ''' figure 2 panels '''
    
    dest_folder = fig_dir + '/Fig2'
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    ''' panel A '''
    example_pre_autocorr = pre_autocorrs[cellnames.index('PL762019-12-10TT3_SS_09.txt')]
    example_exp_autocorr = exp_autocorrs[cellnames.index('PL762019-12-10TT3_SS_09.txt')]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(example_pre_autocorr,cmap='viridis',origin='lower')
    ax.axis('off')
    fig.savefig(dest_folder + '/panel_A_left.png', dpi=400)
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(example_exp_autocorr,cmap='viridis',origin='lower')
    ax.axis('off')
    fig.savefig(dest_folder + '/panel_A_middle.png', dpi=400)
    plt.close()

    example_rotations = all_exp_corrs[cellnames.index('PL762019-12-10TT3_SS_09.txt')]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(-90,90),example_rotations,'black',linewidth=3)
    ax.set_xticks([-60,0,60])
    ax.set_ylim([-.35,.75])
    ax.set_yticks([-.2,0,.2,.4,.6])
    ax.set_ylabel('Correlation with Raised')
    ax.set_xlabel('Floor 1 rotation (deg)')
    plt.tight_layout()
    fig.savefig(dest_folder + '/panel_A_right.png', dpi=400)
    plt.close()

    
    ''' panel B '''
    all_exp_corrs = np.array(all_exp_corrs)
    exp_sds = np.std(all_exp_corrs,axis=0)
    exp_sems = sem(all_exp_corrs,axis=0)

    exp_means = np.mean(all_exp_corrs,axis=0)    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(-90,90), exp_means, 'black',linewidth=3)
    ax.fill_between(np.arange(-90,90), exp_means - exp_sds, exp_means + exp_sds, color='gray',alpha=.5)
    ax.fill_between(np.arange(-90,90), exp_means - exp_sems, exp_means + exp_sems, color='gray',alpha=.8)
    ax.set_xlim([-90,90])
    ax.set_xticks([-60,0,60])
    ax.set_ylim([-.15,.45])
    ax.set_ylabel('Correlation with Raised')
    ax.set_xlabel('Floor 1 rotation (deg)')
    fig.savefig(dest_folder + '/panel_B', dpi=400)
    plt.close()


    ''' determine grid pattern translation between floor and raised sessions '''
    
    xcenter = 36.
    ycenter = 40.
    bin_size = 2.5**2 #cm

    closest_cms = []
    
    for i in range(len(crosscorrs)):
        
        threshed = copy.deepcopy(crosscorrs[i])
        threshed[threshed<.2*np.nanmax(threshed)]=0
        nodes,num_nodes = ndimage.label(threshed)
        
        for k in range(1,num_nodes+1):
            if np.sum(nodes==k) * bin_size < 150 or np.sum(nodes==k) * bin_size > 1500:
                nodes[nodes==k] = 0
                
        full_nodes = np.unique(nodes)
        
        node_size = []
        center_of_mass = []
        
        for k in full_nodes:
            if k != 0:
                node_size.append(bin_size * np.sum(nodes==k))
                only_k = copy.deepcopy(nodes)
                only_k[nodes!=k] = 0
                only_k[nodes==k] = 1
                center_of_mass.append(ndimage.center_of_mass(only_k))
                
        cms = np.array(center_of_mass)
        xdists = cms[:,1] - xcenter
        ydists = cms[:,0] - ycenter
        dists = np.sqrt(xdists**2+ydists**2)
        closest_cm = cms[np.argmin(dists)]
        
        closest_cms.append(closest_cm)
        
    closest_cms = np.array(closest_cms)
    trialnames = np.array(trialnames)
    animalnames = np.array(animalnames)
    
    
    ''' panel C '''
    dest_dir = fig_dir + '/Fig2/panel_C/PL76_12-13_TT1_C2'
    cellname = 'PL762019-12-13TT1_SS_02.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 25
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')

    dest_dir = fig_dir + '/Fig2/panel_C/PL76_12-13_TT1_C3'
    cellname = 'PL762019-12-13TT1_SS_03.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 11
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')
    
    dest_dir = fig_dir + '/Fig2/panel_C/PL76_12-13_TT2_C3'
    cellname = 'PL762019-12-13TT2_SS_03.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 22
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')
    
    dest_dir = fig_dir + '/Fig2/panel_C/PL76_1-13_TT2_C2'
    cellname = 'PL762020-1-13TT2_SS_02.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 13
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')
    
    dest_dir = fig_dir + '/Fig2/panel_C/PL76_1-13_TT2_C3'
    cellname = 'PL762020-1-13TT2_SS_03.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 5
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')
    
    dest_dir = fig_dir + '/Fig2/panel_C/PL76_1-13_TT2_C4'
    cellname = 'PL762020-1-13TT2_SS_04.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 15
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')
    

    ''' panel D '''
    dest_dir = fig_dir + '/Fig2/panel_D/SSW103_3-16_TT4_C1'
    cellname = 'SSW1032016-3-16TT4_SS_01.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 27
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')
    
    dest_dir = fig_dir + '/Fig2/panel_D/AM8_4-9_TT2_C1'
    cellname = 'AM82021-4-9TT2_SS_01.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 19
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')
    
    dest_dir = fig_dir + '/Fig2/panel_D/PL87_2-19_TT2_C2'
    cellname = 'PL872021-2-19TT2_SS_02.txt'
    crosscorr = crosscorrs[cellnames.index(cellname)]
    closest_cm = closest_cms[cellnames.index(cellname)]
    max_fr = 23
    plot_heatmap(active_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_grid_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_crosscorr_translation(crosscorr,closest_cm,dest_dir + '/crosscorr.png')
    
    ''' panel E '''
    
    within_dists = []
    within_animal_dists = []
    without_dists = []
    for i in range(len(trialnames)):
        
        for j in range(len(trialnames)):
            
            if i > j:
                
                dist = np.sqrt((closest_cms[i,1] - closest_cms[j,1])**2 + (closest_cms[i,0] - closest_cms[j,0])**2)
                
                if trialnames[i] == trialnames[j]:
                    within_dists.append(dist)
                elif animalnames[i] != animalnames[j]:
                    without_dists.append(dist)
                elif animalnames[i] == animalnames[j]:
                    within_animal_dists.append(dist)

    within_counts, xbins = np.histogram(within_dists,bins=np.linspace(0,60,31))
    without_counts, xbins = np.histogram(without_dists,bins=np.linspace(0,60,31))
    within_animal_counts, xbins = np.histogram(within_animal_dists,bins=np.linspace(0,60,31))

    within_counts = within_counts.astype(float)/np.sum(within_counts.astype(float))
    without_counts = without_counts.astype(float)/np.sum(without_counts.astype(float))
    within_animal_counts = within_animal_counts.astype(float)/np.sum(within_animal_counts.astype(float))

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True,sharey=True)
    fig.set_figheight(7)
    ax1.hist(xbins[:-1],xbins,weights=within_counts,color='violet')
    ax2.hist(xbins[:-1],xbins,weights=within_animal_counts,color='black')
    ax3.hist(xbins[:-1],xbins,weights=without_counts,color='gray')
    ax1.set_ylim([0,.28])
    ax1.set_yticks([0,.2])
    ax2.set_yticks([0,.2])
    ax3.set_yticks([0,.2])
    ax3.set_xlabel('Translation difference (cm)')
    ax2.set_ylabel('Fraction of cell pairs')
    plt.subplots_adjust(hspace=0)
    fig.savefig(dest_folder+'/panel_E.png',dpi=400)
    plt.close()
    
    
    ''' Figure 3 '''
    
    data_dir = os.getcwd() + '/data'

    passive_grid_values = pd.read_csv(data_dir + '/passive_grid_values.csv')
    passive_grid_heatmaps = pickle.load(open(data_dir + '/passive_grid_heatmaps.pickle','rb'))
    passive_grid_tracking = pickle.load(open(data_dir + '/passive_grid_tracking.pickle','rb'))
    
    passive_grid_values['unique_id'] = passive_grid_values['animal'] + passive_grid_values['day'] + passive_grid_values['cell']
    
    all_floor1_heatmaps = []
    all_raised1_heatmaps = []
    all_floor2_heatmaps = []
    all_raised2_heatmaps = []
    all_floor3_heatmaps = []
    for cell in passive_grid_heatmaps.keys():
        all_floor1_heatmaps.append(passive_grid_heatmaps[cell]['pre'])
        all_raised1_heatmaps.append(passive_grid_heatmaps[cell]['exp1'])
        all_floor2_heatmaps.append(passive_grid_heatmaps[cell]['post1'])
        all_raised2_heatmaps.append(passive_grid_heatmaps[cell]['exp2'])
        all_floor3_heatmaps.append(passive_grid_heatmaps[cell]['post2'])
        
    ''' computing shuffle distribution for ratemap correlations '''

    exp1_bootstrapped_corr_means = []
    post1_bootstrapped_corr_means = []
    exp2_bootstrapped_corr_means = []
    post2_bootstrapped_corr_means = []
    ap_bootstrapped_corr_means = []
    
    for shuffle in range(1000):
        shuffle_exp1 = []
        shuffle_post1 = []
        shuffle_exp2 = []
        shuffle_post2 = []
        shuffle_ap = []
        
        shuffled_inds = np.arange(len(all_floor1_heatmaps))
        np.random.shuffle(shuffled_inds)
        
        for i in range(len(shuffled_inds)):
            pre1 = all_floor1_heatmaps[i]
            exp1 = all_raised1_heatmaps[shuffled_inds[i]]
            shuffle_exp1.append(pearsonr(pre1.flatten(),exp1.flatten())[0])
            
            post1 = all_floor2_heatmaps[shuffled_inds[i]]
            shuffle_post1.append(pearsonr(pre1.flatten(),post1.flatten())[0])
            
            pre2 = all_floor2_heatmaps[i]
            exp2 = all_raised2_heatmaps[shuffled_inds[i]]
            shuffle_exp2.append(pearsonr(pre2.flatten(),exp2.flatten())[0])
            
            post2 = all_floor3_heatmaps[shuffled_inds[i]]
            shuffle_post2.append(pearsonr(pre2.flatten(),post2.flatten())[0])
            
            active = all_raised1_heatmaps[i]
            passive = all_raised2_heatmaps[shuffled_inds[i]]
            shuffle_ap.append(pearsonr(active.flatten(),passive.flatten())[0])
            
        exp1_bootstrapped_corr_means.append(np.mean(shuffle_exp1))
        post1_bootstrapped_corr_means.append(np.mean(shuffle_post1))
        exp2_bootstrapped_corr_means.append(np.mean(shuffle_exp2))
        post2_bootstrapped_corr_means.append(np.mean(shuffle_post2))
        ap_bootstrapped_corr_means.append(np.mean(shuffle_ap))
            
    ''' computing ratemap correlations '''
    onetwo_corrs = []
    onethree_corrs = []
    twofour_corrs = []
    threefour_corrs = []
    threefive_corrs = []
    for i in range(len(all_floor1_heatmaps)):
        onetwo_corrs.append(pearsonr(all_floor1_heatmaps[i].flatten(),all_raised1_heatmaps[i].flatten())[0])
        onethree_corrs.append(pearsonr(all_floor1_heatmaps[i].flatten(),all_floor2_heatmaps[i].flatten())[0])
        twofour_corrs.append(pearsonr(all_raised1_heatmaps[i].flatten(),all_raised2_heatmaps[i].flatten())[0])
        threefour_corrs.append(pearsonr(all_floor2_heatmaps[i].flatten(),all_raised2_heatmaps[i].flatten())[0])
        threefive_corrs.append(pearsonr(all_floor2_heatmaps[i].flatten(),all_floor3_heatmaps[i].flatten())[0])
        
    cellnames = list(passive_grid_heatmaps.keys())
        
    ''' fig 3 panel B (example cells) '''
    dest_dir = fig_dir + '/Fig3/panel_B/PL76_1-13_TT2_C4'
    cellname = 'PL762020-1-13TT2_SS_04.txt'
    max_fr = 17
    plot_heatmap(passive_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['exp1'],dest_dir + '/active_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['post1'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['exp2'],dest_dir + '/passive_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['post2'],dest_dir + '/floor3_heatmap.png',max_fr=max_fr)
    plot_spikes(passive_grid_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['exp1'],dest_dir + '/active_raised_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['post1'],dest_dir + '/floor2_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['exp2'],dest_dir + '/passive_raised_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['post2'],dest_dir + '/floor3_path_spikes.png')
    
    dest_dir = fig_dir + '/Fig3/panel_B/PL76_12-13_TT2_C3'
    cellname = 'PL762019-12-13TT2_SS_03.txt'
    max_fr = 22
    plot_heatmap(passive_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['exp1'],dest_dir + '/active_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['post1'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['exp2'],dest_dir + '/passive_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['post2'],dest_dir + '/floor3_heatmap.png',max_fr=max_fr)
    plot_spikes(passive_grid_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['exp1'],dest_dir + '/active_raised_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['post1'],dest_dir + '/floor2_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['exp2'],dest_dir + '/passive_raised_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['post2'],dest_dir + '/floor3_path_spikes.png')
    
    dest_dir = fig_dir + '/Fig3/panel_B/PL76_12-10_TT3_C9'
    cellname = 'PL762019-12-10TT3_SS_09.txt'
    max_fr = 22
    plot_heatmap(passive_grid_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['exp1'],dest_dir + '/active_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['post1'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['exp2'],dest_dir + '/passive_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_grid_heatmaps[cellname]['post2'],dest_dir + '/floor3_heatmap.png',max_fr=max_fr)
    plot_spikes(passive_grid_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['exp1'],dest_dir + '/active_raised_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['post1'],dest_dir + '/floor2_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['exp2'],dest_dir + '/passive_raised_path_spikes.png')
    plot_spikes(passive_grid_tracking[cellname]['post2'],dest_dir + '/floor3_path_spikes.png')


    ''' panel C '''

    corr_data = [onetwo_corrs, onethree_corrs, threefour_corrs, threefive_corrs, twofour_corrs]
    corr_ids = [['F1 v. A'] * len(onetwo_corrs), ['F1 v. F2'] * len(onethree_corrs), ['F2 v. P'] * len(threefour_corrs),
                ['F2 v. F3'] * len(threefive_corrs), ['A v. P'] * len(twofour_corrs)]
    shuffle_vals = [np.percentile(exp1_bootstrapped_corr_means,95), np.percentile(post1_bootstrapped_corr_means,95),np.percentile(exp2_bootstrapped_corr_means,95), np.percentile(post2_bootstrapped_corr_means,95), np.percentile(ap_bootstrapped_corr_means,95)]
    dest = fig_dir + '/Fig3/panel_C.png'
    strip_plot(corr_data, corr_ids, cellnames, [-.5,1],'Rate map correlation',dest, shuffle_vals=shuffle_vals)

    ''' panel D '''
    
    spacing_data = [list(np.array(passive_grid_values[passive_grid_values['session']=='1m active raised s3']['field spacing']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s2']['field spacing']))]
    spacing_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['field spacing']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s2']['field spacing'])))
    spacing_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m passive raised s5']['field spacing']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['field spacing'])))
    spacing_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m floor s6']['field spacing']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['field spacing'])))
    spacing_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m passive raised s5']['field spacing']) - np.array(passive_grid_values[passive_grid_values['session']=='1m active raised s3']['field spacing'])))
    spacing_ids = [['A - F1'] * len(onetwo_corrs), ['F2 - F1'] * len(onethree_corrs), ['P - F2'] * len(threefour_corrs),
                ['F3 - F2'] * len(threefive_corrs), ['P - A'] * len(twofour_corrs)]

    dest = fig_dir + '/Fig3/panel_D.png'
    strip_plot(spacing_data, spacing_ids, cellnames, [-45,50], r'$\Delta$' + ' Field spacing (cm)', dest)
    
    ''' panel E '''
    
    size_data = [list(np.array(passive_grid_values[passive_grid_values['session']=='1m active raised s3']['field radius']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s2']['field radius']))]
    size_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['field radius']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s2']['field radius'])))
    size_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m passive raised s5']['field radius']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['field radius'])))
    size_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m floor s6']['field radius']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['field radius'])))
    size_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m passive raised s5']['field radius']) - np.array(passive_grid_values[passive_grid_values['session']=='1m active raised s3']['field radius'])))
    size_ids = [['A - F1'] * len(onetwo_corrs), ['F2 - F1'] * len(onethree_corrs), ['P - F2'] * len(threefour_corrs),
                ['F3 - F2'] * len(threefive_corrs), ['P - A'] * len(twofour_corrs)]

    dest = fig_dir + '/Fig3/panel_E.png'
    strip_plot(size_data, size_ids, cellnames, [-15,10], r'$\Delta$' + ' Field radius (cm)', dest)
    
    ''' panel F '''
    
    coherence_data = [list(np.array(passive_grid_values[passive_grid_values['session']=='1m active raised s3']['coherence']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s2']['coherence']))]
    coherence_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['coherence']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s2']['coherence'])))
    coherence_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m passive raised s5']['coherence']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['coherence'])))
    coherence_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m floor s6']['coherence']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['coherence'])))
    coherence_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m passive raised s5']['coherence']) - np.array(passive_grid_values[passive_grid_values['session']=='1m active raised s3']['coherence'])))
    coherence_ids = [['A - F1'] * len(onetwo_corrs), ['F2 - F1'] * len(onethree_corrs), ['P - F2'] * len(threefour_corrs),
                ['F3 - F2'] * len(threefive_corrs), ['P - A'] * len(twofour_corrs)]

    dest = fig_dir + '/Fig3/panel_F.png'
    strip_plot(coherence_data, coherence_ids, cellnames, [-.4,.4], r'$\Delta$' + ' Rate map coherence', dest)
    
    
    ''' panel G '''
    
    pre_peaks = np.nanmax(np.array(all_floor1_heatmaps).reshape((len(all_floor1_heatmaps),1440)),axis=1)
    exp1_peaks = np.nanmax(np.array(all_raised1_heatmaps).reshape((len(all_raised1_heatmaps),1440)),axis=1)
    post1_peaks = np.nanmax(np.array(all_floor2_heatmaps).reshape((len(all_floor2_heatmaps),1440)),axis=1)
    exp2_peaks = np.nanmax(np.array(all_raised2_heatmaps).reshape((len(all_raised2_heatmaps),1440)),axis=1)
    post2_peaks = np.nanmax(np.array(all_floor3_heatmaps).reshape((len(all_floor3_heatmaps),1440)),axis=1)
    
    exp1_pre = exp1_peaks - pre_peaks
    post1_pre = post1_peaks - pre_peaks
    exp2_post1 = exp2_peaks - post1_peaks
    post2_post1 = post2_peaks - post1_peaks
    active_passive = exp2_peaks - exp1_peaks
    
    peak_data = [list(exp1_pre), list(post1_pre), list(exp2_post1), list(post2_post1), list(active_passive)]
    peak_ids = [['A - F1'] * len(onetwo_corrs), ['F2 - F1'] * len(onethree_corrs), ['P - F2'] * len(threefour_corrs),
                ['F3 - F2'] * len(threefive_corrs), ['P - A'] * len(twofour_corrs)]
    
    dest = fig_dir + '/Fig3/panel_G.png'
    strip_plot(peak_data, peak_ids, cellnames, [-20,20], r'$\Delta$' + ' Peak firing rate (Hz)', dest)
    
    
    ''' panel H '''
    
    gridness_data = [list(np.array(passive_grid_values[passive_grid_values['session']=='1m active raised s3']['gridness']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s2']['gridness']))]
    gridness_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['gridness']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s2']['gridness'])))
    gridness_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m passive raised s5']['gridness']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['gridness'])))
    gridness_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m floor s6']['gridness']) - np.array(passive_grid_values[passive_grid_values['session']=='1m floor s4']['gridness'])))
    gridness_data.append(list(np.array(passive_grid_values[passive_grid_values['session']=='1m passive raised s5']['gridness']) - np.array(passive_grid_values[passive_grid_values['session']=='1m active raised s3']['gridness'])))
    gridness_ids = [['A - F1'] * len(onetwo_corrs), ['F2 - F1'] * len(onethree_corrs), ['P - F2'] * len(threefour_corrs),
                ['F3 - F2'] * len(threefive_corrs), ['P - A'] * len(twofour_corrs)]

    dest = fig_dir + '/Fig3/panel_H.png'
    strip_plot(gridness_data, gridness_ids, cellnames, [-2,2], r'$\Delta$' + ' Grid score', dest)
    
    
    
    ''' Figure 4 '''
    
    active_spatial_values = pd.read_csv(data_dir + '/active_spatial_values.csv')
    active_spatial_heatmaps = pickle.load(open(data_dir + '/active_spatial_heatmaps.pickle','rb'))
    active_spatial_tracking = pickle.load(open(data_dir + '/active_spatial_tracking.pickle','rb'))
    
    active_spatial_values['unique_id'] = active_spatial_values['animal'] + active_spatial_values['day'] + active_spatial_values['cell']
    
    all_floor1_heatmaps = []
    all_raised_heatmaps = []
    all_floor2_heatmaps = []
    for cell in active_spatial_heatmaps.keys():
        all_floor1_heatmaps.append(active_spatial_heatmaps[cell]['pre'])
        all_raised_heatmaps.append(active_spatial_heatmaps[cell]['exp'])
        all_floor2_heatmaps.append(active_spatial_heatmaps[cell]['post'])
    
    ''' computing shuffle distribution for ratemap correlations '''

    exp_bootstrapped_corr_means = []
    post_bootstrapped_corr_means = []
    
    for shuffle in range(1000):
        shuffle_exp = []
        shuffle_post = []
        
        shuffled_inds = np.arange(len(all_floor1_heatmaps))
        np.random.shuffle(shuffled_inds)
        
        for i in range(len(shuffled_inds)):
            pre = all_floor1_heatmaps[i]
            exp = all_raised_heatmaps[shuffled_inds[i]]
            shuffle_exp.append(pearsonr(pre.flatten(),exp.flatten())[0])

            post = all_floor2_heatmaps[shuffled_inds[i]]
            shuffle_post.append(pearsonr(pre.flatten(),post.flatten())[0])
            
        exp_bootstrapped_corr_means.append(np.mean(shuffle_exp))
        post_bootstrapped_corr_means.append(np.mean(shuffle_post))

    ''' computing ratemap correlations '''
    
    onetwo_corrs = []
    onethree_corrs = []

    cellnames = []
    trialnames = []
    animalnames = []

    for cell in active_spatial_heatmaps.keys():
        onetwo_corrs.append(pearsonr(active_spatial_heatmaps[cell]['pre'].flatten(),active_spatial_heatmaps[cell]['exp'].flatten())[0])
        onethree_corrs.append(pearsonr(active_spatial_heatmaps[cell]['pre'].flatten(),active_spatial_heatmaps[cell]['post'].flatten())[0])
        
        cellnames.append(cell)
        trialnames.append(active_spatial_values[active_spatial_values['unique_id']==cell].iloc[0]['day'] + active_spatial_values[active_spatial_values['unique_id']==cell].iloc[0]['animal'])
        animalnames.append(active_spatial_values[active_spatial_values['unique_id']==cell].iloc[0]['animal'])

    ''' fig 4 panel A (example cell) '''
    dest_dir = fig_dir + '/Fig4/panel_A/PL68_4-22a_TT2_C2'
    cellname = 'PL682019-4-22aTT2_SS_02.txt'
    max_fr = 23
    plot_heatmap(active_spatial_heatmaps[cellname]['baseline'],dest_dir + '/baseline_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_spatial_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_spatial_heatmaps[cellname]['exp'],dest_dir + '/raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(active_spatial_heatmaps[cellname]['post'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_spikes(active_spatial_tracking[cellname]['baseline'],dest_dir + '/baseline_path_spikes.png')
    plot_spikes(active_spatial_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(active_spatial_tracking[cellname]['exp'],dest_dir + '/raised_path_spikes.png')
    plot_spikes(active_spatial_tracking[cellname]['post'],dest_dir + '/floor2_path_spikes.png')
    
    
    ''' fig 4 panel B '''
    
    corr_data = [onetwo_corrs, onethree_corrs]
    corr_ids = [['Raised v. Floor1'] * len(onetwo_corrs), ['Floor2 v. Floor1'] * len(onethree_corrs)]
    shuffle_vals = [np.percentile(exp_bootstrapped_corr_means,95), np.percentile(post_bootstrapped_corr_means,95)]
    dest = fig_dir + '/Fig4/panel_B.png'
    strip_plot(corr_data, corr_ids, cellnames, [-.5,1],'Rate map correlation',dest, shuffle_vals=shuffle_vals)
    
    ''' fig 4 panel C '''
    
    exp_pre = np.array(active_spatial_values[active_spatial_values['session']=='1m active raised s3']['coherence']) - np.array(active_spatial_values[active_spatial_values['session']=='1m floor s2']['coherence'])
    post_pre = np.array(active_spatial_values[active_spatial_values['session']=='1m floor s4']['coherence']) - np.array(active_spatial_values[active_spatial_values['session']=='1m floor s2']['coherence'])

    coherence_data = [list(exp_pre), list(post_pre)]
    coherence_ids = [['Raised-Floor1'] * len(exp_pre), ['Floor2-Floor1'] * len(post_pre)]
    dest = fig_dir + '/Fig4/panel_C.png'
    strip_plot(coherence_data, coherence_ids, cellnames, [-.4, .4], r'$\Delta$' + ' Rate map coherence', dest)
    
    ''' fig 4 panel D '''
    
    pre_peaks = np.nanmax(np.array(all_floor1_heatmaps).reshape((len(all_floor1_heatmaps),1440)),axis=1)
    exp_peaks = np.nanmax(np.array(all_raised_heatmaps).reshape((len(all_raised_heatmaps),1440)),axis=1)
    post_peaks = np.nanmax(np.array(all_floor2_heatmaps).reshape((len(all_floor2_heatmaps),1440)),axis=1)
    
    exp_pre = exp_peaks - pre_peaks
    post_pre = post_peaks - pre_peaks

    peak_data = [list(exp_pre), list(post_pre)]
    peak_ids = [['Raised-Floor1'] * len(exp_pre), ['Floor2-Floor1'] * len(post_pre)]
    dest = fig_dir + '/Fig4/panel_D.png'
    strip_plot(peak_data, peak_ids, cellnames, [-20, 15], r'$\Delta$' + ' Peak firing rate (Hz)', dest)

    ''' fig 4 panel E '''
    
    exp_pre = np.array(active_spatial_values[active_spatial_values['session']=='1m active raised s3']['spatial info']) - np.array(active_spatial_values[active_spatial_values['session']=='1m floor s2']['spatial info'])
    post_pre = np.array(active_spatial_values[active_spatial_values['session']=='1m floor s4']['spatial info']) - np.array(active_spatial_values[active_spatial_values['session']=='1m floor s2']['spatial info'])

    spatial_info_data = [list(exp_pre), list(post_pre)]
    spatial_info_ids = [['Raised-Floor1'] * len(exp_pre), ['Floor2-Floor1'] * len(post_pre)]
    dest = fig_dir + '/Fig4/panel_E.png'
    strip_plot(spatial_info_data, spatial_info_ids, cellnames, [-.6, .6], r'$\Delta$' + ' Spatial info (bits/spike)', dest)
    
    

    ''' passive sessions for nongrid cells '''
    
    passive_spatial_values = pd.read_csv(data_dir + '/passive_spatial_values.csv')
    passive_spatial_heatmaps = pickle.load(open(data_dir + '/passive_spatial_heatmaps.pickle','rb'))
    passive_spatial_tracking = pickle.load(open(data_dir + '/passive_spatial_tracking.pickle','rb'))
    
    passive_spatial_values['unique_id'] = passive_spatial_values['animal'] + passive_spatial_values['day'] + passive_spatial_values['cell']
    
    all_floor1_heatmaps = []
    all_raised1_heatmaps = []
    all_floor2_heatmaps = []
    all_raised2_heatmaps = []
    all_floor3_heatmaps = []
    for cell in passive_spatial_heatmaps.keys():
        all_floor1_heatmaps.append(passive_spatial_heatmaps[cell]['pre'])
        all_raised1_heatmaps.append(passive_spatial_heatmaps[cell]['exp1'])
        all_floor2_heatmaps.append(passive_spatial_heatmaps[cell]['post1'])
        all_raised2_heatmaps.append(passive_spatial_heatmaps[cell]['exp2'])
        all_floor3_heatmaps.append(passive_spatial_heatmaps[cell]['post2'])
        
    ''' computing shuffle distribution for ratemap correlations '''

    exp1_bootstrapped_corr_means = []
    post1_bootstrapped_corr_means = []
    exp2_bootstrapped_corr_means = []
    post2_bootstrapped_corr_means = []
    ap_bootstrapped_corr_means = []
    
    for shuffle in range(1000):
        shuffle_exp1 = []
        shuffle_post1 = []
        shuffle_exp2 = []
        shuffle_post2 = []
        shuffle_ap = []
        
        shuffled_inds = np.arange(len(all_floor1_heatmaps))
        np.random.shuffle(shuffled_inds)
        
        for i in range(len(shuffled_inds)):
            pre1 = all_floor1_heatmaps[i]
            exp1 = all_raised1_heatmaps[shuffled_inds[i]]
            shuffle_exp1.append(pearsonr(pre1.flatten(),exp1.flatten())[0])
            
            post1 = all_floor2_heatmaps[shuffled_inds[i]]
            shuffle_post1.append(pearsonr(pre1.flatten(),post1.flatten())[0])
            
            pre2 = all_floor2_heatmaps[i]
            exp2 = all_raised2_heatmaps[shuffled_inds[i]]
            shuffle_exp2.append(pearsonr(pre2.flatten(),exp2.flatten())[0])
            
            post2 = all_floor3_heatmaps[shuffled_inds[i]]
            shuffle_post2.append(pearsonr(pre2.flatten(),post2.flatten())[0])
            
            active = all_raised1_heatmaps[i]
            passive = all_raised2_heatmaps[shuffled_inds[i]]
            shuffle_ap.append(pearsonr(active.flatten(),passive.flatten())[0])
            
        exp1_bootstrapped_corr_means.append(np.mean(shuffle_exp1))
        post1_bootstrapped_corr_means.append(np.mean(shuffle_post1))
        exp2_bootstrapped_corr_means.append(np.mean(shuffle_exp2))
        post2_bootstrapped_corr_means.append(np.mean(shuffle_post2))
        ap_bootstrapped_corr_means.append(np.mean(shuffle_ap))
            
    ''' computing ratemap correlations '''
    
    onetwo_corrs = []
    onethree_corrs = []
    twofour_corrs = []
    threefour_corrs = []
    threefive_corrs = []
    for i in range(len(all_floor1_heatmaps)):
        onetwo_corrs.append(pearsonr(all_floor1_heatmaps[i].flatten(),all_raised1_heatmaps[i].flatten())[0])
        onethree_corrs.append(pearsonr(all_floor1_heatmaps[i].flatten(),all_floor2_heatmaps[i].flatten())[0])
        twofour_corrs.append(pearsonr(all_raised1_heatmaps[i].flatten(),all_raised2_heatmaps[i].flatten())[0])
        threefour_corrs.append(pearsonr(all_floor2_heatmaps[i].flatten(),all_raised2_heatmaps[i].flatten())[0])
        threefive_corrs.append(pearsonr(all_floor2_heatmaps[i].flatten(),all_floor3_heatmaps[i].flatten())[0])
        
    cellnames = list(passive_spatial_heatmaps.keys())
        
    ''' fig 4 panel F (example cells) '''
    dest_dir = fig_dir + '/Fig4/panel_F/PL87_2-19_TT2_C4'
    cellname = 'PL872021-2-19TT2_SS_04.txt'
    max_fr = 21
    plot_heatmap(passive_spatial_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_spatial_heatmaps[cellname]['exp1'],dest_dir + '/active_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_spatial_heatmaps[cellname]['post1'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_spatial_heatmaps[cellname]['exp2'],dest_dir + '/passive_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_spatial_heatmaps[cellname]['post2'],dest_dir + '/floor3_heatmap.png',max_fr=max_fr)
    plot_spikes(passive_spatial_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(passive_spatial_tracking[cellname]['exp1'],dest_dir + '/active_raised_path_spikes.png')
    plot_spikes(passive_spatial_tracking[cellname]['post1'],dest_dir + '/floor2_path_spikes.png')
    plot_spikes(passive_spatial_tracking[cellname]['exp2'],dest_dir + '/passive_raised_path_spikes.png')
    plot_spikes(passive_spatial_tracking[cellname]['post2'],dest_dir + '/floor3_path_spikes.png')
    
    dest_dir = fig_dir + '/Fig4/panel_F/PL76_12-4_TT3_C2'
    cellname = 'PL762019-12-4TT3_SS_02.txt'
    max_fr = 12
    plot_heatmap(passive_spatial_heatmaps[cellname]['pre'],dest_dir + '/floor1_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_spatial_heatmaps[cellname]['exp1'],dest_dir + '/active_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_spatial_heatmaps[cellname]['post1'],dest_dir + '/floor2_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_spatial_heatmaps[cellname]['exp2'],dest_dir + '/passive_raised_heatmap.png',max_fr=max_fr)
    plot_heatmap(passive_spatial_heatmaps[cellname]['post2'],dest_dir + '/floor3_heatmap.png',max_fr=max_fr)
    plot_spikes(passive_spatial_tracking[cellname]['pre'],dest_dir + '/floor1_path_spikes.png')
    plot_spikes(passive_spatial_tracking[cellname]['exp1'],dest_dir + '/active_raised_path_spikes.png')
    plot_spikes(passive_spatial_tracking[cellname]['post1'],dest_dir + '/floor2_path_spikes.png')
    plot_spikes(passive_spatial_tracking[cellname]['exp2'],dest_dir + '/passive_raised_path_spikes.png')
    plot_spikes(passive_spatial_tracking[cellname]['post2'],dest_dir + '/floor3_path_spikes.png')
    
    
    ''' panel G '''

    corr_data = [onetwo_corrs, onethree_corrs, threefour_corrs, threefive_corrs, twofour_corrs]
    corr_ids = [['A v. F1'] * len(onetwo_corrs), ['F2 v. F1'] * len(onethree_corrs), ['P v. F2'] * len(threefour_corrs),
                ['F3 v. F2'] * len(threefive_corrs), ['P v. A'] * len(twofour_corrs)]
    shuffle_vals = [np.percentile(exp1_bootstrapped_corr_means,95), np.percentile(post1_bootstrapped_corr_means,95),np.percentile(exp2_bootstrapped_corr_means,95), np.percentile(post2_bootstrapped_corr_means,95), np.percentile(ap_bootstrapped_corr_means,95)]
    dest = fig_dir + '/Fig4/panel_G.png'
    strip_plot(corr_data, corr_ids, cellnames, [-.5,1],'Rate map correlation',dest, shuffle_vals=shuffle_vals)

    
    ''' panel H '''
    
    coherence_data = [list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m active raised s3']['coherence']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s2']['coherence']))]
    coherence_data.append(list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s4']['coherence']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s2']['coherence'])))
    coherence_data.append(list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m passive raised s5']['coherence']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s4']['coherence'])))
    coherence_data.append(list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s6']['coherence']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s4']['coherence'])))
    coherence_data.append(list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m passive raised s5']['coherence']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m active raised s3']['coherence'])))
    coherence_ids = [['A - F1'] * len(onetwo_corrs), ['F2 - F1'] * len(onethree_corrs), ['P - F2'] * len(threefour_corrs),
                ['F3 - F2'] * len(threefive_corrs), ['P - A'] * len(twofour_corrs)]

    dest = fig_dir + '/Fig4/panel_H.png'
    strip_plot(coherence_data, coherence_ids, cellnames, [-.4,.4], r'$\Delta$' + ' Rate map coherence', dest)
    
    
    ''' panel I '''
    
    pre_peaks = np.nanmax(np.array(all_floor1_heatmaps).reshape((len(all_floor1_heatmaps),1440)),axis=1)
    exp1_peaks = np.nanmax(np.array(all_raised1_heatmaps).reshape((len(all_raised1_heatmaps),1440)),axis=1)
    post1_peaks = np.nanmax(np.array(all_floor2_heatmaps).reshape((len(all_floor2_heatmaps),1440)),axis=1)
    exp2_peaks = np.nanmax(np.array(all_raised2_heatmaps).reshape((len(all_raised2_heatmaps),1440)),axis=1)
    post2_peaks = np.nanmax(np.array(all_floor3_heatmaps).reshape((len(all_floor3_heatmaps),1440)),axis=1)
    
    exp1_pre = exp1_peaks - pre_peaks
    post1_pre = post1_peaks - pre_peaks
    exp2_post1 = exp2_peaks - post1_peaks
    post2_post1 = post2_peaks - post1_peaks
    active_passive = exp2_peaks - exp1_peaks
    
    peak_data = [list(exp1_pre), list(post1_pre), list(exp2_post1), list(post2_post1), list(active_passive)]
    peak_ids = [['A - F1'] * len(onetwo_corrs), ['F2 - F1'] * len(onethree_corrs), ['P - F2'] * len(threefour_corrs),
                ['F3 - F2'] * len(threefive_corrs), ['P - A'] * len(twofour_corrs)]
    
    dest = fig_dir + '/Fig4/panel_I.png'
    strip_plot(peak_data, peak_ids, cellnames, [-20,20], r'$\Delta$' + ' Peak firing rate (Hz)', dest)
    
    
    ''' panel J '''
    
    spatial_info_data = [list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m active raised s3']['spatial info']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s2']['spatial info']))]
    spatial_info_data.append(list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s4']['spatial info']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s2']['spatial info'])))
    spatial_info_data.append(list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m passive raised s5']['spatial info']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s4']['spatial info'])))
    spatial_info_data.append(list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s6']['spatial info']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m floor s4']['spatial info'])))
    spatial_info_data.append(list(np.array(passive_spatial_values[passive_spatial_values['session']=='1m passive raised s5']['spatial info']) - np.array(passive_spatial_values[passive_spatial_values['session']=='1m active raised s3']['spatial info'])))
    spatial_info_ids = [['A - F1'] * len(onetwo_corrs), ['F2 - F1'] * len(onethree_corrs), ['P - F2'] * len(threefour_corrs),
                ['F3 - F2'] * len(threefive_corrs), ['P - A'] * len(twofour_corrs)]

    dest = fig_dir + '/Fig4/panel_J.png'
    strip_plot(spatial_info_data, spatial_info_ids, cellnames, [-.6,.6], r'$\Delta$' + ' Spatial info', dest)
    

if __name__ == '__main__':
    
    create_figures()

# Copyright (c) 2017 by Chris Davis and the other collaborators on GitHub at
# https://github.com/des-science/sompz All rights reserved.
#
# sompz is free software: Redistribution and use in source and binary forms with
# or without modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the disclaimer given in the accompanying
#    LICENSE file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

from __future__ import print_function, division
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import copy
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))) + '/sompz')
from som import evaluate_som, umatrix

# TODO: move this. we shouldnt be doing computation in plots.py
from utils import mag2flux, flux2mag

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.figsize'] = (16.,8.)
dpi = 150

plt.rcParams.update({
'lines.linewidth':1.0,
'lines.linestyle':'-',
'lines.color':'black',
'font.family':'serif',
'font.weight':'bold', #normal
'font.size':16.0, #10.0
'text.color':'black',
'text.usetex':False,
'axes.edgecolor':'black',
'axes.linewidth':1.0,
'axes.grid':False,
'axes.titlesize':'x-large',
'axes.labelsize':'x-large',
'axes.labelweight':'bold', #normal
'axes.labelcolor':'black',
'axes.formatter.limits':[-4,4],
'xtick.major.size':7,
'xtick.minor.size':4,
'xtick.major.pad':8,
'xtick.minor.pad':8,
'xtick.labelsize':'x-large',
'xtick.minor.width':1.0,
'xtick.major.width':1.0,
'ytick.major.size':7,
'ytick.minor.size':4,
'ytick.major.pad':8,
'ytick.minor.pad':8,
'ytick.labelsize':'x-large',
'ytick.minor.width':1.0,
'ytick.major.width':1.0,
'legend.numpoints':1,
'legend.fontsize':'x-large',
'legend.shadow':False,
'legend.frameon':False})

'''
.. module:: plots
'''

def plot_hist_of_meanz_before_and_after_pit(all_means_before, all_means_after, name, suffix):
    fig, axarr = plt.subplots(2,2, figsize=(16,9))
    fig.suptitle(name, fontsize=14)
    for i in range(4):
        sig_before = np.std(all_means_before[:,i])
        sig_after = np.std(all_means_after[:,i])

        mean_before = np.mean(all_means_before[:,i])
        mean_after = np.mean(all_means_after[:,i])

        diffstr = 'bin {} '.format(i+1) + r'$<z>$' + ' after - ' + r'$<z>$' + ' before = {:6.4f}'.format(mean_after-mean_before)
        print(diffstr)
        label = 'before PIT ' + r'$\langle z \rangle = {:.4f}$'.format(mean_before) + '  ' + r'$\sigma_z=  {:.3f}$'.format(sig_before)
        axarr[i//2, i%2].hist(all_means_before[:,i], 
                              histtype='step', bins=30, lw=3, density=True,color=colors[0],
                              label=label)
        label = 'after PIT + shift ' + r'$\langle z \rangle = {:.4f}$'.format(mean_after) + '  ' + r'$\sigma_z=  {:.3f}$'.format(sig_after)
        axarr[i//2, i%2].hist(all_means_after[:,i], 
                              histtype='step', bins=30, lw=3, density=True,color=colors[1],linestyle='--',
                              label=label)

        axarr[i//2,i%2].legend(fontsize=12)
        axarr[i//2,i%2].set_title(diffstr, fontsize=13)#'bin {}'.format(i+1))
        axarr[i//2,i%2].set_xlabel('z')

        axarr[i//2, i%2].axvline(mean_before, color=colors[0], lw=2)
        axarr[i//2, i%2].axvline(mean_after, color=colors[1], lw=2)

    axarr[0,0].set_xlim((0.28, 0.37))
    axarr[0,1].set_xlim((0.48, 0.58))
    axarr[1,0].set_xlim((0.72, 0.80))
    axarr[1,1].set_xlim((0.9, 1.0))
    plt.tight_layout()
    outfile = os.path.join(outdir,'hist_of_meanz_before_and_after_{}{}.png'.format(name, suffix))
    print(outfile)
    plt.savefig(outfile, dpi=150)
                
def plot_ensemble_from_twopoint(infile, outfile, name, nrealizations):
    dat = fits.open(infile)
    #dat.info() 
    #print(dat[1].header)
    data= dat[1].data

    nz=dat['nz_source'].data
    plt.figure(figsize=(16.,9.))
    plt.plot(dat['nz_source'].data['Z_MID'],dat['nz_source'].data['BIN1'],color='blue',lw=3)
    plt.plot(dat['nz_source'].data['Z_MID'],dat['nz_source'].data['BIN2'],color='orange',lw=3)
    plt.plot(dat['nz_source'].data['Z_MID'],dat['nz_source'].data['BIN3'],color='green',lw=3)
    plt.plot(dat['nz_source'].data['Z_MID'],dat['nz_source'].data['BIN4'],color='red',lw=3)
    for i in range(nrealizations):
        plt.plot(dat['nz_source_realisation_{}'.format(i)].data['Z_MID'],dat['nz_source_realisation_{}'.format(i)].data['BIN1'],color='blue',lw=3, alpha=0.01)
        plt.plot(dat['nz_source_realisation_{}'.format(i)].data['Z_MID'],dat['nz_source_realisation_{}'.format(i)].data['BIN2'],color='orange',lw=3, alpha=0.01)
        plt.plot(dat['nz_source_realisation_{}'.format(i)].data['Z_MID'],dat['nz_source_realisation_{}'.format(i)].data['BIN3'],color='green',lw=3, alpha=0.01)
        plt.plot(dat['nz_source_realisation_{}'.format(i)].data['Z_MID'],dat['nz_source_realisation_{}'.format(i)].data['BIN4'],color='red',lw=3, alpha=0.01)
    plt.title('{} sompz + wz + PIT + shifted ensemble'.format(name))
    #plt.xlim((0,0.5))
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.savefig(outfile, dpi=150)
    print('write', outfile)
def plot_nz(hists, zbins, outfile, xlimits=(0,2), ylimits=(0,3.25)):
    plt.figure(figsize=(16.,9.))
    for i in range(len(hists)):
        plt.plot((zbins[1:] + zbins[:-1])/2., hists[i], label='bin ' + str(i))
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$p(z)$')
    plt.legend()
    plt.title('n(z)')
    plt.savefig(outfile)
    print('write {}'.format(outfile))
    plt.close()

def plot_nz_overlap(list_of_nz, list_of_labels, outdir):
    colors = 'rgbcmyk'
    plt.figure(figsize=(16,9))

    ### nz_all is an array: (n_tomobins, n_zbins)
    for j, (nz_all, label) in enumerate(zip(list_of_nz, list_of_labels)):
        nz_all_overlap_somM = np.einsum('mz,nz->mn', nz_all, nz_all)
        nz_all_overlap_var_somM = np.einsum('m,n->mn', np.sqrt(np.diag(nz_all_overlap_somM)), np.sqrt(np.diag(nz_all_overlap_somM)))
        nz_all_overlap_somM /= nz_all_overlap_var_somM

        ### Plot the overlap matrix as a series of 4 lines showing the columns of the matrix.
        [plt.plot(range(1+i,5), x[i:], 's--', color=colors[j]) for i,x in enumerate(nz_all_overlap_somM)]
        plt.plot(range(1+i,5), x[i:], 's--', color=colors[j], label=label)
        plt.ylabel('N(z) Overlap')
        plt.legend(loc=0, fontsize=15)
        plt.xticks(np.arange(1,5,1))
    plt.savefig(outdir + 'nz_overlap_plot.png');
    plt.close()
    return

def true_vs_noisydeep_buzzard(true, noisy, columns, zp, outdir):
    fig, axarr = plt.subplots(3, 3, figsize=(16.,16.))
    fig.suptitle('True Buzzard Mag vs. Mock Noisy Deep Mag')
    for i, col in enumerate(columns):
        x = flux2mag(true[col].values, zp)
        y = flux2mag(noisy[:,i], zp)
        axarr[i // 3, i % 3].plot(x,y, '.')
        axarr[i // 3, i % 3].set_xlabel(col[8:])
    fig.tight_layout()
    print('save : ' + outdir + 'true_vs_noisydeep_buzzard.png')
    plt.savefig(outdir + 'true_vs_noisydeep_buzzard.png')
    
#MAKES BALROG PLOTS
def balrogplot(balrog_data, outdir, zp, deepcolname, widecolname, title=None): 
    fig, axarr = plt.subplots(1,2, figsize=(20.,6.))
    if title is not None:
        fig.suptitle(title, fontsize=20)
    hb=axarr[0].hexbin(flux2mag(balrog_data[deepcolname], zp), flux2mag(balrog_data[widecolname], zp), bins='log')
    axarr[0].set_xlabel(deepcolname.replace('flux','mag').replace('FLUX','MAG'))
    axarr[0].set_ylabel(widecolname.replace('flux','mag').replace('FLUX','MAG'))
    cb = fig.colorbar(hb, ax=axarr[0])
    axarr[0].set_xlim((18.5,26))
    axarr[0].set_ylim((18.5,23.5))

    hb=axarr[1].hexbin(flux2mag(balrog_data[deepcolname], zp), 
               flux2mag(balrog_data[widecolname], zp) - flux2mag(balrog_data[deepcolname], zp), bins='log')
    axarr[1].set_xlabel(deepcolname.replace('flux','mag').replace('FLUX','MAG'))
    axarr[1].set_ylabel('{} - {}'.format(widecolname.replace('flux','mag').replace('FLUX','MAG'),
                                         deepcolname.replace('flux','mag').replace('FLUX','MAG')))
    axarr[1].set_xlim((18.5,26))
    axarr[1].set_ylim((-6,2.5))
    cb = fig.colorbar(hb, ax=axarr[1])
    plt.savefig(outdir + 'balrogplot.png', dpi=100)


#MAKES DEEP SOM DIAGNOSTIC PLOTS - HISTOGRAMS OF THE SIGMAZ, MEANZ AND SIGMAZ/1+Z PER CELL
def deep_som_diagnostics(cm, outdir, cell_key, zp):
    deep_cell_std = cm.data.groupby(cell_key)['Z'].std()
    deep_cell_weights = cm.data.groupby(cell_key).size()
    deep_cell_std[np.isnan(cm.data.groupby(cell_key)['Z'].std())] = -1
    minus1=len((deep_cell_std[deep_cell_std==-1]))
    zero=len((deep_cell_std[deep_cell_std==0]))
    print("Number of deep cells with z out of {}: {}".format(cm.deep_som.w[0] * cm.deep_som.w[1], len(deep_cell_std)))
    print("Number of cells with std -1 and zero: ", minus1, zero)
    print("Number of cells with std>0: ", len(deep_cell_std[deep_cell_std>0]))
    deep_cell_mean = cm.data.groupby(cell_key)['Z'].mean()
    deep_cell_weights = cm.data.groupby(cell_key).size()
    deep_cell_std[np.isnan(cm.data.groupby(cell_key)['Z'].mean())] = -1

    #Histograms- weighted and not, of std z
    print(deep_cell_std.shape)
    fig, axarr = plt.subplots(2, 1, figsize=(16.,9.))
    axarr[0].hist(deep_cell_std, bins=100)
    axarr[0].set_ylabel(r'$N$')
    axarr[0].axvline(np.median(deep_cell_std[deep_cell_std>0]), color='k', linestyle='--',label="median %s" %np.median(deep_cell_std[deep_cell_std>0]))
    axarr[0].axvline(np.mean(deep_cell_std[deep_cell_std>0]), color='r', linestyle='--',label="mean %s" %np.mean(deep_cell_std[deep_cell_std>0]))
    axarr[0].axvline(x=-1, color='grey', linestyle='-',label="%s cells with 1 DF" %minus1)
    axarr[0].axvline(x=0, color='black', linestyle='-',label="%s cells with 1 z" %zero)
    axarr[0].legend()
    axarr[0].set_yscale('log')
    axarr[0].set_title('deep (-1 when 1 DF, 0 when 1 z) ff02')
    axarr[1].hist(deep_cell_std, weights=deep_cell_weights, bins=100);
    axarr[1].set_xlabel(r'$\sigma (p(z|c))$')
    plt.ylabel(r'$\sum p(c)$');
    axarr[1].axvline(np.median(deep_cell_std[deep_cell_std>0]), color='k', linestyle='--',label="median %s" %np.median(deep_cell_std))
    axarr[1].legend()
    axarr[1].set_yscale('log')
    plt.savefig(outdir + 'deep_cell_sigmaz.png', dpi=100)

    #Histograms- weighted and not, of mean z
    fig, axarr = plt.subplots(2, 1, figsize=(16.,9.))
    axarr[0].hist(deep_cell_mean, bins=100)
    axarr[0].set_ylabel(r'$N$')
    axarr[0].axvline(np.median(deep_cell_mean), color='k', linestyle='--',label="median %s" %np.median(deep_cell_mean))
    axarr[0].axvline(np.mean(deep_cell_mean), color='r', linestyle='--',label="mean %s" %np.mean(deep_cell_mean))
    axarr[0].legend()
    axarr[0].set_title('deep')
    axarr[1].hist(deep_cell_mean, weights=deep_cell_weights, bins=100);
    axarr[1].set_xlabel(r'$ <z> (p(z|c))$')
    #axarr[1].set_yscale('log')
    plt.ylabel(r'$\sum p(c)$');
    axarr[1].axvline(np.median(deep_cell_mean), color='k', linestyle='--',label="median %s" %np.median(deep_cell_mean))
    axarr[1].legend()
    plt.savefig(outdir + 'deep_cell_mean_pz.png', dpi=100)

    fig, ax = plt.subplots(1, 1, figsize=(16.,9.))
    ax.scatter(deep_cell_mean, deep_cell_std)
    ax.set_ylabel(r'$\sigma_z$')
    ax.set_xlabel(r'$\langle z \rangle$')
    ax.set_title('deep')
    plt.savefig(outdir + 'deep_cell_meanz_vs_sigmaz.png', dpi=100)
    
#MAKES WIDE SOM DIAGNOSTIC PLOTS - HISTOGRAMS OF THE SIGMAZ, MEANZ AND SIGMAZ/1+Z PER CELL, MEANZ AND SIGMAZ AS A FN OF MAG

def wide_som_diagnostics(cm, wide_data, outdir, cell_key_cm, cell_key_wide, flux_wide, zp):
    #Histograms- weighted and not, of sigma z
    wide_cell_std = cm.data.groupby(cell_key_cm)['Z'].std()
    wide_cell_weights = cm.data.groupby(cell_key_cm).size()
    wide_cell_std[np.isnan(cm.data.groupby(cell_key_cm)['Z'].std())] = -1

    fig, axarr = plt.subplots(2, 1, figsize=(16.,9.))
    axarr[0].hist(wide_cell_std, bins=100)
    axarr[0].set_ylabel(r'$N$')
    axarr[0].axvline(np.median(wide_cell_std), color='k', linestyle='--',label="median %s" %np.median(wide_cell_std))
    axarr[0].axvline(np.mean(wide_cell_std), color='r', linestyle='--',label="mean %s" %np.mean(wide_cell_std))
    axarr[0].legend()
    axarr[0].set_title('Wide')
    axarr[1].hist(wide_cell_std, weights=wide_cell_weights, bins=100);
    axarr[1].set_xlabel(r'$\sigma (p(z|\hat{c}))$')
    plt.ylabel(r'$\sum p(\hat{c})$');
    axarr[1].axvline(np.median(wide_cell_std), color='k', linestyle='--',label="median %s" %np.median(wide_cell_std))
    axarr[1].legend()
    plt.savefig(outdir + 'wide_cell_sigma_pz.png', dpi=100)

    #Histograms- weighted and not, of mean z
    wide_cell_mean = cm.data.groupby(cell_key_cm)['Z'].mean()
    wide_cell_weights = cm.data.groupby(cell_key_cm).size()
    wide_cell_std[np.isnan(cm.data.groupby(cell_key_cm)['Z'].mean())] = -1

    fig, axarr = plt.subplots(2, 1, figsize=(16.,9.))
    axarr[0].hist(wide_cell_mean, bins=100)
    axarr[0].set_ylabel(r'$N$')
    axarr[0].axvline(np.median(wide_cell_mean), color='k', linestyle='--',label="median %s" %np.median(wide_cell_mean))
    axarr[0].axvline(np.mean(wide_cell_mean), color='r', linestyle='--',label="mean %s" %np.mean(wide_cell_mean))
    axarr[0].legend()
    axarr[0].set_title('Wide')
    axarr[1].hist(wide_cell_mean, weights=wide_cell_weights, bins=100);
    axarr[1].set_xlabel(r'$ <z> (p(z|\hat{c}))$')
    plt.ylabel(r'$\sum p(\hat{c})$');
    axarr[1].axvline(np.median(wide_cell_mean), color='k', linestyle='--',label="median %s" %np.median(wide_cell_mean))
    axarr[1].legend()
    plt.savefig(outdir + 'wide_cell_mean_pz.png', dpi=100)

    #Histograms- weighted and not, of sigma/1+mean z
    wide_cell_ratio=wide_cell_std/(1+wide_cell_mean)
    fig, axarr = plt.subplots(2, 1, figsize=(16.,9.))
    axarr[0].hist(wide_cell_ratio, bins=100)
    axarr[0].set_ylabel(r'$N$')
    axarr[0].set_title('wide')
    axarr[0].axvline(np.median(wide_cell_ratio), color='k', linestyle='--',label="median %s" %np.median(wide_cell_ratio))
    axarr[0].axvline(np.mean(wide_cell_ratio), color='r', linestyle='--',label="mean %s" %np.mean(wide_cell_ratio))
    axarr[0].legend()
    axarr[0].set_yscale('log')
    # weight each cell by number of wl sample that are in that cell. (using matched balrog catalog)
    axarr[1].hist(wide_cell_ratio, weights=wide_cell_weights, bins=100);
    axarr[1].set_ylabel('Weighted N')
    axarr[1].set_xlabel(r'$\sigma / (1 + <z>) (p(z|c))$')
    axarr[1].axvline(np.median(wide_cell_ratio), color='k', linestyle='--',label="median %s" %np.median(wide_cell_ratio))
    axarr[1].legend()
    axarr[1].set_yscale('log')
    plt.savefig(outdir + 'wide_cell_sigzon1plusmean_pz.png', dpi=100)
    
    #Plot  mean and sigma z as a fn of mag
    all_cell_idx = np.arange(0, np.product(cm.wide_som.shape[:-1]))
    occupied_cells = np.unique(wide_data[cell_key_wide].values)
    unoccupied_cells = np.setdiff1d(all_cell_idx, occupied_cells)
    
    wide_cell_mean_mag_i = flux2mag(wide_data.groupby(cell_key_wide)[flux_wide].mean(), zp)

    for cell in unoccupied_cells:
        wide_cell_mean_mag_i.loc[cell] = -1

    wide_cell_mean_mag_i = np.array(wide_cell_mean_mag_i)

    '''
    wide_cell_mean_mag_i = flux2mag(wide_data.groupby(cell_key_wide)[flux_wide].mean(), zp)
    print(len(wide_cell_mean_mag_i))
    print(len(wide_cell_mean))
    if len(wide_cell_mean_mag_i) == len(wide_cell_mean):
        start_idx = 0
    elif len(wide_cell_mean_mag_i) == len(wide_cell_mean) + 1:
        start_idx = 1
    else:
        assert False
    '''
    fig, axarr = plt.subplots(2, 1, figsize=(16.,9.))
    axarr[0].scatter(wide_cell_mean_mag_i, wide_cell_mean) 
    axarr[0].axhline(1, color='k', linestyle='--')
    axarr[0].set_ylabel(r'<z>')
    axarr[0].set_title('wide')

    # weight each cell by number of wl sample that are in that cell. (using matched balrog catalog)
    axarr[1].scatter(wide_cell_mean_mag_i, wide_cell_std)
    axarr[1].axhline(1, color='k', linestyle='--')
    axarr[1].set_xlabel('magnitude')
    axarr[1].set_ylabel(r'$\sigma_z$')
    axarr[1].set_xlim((15,25))
    plt.savefig(outdir + 'wide_cell_mag_v_meansigmaz.png', dpi=100)

#WIDE SOM PLOTTING z, sigma_z, N

def plot_som_z(cm, outdir, cell_key, flux_key):
    n=np.array(cm.data.groupby(cell_key)[flux_key].count())
    print("total number of redshift objs: ", sum(n))
    print(len(n))
    print(len(np.unique(cm.data['ID'])), len(cm.data))

    fig, axarr = plt.subplots(1,3, figsize=(16.,16.))
    s=axarr[0].imshow(n.reshape(cm.wide_som.shape[:-1]), cmap='viridis_r')
    axarr[0].axis('off')
    divider = make_axes_locatable(axarr[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(s,ax=axarr[0],cax=cax)
    axarr[0].set_title('Num spec_data')

    zcells=np.array(cm.data.groupby(cell_key)['Z'].mean()) #.sort_values(ascending=False))
    s=axarr[1].imshow(zcells.reshape(cm.wide_som.shape[:-1]))
    axarr[1].axis('off')
    divider = make_axes_locatable(axarr[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(s,ax=axarr[1],cax=cax)
    axarr[1].set_title('<Z>')

    zcells=np.array(cm.data.groupby(cell_key)['Z'].std()) #.sort_values(ascending=False))
    s=axarr[2].imshow(zcells.reshape(cm.wide_som.shape[:-1]))
    axarr[2].axis('off')
    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(s,ax=axarr[2],cax=cax)
    axarr[2].set_title('sigma_Z')
    plt.savefig(outdir + 'plot_som_z.png')
    

#WIDE SOM PLOTTING I, R-I, Z-I REAL MAG VALUES
def plot_som_magcol(cm ,wide_data, outdir, cell_key, flux_cols, zp):
    all_cell_idx = np.arange(0, np.product(cm.wide_som.shape[:-1]))
    occupied_cells = np.unique(wide_data[cell_key].values)
    unoccupied_cells = np.setdiff1d(all_cell_idx, occupied_cells)
    
    wide_cell_mean_mag_i = flux2mag(wide_data.groupby(cell_key)[flux_cols[0]].mean(), zp)
    wide_cell_mean_mag_r = flux2mag(wide_data.groupby(cell_key)[flux_cols[1]].mean(), zp)
    wide_cell_mean_mag_z = flux2mag(wide_data.groupby(cell_key)[flux_cols[2]].mean(), zp)

    for cell in unoccupied_cells:
        wide_cell_mean_mag_i.loc[cell] = -1
        wide_cell_mean_mag_r.loc[cell] = -1
        wide_cell_mean_mag_z.loc[cell] = -1

    wide_cell_mean_mag_i = np.array(wide_cell_mean_mag_i)
    wide_cell_mean_mag_r = np.array(wide_cell_mean_mag_r)
    wide_cell_mean_mag_z = np.array(wide_cell_mean_mag_z)
    
    ri=wide_cell_mean_mag_r-wide_cell_mean_mag_i
    zi=wide_cell_mean_mag_z-wide_cell_mean_mag_i
    '''
    ncells_in_df = len(wide_cell_mean_mag_i)
    ncells_in_som = cm.wide_som.shape[0] * cm.wide_som.shape[1]
    if ncells_in_df == ncells_in_som:
        start_idx = 0
    elif ncells_in_df + 1 == ncells_in_som:
        start_idx = 1
    else:
        assert False
    '''
    fig, axarr = plt.subplots(1, cm.wide_som.w.shape[-1], figsize=(16.,9.))
    titles = ['i', 'r-i', 'z-i']

    s=axarr[0].imshow(wide_cell_mean_mag_i.reshape(cm.wide_som.shape[:-1]))
    axarr[0].axis('off')
    divider = make_axes_locatable(axarr[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(s,ax=axarr[0],cax=cax)
    axarr[0].set_title('<mag-i>')

    s=axarr[1].imshow(ri.reshape(cm.wide_som.shape[:-1]))
    axarr[1].axis('off')
    divider = make_axes_locatable(axarr[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(s,ax=axarr[1],cax=cax)
    axarr[1].set_title('<r-i>')

    s=axarr[2].imshow(zi.reshape(cm.wide_som.shape[:-1]))
    axarr[2].axis('off')
    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(s,ax=axarr[2],cax=cax)
    axarr[2].set_title('<z-i>')
    plt.savefig(outdir + 'plot_som_magcol.png')


#HISTOGRAMS OF MAGS IN DEEP V METACAL 

def compare_data():
    
    fig, ax = plt.subplots()
    plt.hist(flux2mag(balrog_data['unsheared/flux_i']),range=(15,30),bins=100,histtype='step', label='balrog_data %s' %len(balrog_data['unsheared/flux_i']),normed=True)
    plt.hist(flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),range=(15,30),bins=100,histtype='step', label='wide_metacal cuts%s' %len(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),normed=True)
    plt.xlabel('i band mag')
    plt.legend()
    ##plt.show()
    plt.savefig(outdir + 'wide_balrog_deep_mag.png', dpi=100)


    fig, ax = plt.subplots()
    plt.hist(balrog_data['BDF_MAG_DERED_CALIB_I'],range=(15,30),bins=100,histtype='step', label='deep_data %s' %len(deep_data['BDF_MAG_DERED_CALIB_I']))
    plt.hist(flux2mag(balrog_metacal['unsheared/flux_i']),range=(15,30),bins=100,histtype='step', label='balrog_mcal %s' %len(balrog_metacal['unsheared/flux_i']))
    plt.hist(flux2mag(balrog_data['unsheared/flux_i']),range=(15,30),bins=100,histtype='step', label='balrog_data %s' %len(balrog_data['unsheared/flux_i']))
    plt.hist(flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),range=(15,30),bins=100,histtype='step', label='wide_metacal cuts%s' %len(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]))
    plt.xlabel('i band mag')
    plt.yscale('log')
    plt.legend()
    ##plt.show()
    plt.savefig(outdir + 'wide_balrog_deep_mag_nonorm.png', dpi=100)


    fig, ax = plt.subplots(1,3, figsize=(20.,9.))
    #plt.hist(snr_deep,range=(0,1000),bins=50,histtype='step', label='deep_data')
    ax[0].hist(flux2mag(balrog_data['unsheared/flux_r'])-flux2mag(balrog_data['unsheared/flux_i']),range=(-10,10),bins=50,histtype='step', label='balrog_data')
    #plt.hist(flux2mag(wide_data['unsheared/flux_r'])-flux2mag(wide_data['unsheared/flux_i']) ,range=(-10,10),bins=50,histtype='step', label='wide_metacal')
    ax[0].hist(flux2mag(wide_data['unsheared/flux_r'][wide_data['PASS_WL_unsheared']==1])-flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]) ,range=(-10,10),bins=50,histtype='step', label='wide_metacal cuts ')
    ax[0].hist(balrog_data['BDF_MAG_DERED_CALIB_R']-balrog_data['BDF_MAG_DERED_CALIB_I'],range=(-10,10),bins=50,histtype='step', label='deep')
    ax[0].set_xlabel('r-i')
    ax[0].set_yscale('log')
    ax[0].legend()

    #plt.hist(snr_deep,range=(0,1000),bins=50,histtype='step', label='deep_data')
    ax[1].hist(flux2mag(balrog_data['unsheared/flux_z'])-flux2mag(balrog_data['unsheared/flux_i']),bins=50,range=(-10,10),histtype='step', label='balrog_data')
    #plt.hist(flux2mag(wide_data['unsheared/flux_z'])-flux2mag(wide_data['unsheared/flux_i']) ,bins=50,range=(-10,10),histtype='step', label='wide_metacal')
    ax[1].hist(flux2mag(wide_data['unsheared/flux_z'][wide_data['PASS_WL_unsheared']==1])-flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]) ,bins=50,range=(-10,10),histtype='step', label='wide_metacal cuts')
    ax[1].hist(balrog_data['BDF_MAG_DERED_CALIB_Z']-balrog_data['BDF_MAG_DERED_CALIB_I'],bins=50,range=(-10,10),histtype='step', label='deep')
    ax[1].set_xlabel('z-i')
    ax[1].set_yscale('log')
    ax[1].legend()

    #plt.hist(snr_deep,range=(0,1000),bins=50,histtype='step', label='deep_data')
    ax[2].hist(flux2mag(balrog_data['unsheared/flux_r'])-flux2mag(balrog_data['unsheared/flux_z']),bins=50,range=(-10,10),histtype='step', label='balrog_data')
    #plt.hist(flux2mag(wide_data['unsheared/flux_r'])-flux2mag(wide_data['unsheared/flux_z']) ,bins=50,range=(-10,10),histtype='step', label='wide_metacal')
    ax[2].hist(flux2mag(wide_data['unsheared/flux_r'][wide_data['PASS_WL_unsheared']==1])-flux2mag(wide_data['unsheared/flux_z'][wide_data['PASS_WL_unsheared']==1]) ,bins=50,range=(-10,10),histtype='step', label='wide_metacal cuts')
    ax[2].hist(balrog_data['BDF_MAG_DERED_CALIB_R']-balrog_data['BDF_MAG_DERED_CALIB_Z'],bins=50,range=(-10,10),histtype='step', label='deep')
    ax[2].set_xlabel('r-z')
    ax[2].set_yscale('log')
    ax[2].legend()
    plt.savefig(outdir + 'ri_zi_rz_hists.png', dpi=100)
    ##plt.show()

    plt.figure()
    fig, axarr = plt.subplots(1,3, figsize=(20.,9.))
    axarr[0].hexbin(deep_data['BDF_MAG_DERED_CALIB_I'],deep_data['BDF_MAG_DERED_CALIB_R']-deep_data['BDF_MAG_DERED_CALIB_I'],bins='log')
    #axarr[0].set_xlim(10,40)
    #axarr[0].set_ylim(42,44)
    axarr[0].set_title('deep')
    axarr[0].set_xlabel('BDF_MAG_DERED_CALIB_I')
    axarr[0].set_ylabel('BDF_MAG_DERED_CALIB_R-BDF_MAG_DERED_CALIB_I')
    axarr[1].hexbin(flux2mag(balrog_data['unsheared/flux_i']),flux2mag(balrog_data['unsheared/flux_r'])-flux2mag(balrog_data['unsheared/flux_i']),bins='log')
    axarr[1].set_xlim(18,24)
    #axarr[1].set_ylim(10,27)
    axarr[1].set_title('balrog_data')
    axarr[1].set_xlabel('unsheared/mag_i')
    axarr[1].set_ylabel('unsheared/mag_r-unsheared/mag_i')
    axarr[2].hexbin(flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),flux2mag(wide_data['unsheared/flux_r'][wide_data['PASS_WL_unsheared']==1])-flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),bins='log')
    axarr[2].set_xlim(18,24)
    #axarr[2].set_ylim(5,27)
    axarr[2].set_title('wide')
    axarr[2].set_xlabel('unsheared/mag_i')
    axarr[2].set_ylabel('unsheared/mag_r- unsheared/mag_i')
    #axarr[1].set_title('Laigle et al. COSMOS Data')
    plt.savefig(outdir + 'ri_i_hexbin.png', dpi=100)

    plt.figure()
    fig, axarr = plt.subplots(1,3, figsize=(20.,9.))
    axarr[0].hexbin(deep_data['BDF_MAG_DERED_CALIB_I'],deep_data['BDF_MAG_DERED_CALIB_Z']-deep_data['BDF_MAG_DERED_CALIB_I'],bins='log')
    axarr[0].set_xlim(10,40)
    axarr[0].set_ylim(-4,2)
    axarr[0].set_title('deep')
    axarr[0].set_xlabel('BDF_MAG_DERED_CALIB_I')
    axarr[0].set_ylabel('BDF_MAG_DERED_CALIB_Z-BDF_MAG_DERED_CALIB_I')
    axarr[1].hexbin(flux2mag(balrog_data['unsheared/flux_i']),flux2mag(balrog_data['unsheared/flux_z'])-flux2mag(balrog_data['unsheared/flux_i']),bins='log')
    axarr[1].set_xlim(18,24)
    axarr[1].set_ylim(-4,2)
    axarr[1].set_title('balrog_data')
    axarr[1].set_xlabel('unsheared/mag_i')
    axarr[1].set_ylabel('unsheared/mag_z-unsheared/mag_i')
    axarr[2].hexbin(flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),flux2mag(wide_data['unsheared/flux_z'][wide_data['PASS_WL_unsheared']==1])-flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),bins='log')
    axarr[2].set_xlim(18,24)
    axarr[2].set_ylim(-4,2)
    axarr[2].set_title('wide')
    axarr[2].set_xlabel('unsheared/mag_i')
    axarr[2].set_ylabel('unsheared/mag_z- unsheared/mag_i')
    #axarr[1].set_title('Laigle et al. COSMOS Data')
    plt.savefig(outdir + 'zi_i_hexbin.png', dpi=100)

    plt.figure()
    fig, axarr = plt.subplots(1,3, figsize=(20.,9.))
    axarr[0].hexbin(deep_data['BDF_MAG_DERED_CALIB_R']-deep_data['BDF_MAG_DERED_CALIB_Z'],deep_data['BDF_MAG_DERED_CALIB_R']-deep_data['BDF_MAG_DERED_CALIB_I'],bins='log')
    #axarr[0].set_xlim(15,30)
    axarr[0].set_ylim(-1.5,4)
    axarr[0].set_title('deep')
    axarr[0].set_xlabel('BDF_MAG_DERED_CALIB_R-BDF_MAG_DERED_CALIB_Z')
    axarr[0].set_ylabel('BDF_MAG_DERED_CALIB_R-BDF_MAG_DERED_CALIB_I')
    axarr[1].hexbin(flux2mag(balrog_data['unsheared/flux_r'])-flux2mag(balrog_data['unsheared/flux_z']),flux2mag(balrog_data['unsheared/flux_r'])-flux2mag(balrog_data['unsheared/flux_i']),bins='log')
    #axarr[1].set_xlim(10,28)
    axarr[1].set_ylim(-1.5,4)
    axarr[1].set_title('balrog_data')
    axarr[1].set_xlabel('unsheared/mag_r-unsheared/mag_z')
    axarr[1].set_ylabel('unsheared/mag_r-unsheared/mag_i')
    axarr[2].hexbin(flux2mag(wide_data['unsheared/flux_r'][wide_data['PASS_WL_unsheared']==1])-flux2mag(wide_data['unsheared/flux_z'][wide_data['PASS_WL_unsheared']==1]),flux2mag(wide_data['unsheared/flux_r'][wide_data['PASS_WL_unsheared']==1])-flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),bins='log')
    #axarr[2].set_xlim(10,27)
    axarr[2].set_ylim(-1.5,4)
    axarr[2].set_title('wide')
    axarr[2].set_xlabel('unsheared/mag_r-unsheared/mag_z')
    axarr[2].set_ylabel('unsheared/mag_r- unsheared/mag_i')
    #axarr[1].set_title('Laigle et al. COSMOS Data')
    plt.savefig(outdir + 'ri_rz_hexbin.png', dpi=100)

    plt.figure()
    fig, axarr = plt.subplots(1,3, figsize=(20.,9.))
    axarr[0].hexbin(deep_data['BDF_MAG_DERED_CALIB_I'],flux2mag(deep_data['BDF_FLUX_ERR_DERED_CALIB_I']),bins='log')
    #axarr[0].set_xlim(15,30)
    #axarr[0].set_ylim(42,44)
    axarr[0].set_title('deep')
    axarr[0].set_xlabel('BDF_MAG_DERED_CALIB_I')
    axarr[0].set_ylabel('mag err')
    axarr[1].hexbin(flux2mag(balrog_data['unsheared/flux_i']),flux2mag(balrog_data['unsheared/flux_err_i']),bins='log')
    axarr[1].set_xlim(18,24)
    axarr[1].set_ylim(10,27)
    axarr[1].set_title('balrog_data')
    axarr[1].set_xlabel('unsheared/mag_i')
    axarr[1].set_ylabel('unsheared/mag_err_i')
    axarr[2].hexbin(flux2mag(wide_data['unsheared/flux_i'][wide_data['PASS_WL_unsheared']==1]),flux2mag(wide_data['unsheared/flux_err_i'][wide_data['PASS_WL_unsheared']==1]),bins='log')
    axarr[2].set_xlim(18,24)
    axarr[2].set_ylim(10,27)
    axarr[2].set_title('wide')
    axarr[2].set_xlabel('unsheared/mag_i')
    axarr[2].set_ylabel('unsheared/mag_err_i')
    #axarr[1].set_title('Laigle et al. COSMOS Data')
    plt.savefig(outdir + 'mag_magerrs_hexbin.png', dpi=100)

    
#plot balrog deep vs wide- transfer matrix
def plot_balrog_transfer(balrog_data, outdir):
    fig, ax = plt.subplots(1,2, figsize=(20.,6.))
    ax[0].hexbin(flux2mag(balrog_data['BDF_FLUX_DERED_CALIB_I']), flux2mag(balrog_data['unsheared/flux_i']), bins='log')
    ax[0].set_xlabel('BDF_FLUX_DERED_CALIB_I')
    ax[0].set_ylabel('unsheared/flux_i')
    #plt.colorbar()

    ax[1].hexbin(flux2mag(balrog_data['BDF_FLUX_DERED_CALIB_I']), 
               flux2mag(balrog_data['unsheared/flux_i']) - flux2mag(balrog_data['BDF_FLUX_DERED_CALIB_I']), bins='log')
    ax[1].set_xlabel('BDF_FLUX_DERED_CALIB_I')
    ax[1].set_ylabel('unsheared/flux_i - BDF_FLUX_DERED_CALIB_I')
    plt.colorbar()
    plt.savefig(outdir + 'balrog_data_wide_vs_deep_diff.png', dpi=100)

#plot SOMs as weights
def plot_som_w(cm, outdir): 
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, axarr = plt.subplots(1, cm.deep_som.w.shape[-1], figsize=(16.,9.))
    for dim in range(cm.deep_som.w.shape[-1]):
        s=axarr[dim].imshow(cm.deep_som.w[:,dim].reshape(cm.deep_som.shape[:-1]))
        axarr[dim].axis('off')
        divider = make_axes_locatable(axarr[dim])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(s,ax=axarr[dim],cax=cax)
    plt.savefig(outdir + '/cellmap_deep_w.png', dpi=100)

    fig, axarr = plt.subplots(1, cm.wide_som.w.shape[-1], figsize=(16.,9.))
    titles = ['i', 'r-i', 'z-i']
    # TODO add meaningful colorbars (represent magnitude/colors)
    for dim in range(cm.wide_som.w.shape[-1]):
        s=axarr[dim].imshow(cm.wide_som.w[:,dim].reshape(cm.wide_som.shape[:-1]))
        axarr[dim].axis('off')
        axarr[dim].set_title(titles[dim])
        divider = make_axes_locatable(axarr[dim])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(s,ax=axarr[dim],cax=cax)
    plt.savefig(outdir + '/cellmap_wide_w.png', dpi=100)
    
def deep_mags_byfield(deep_data, outdir):
    plt.figure()
    fig, ax = plt.subplots(1,3, figsize=(20.,6.))
    ax[0].hist(deep_data[deep_data['FIELD']=='COSMOS']['BDF_MAG_DERED_CALIB_I'],100,(16,38),color='red',label='COSMOS %s' %(len(deep_data[deep_data['FIELD']=='COSMOS'])),histtype='step')
    ax[0].hist(deep_data[deep_data['FIELD']=='C3']['BDF_MAG_DERED_CALIB_I'],100,(16,38),color='green',label='C3 %s'%(len(deep_data[deep_data['FIELD']=='C3'])) ,histtype='step')
    ax[0].hist(deep_data[deep_data['FIELD']=='X3']['BDF_MAG_DERED_CALIB_I'],100,(16,38),color='blue',label='X3 %s' %(len(deep_data[deep_data['FIELD']=='X3'])),histtype='step' )
    ax[0].hist(deep_data[deep_data['FIELD']=='E2']['BDF_MAG_DERED_CALIB_I'],100,(16,38),color='grey',label='E2 %s' %(len(deep_data[deep_data['FIELD']=='E2'])),histtype='step' )
    ax[0].set_xlabel('i')
    ax[0].legend()
    
    ax[1].hist(deep_data[deep_data['FIELD']=='COSMOS']['BDF_MAG_DERED_CALIB_G'],100,(16,38),color='red',label='COSMOS %s' %(len(deep_data[deep_data['FIELD']=='COSMOS'])),histtype='step' )
    ax[1].hist(deep_data[deep_data['FIELD']=='C3']['BDF_MAG_DERED_CALIB_G'],100,(16,38),color='green',label='C3 %s'%(len(deep_data[deep_data['FIELD']=='C3'])) ,histtype='step')
    ax[1].hist(deep_data[deep_data['FIELD']=='X3']['BDF_MAG_DERED_CALIB_G'],100,(16,38),color='blue',label='X3 %s' %(len(deep_data[deep_data['FIELD']=='X3'])) ,histtype='step')
    ax[1].hist(deep_data[deep_data['FIELD']=='E2']['BDF_MAG_DERED_CALIB_G'],100,(16,38),color='grey',label='E2 %s' %(len(deep_data[deep_data['FIELD']=='E2'])),histtype='step')
    ax[1].set_xlabel('g')
    ax[1].legend()
   
    ax[2].hist(deep_data[deep_data['FIELD']=='COSMOS']['BDF_MAG_DERED_CALIB_J'],100,range=(0,40),color='red',label='COSMOS %s' %(len(deep_data[deep_data['FIELD']=='COSMOS'])),histtype='step')
    ax[2].hist(deep_data[deep_data['FIELD']=='C3']['BDF_MAG_DERED_CALIB_J'],100,range=(0,40),color='green',label='C3 %s' %(len(deep_data[deep_data['FIELD']=='C3'])),histtype='step')
    ax[2].hist(deep_data[deep_data['FIELD']=='X3']['BDF_MAG_DERED_CALIB_J'],100,range=(0,40),color='blue',label='X3 %s' %(len(deep_data[deep_data['FIELD']=='X3'])),histtype='step' )
    ax[2].hist(deep_data[deep_data['FIELD']=='E2']['BDF_MAG_DERED_CALIB_J'],100,range=(0,40),color='grey',label='E2 %s'%(len(deep_data[deep_data['FIELD']=='E2'])),histtype='step' )
    ax[2].set_xlabel('J')
    ax[2].legend()
    plt.savefig(outdir + 'hist_deep_mags.png', dpi=100)
    
def deep_mag_icolour(deep_data, outdir, label):
    fig, ax = plt.subplots(1,2, figsize=(12.,4.))
    #plt.hist(snr_deep,range=(0,1000),ins=50,histtype='step', label='deep_data')
    ax[0].hist(deep_data['BDF_MAG_DERED_CALIB_U'],bins=120,range=(10,40),histtype='step', label='u')#,normed=True)
    ax[0].hist(deep_data['BDF_MAG_DERED_CALIB_G'],bins=120,range=(10,40),histtype='step', label='g')#,normed=True)
    ax[0].hist(deep_data['BDF_MAG_DERED_CALIB_R'],bins=120,range=(10,40),histtype='step', label='r')#,normed=True)
    ax[0].hist(deep_data['BDF_MAG_DERED_CALIB_I'],bins=120,range=(10,40),histtype='step', label='i')#,normed=True)
    ax[0].hist(deep_data['BDF_MAG_DERED_CALIB_Z'],bins=120,range=(10,40),histtype='step', label='z')#,normed=True)
    ax[0].hist(deep_data['BDF_MAG_DERED_CALIB_J'],bins=120,range=(10,40),histtype='step', label='j')#,normed=True)
    ax[0].hist(deep_data['BDF_MAG_DERED_CALIB_H'],bins=120,range=(10,40),histtype='step', label='h')#,normed=True)
    ax[0].hist(deep_data['BDF_MAG_DERED_CALIB_K'],bins=120,range=(10,40),histtype='step', label='k')#,normed=True)
    ax[0].set_xlabel('mag %s'%label)
    ax[0].set_yscale('log')
    ax[0].legend(loc='upper left')

    #fig, ax = plt.subplots()
    #plt.hist(snr_deep,range=(0,1000),ins=50,histtype='step', label='deep_data')
    ax[1].hist(deep_data['BDF_MAG_DERED_CALIB_U']-deep_data['BDF_MAG_DERED_CALIB_I'],bins=120,range=(-15,25),histtype='step', label='u-i')#,normed=True)
    ax[1].hist(deep_data['BDF_MAG_DERED_CALIB_G']-deep_data['BDF_MAG_DERED_CALIB_I'],bins=120,range=(-15,25),histtype='step', label='g-i')#,normed=True)
    ax[1].hist(deep_data['BDF_MAG_DERED_CALIB_R']-deep_data['BDF_MAG_DERED_CALIB_I'],bins=120,range=(-15,25),histtype='step', label='r-i')#,normed=True)
    ax[1].hist(deep_data['BDF_MAG_DERED_CALIB_Z']-deep_data['BDF_MAG_DERED_CALIB_I'],bins=120,range=(-15,25),histtype='step', label='z-i')#,normed=True)
    ax[1].hist(deep_data['BDF_MAG_DERED_CALIB_J']-deep_data['BDF_MAG_DERED_CALIB_I'],bins=120,range=(-15,25),histtype='step', label='j-i')#,normed=True)
    ax[1].hist(deep_data['BDF_MAG_DERED_CALIB_H']-deep_data['BDF_MAG_DERED_CALIB_I'],bins=120,range=(-15,25),histtype='step', label='h-i')#,normed=True)
    ax[1].hist(deep_data['BDF_MAG_DERED_CALIB_K']-deep_data['BDF_MAG_DERED_CALIB_I'],bins=120,range=(-15,25),histtype='step', label='k-i')#,normed=True)
    ax[1].set_xlabel('colour %s'%label)
    ax[1].set_yscale('log')
    plt.legend()
    name=outdir + '%s_postcuts_mag_color_hist.png'%(label)
    plt.savefig(name, dpi=300)  


# one plot to rule them
def plot_som(df, w, assign_names, map_shape, flux_keys, ivar_keys=[], path=''):
    '''Plot Properties of a SOM

    Parameters
    ----------
    df : pandas dataframe containing keys in assign_names, flux_keys, and possibly ivar_keys
    w : SOM weights
    assign_names : the two columns by which a row is assigned to a SOM
    map_shape : The shape of the SOM
    flux_keys : keys that correspond to the fluxes for the SOM
    ivar_keys : used in evaluating the chi2; refers to the diagonal inverse variance
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object

    Notes
    -----
    3 cols, n_dim + 1 rows

    row 1: logN assignment, umatrix, average chi2 of match

    for rest of rows, each dim of assignment
    col 1: weight vector value
    col 2: mean value of assigned dim
    col 3: std of assigned dim
    '''
    n_bmu0, n_bmu1, n_dim = w.shape

    ncols = 3
    nrows = n_dim + 1
    cell_size = 4
    figsize_x = cell_size * ncols * 1.3
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(figsize_x, figsize_y))

    # row 1: logN assignment, umatrix, average chi2 of cell
    grp = df.groupby(assign_names)
    agg = grp.agg(['size', 'mean', 'std'])
    C = agg_matrix(agg[flux_keys[0]]['size'], np.log10(agg[flux_keys[0]]['size'].values), map_shape=map_shape)
    ax = axs[0, 0]
    ax.set_xlabel(assign_names[0])
    ax.set_ylabel(assign_names[1])
    ax.set_title('log10 N')
    im = ax.pcolor(C)
    fig.colorbar(im, ax=ax)

    u = umatrix(w)
    ax = axs[0, 1]
    ax.set_xlabel(assign_names[0])
    ax.set_ylabel(assign_names[1])
    ax.set_title('umatrix')
    im = ax.pcolor(u)
    fig.colorbar(im, ax=ax)

    fluxes = df[flux_keys].values
    if len(ivar_keys) == len(flux_keys):
        ivar = df[ivar_keys].values
    else:
        ivar = np.ones_like(fluxes)
    chi2 = evaluate_som(fluxes, ivar, w)
    ax = axs[0, 2]
    ax.set_xlabel(assign_names[0])
    ax.set_ylabel(assign_names[1])
    ax.set_title('Average chi2')
    im = ax.pcolor(np.mean(chi2, axis=0))
    fig.colorbar(im, ax=ax)

    for i, flux_key in enumerate(flux_keys):
        # weight vector
        ax = axs[i + 1, 0]
        ax.set_xlabel(assign_names[0])
        ax.set_ylabel(assign_names[1])
        ax.set_title('Weight ' + flux_key)
        im = ax.pcolor(w[:, :, i])
        fig.colorbar(im, ax=ax)

        # mean
        C = agg_matrix(agg[flux_key]['mean'], agg[flux_key]['mean'].values, map_shape=map_shape)
        ax = axs[i + 1, 1]
        ax.set_xlabel(assign_names[0])
        ax.set_ylabel(assign_names[1])
        ax.set_title('Average ' + flux_key)
        im = ax.pcolor(C)
        fig.colorbar(im, ax=ax)

        # std
        C = agg_matrix(agg[flux_key]['std'], agg[flux_key]['std'].values, map_shape=map_shape)
        ax = axs[i + 1, 2]
        ax.set_xlabel(assign_names[0])
        ax.set_ylabel(assign_names[1])
        ax.set_title('STD ' + flux_key)
        im = ax.pcolor(C)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig

def plot_key(dfs, key, assign_names, map_shape, labels=[], log_scale=False, histogram_kwargs={'bins': 'auto', 'density': True}, plot_histogram_kwargs={'linewidth': 2, 'linestyle': '-'}, path='', **kwargs):
    '''Plots 1d histogram of key and then also SOM plots of the key for each df

    Parameters
    ----------
    dfs : a list of pandas dataframes each containing assign_names and key
    assign_names : the two columns by which a row is assigned to a SOM
    map_shape : The shape of the SOM
    labels : if given, will be the labels used in the plots
    log_scale : TODO: seems to be depreciated, and does not do anything?
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object
    '''
    # reduce df to keys we care about: assign_names and key
    dfs = [df[assign_names + [key]] for df in dfs]
    if len(labels) != len(dfs):
        if len(dfs) == 1:
            labels = ['Full Sample']
        else:
            labels = ['Sample {0}'.format(i + 1) for i in range(len(dfs))]
    # 3 col, len(dfs) + 1 rows.
    # row 1: 1d hist of key
    # further rows: [logN occupation, average in cell, std in cell]
    nrows = len(dfs) + 1
    ncols = 3
    cell_size = 4
    figsize_x = cell_size * ncols * 1.3
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max
    fig = plt.figure(figsize=(figsize_x, figsize_y))  # slight adjust for the axes
    gs = gridspec.GridSpec(nrows, ncols)

    # 1d hist of key based on each sample
    ax_1dhist = plt.subplot(gs[0, :])
    xs = []
    ys = []
    for i, df in enumerate(dfs):
        z = df[key].values
        # cut to be finite
        conds = np.isfinite(z)
        if np.sum(conds) != len(conds):
            z = z[conds]
            print('Warning! Cutting Sample {0} with key {1} from {2} to {3} due to non-finites!'.format(i, key, len(conds), np.sum(conds)))
        yi, edges = np.histogram(z, **histogram_kwargs)
        if i == 0:
            histogram_kwargs = copy.deepcopy(histogram_kwargs)
            histogram_kwargs['bins'] = edges  # so that we bin the same data
        x, y = histogramize(edges, yi)
        xs.append(x)
        ys.append(y)
    for x, y, label in zip(xs, ys, labels):
        ax_1dhist.plot(x, y, label=label, **plot_histogram_kwargs)
    ax_1dhist.legend()
    # try:
    #     ax_1dhist.legend()
    # # catch all labels == None
    # except:

    # for each df, assign galaxies to cell
    sizes = []
    means = []
    stds = []
    for df in dfs:
        # groupby
        grp = df.groupby(assign_names)
        # agg for size, mean, std.
        agg = grp.agg(['size', 'mean', 'std'])
        agg_size = agg[key]['size']
        agg_mean = agg[key]['mean']
        agg_std = agg[key]['std']

        # agging by size removes column names
        sizes.append(agg_matrix(agg_size, np.log10(agg_size.values), map_shape=map_shape))
        means.append(agg_matrix(agg_mean, agg_mean.values, map_shape=map_shape))
        stds.append(agg_matrix(agg_std, agg_std.values, map_shape=map_shape))
    # get scales for consistency
    vmin_size = np.nanpercentile(sizes, 2)
    vmax_size = np.nanpercentile(sizes, 98)
    vmin_mean = np.nanpercentile(means, 2)
    vmax_mean = np.nanpercentile(means, 98)
    vmin_std = np.nanpercentile(stds, 2)
    vmax_std = np.nanpercentile(stds, 98)

    # plot!
    for i, size, mean, std, label in zip(range(len(sizes)), sizes, means, stds, labels):
        for j, C, jlabel, vmin, vmax in zip(range(3), [size, mean, std], ['log10 N', 'mean', 'std'], [vmin_size, vmin_mean, vmin_std], [vmax_size, vmax_mean, vmax_std]):
            ax = plt.subplot(gs[i + 1, j])
            ax.set_xlabel(assign_names[0])
            ax.set_ylabel(assign_names[1])
            ax.set_title('{0} {1}: {2}'.format(label, key, jlabel))
            im = ax.pcolor(C, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig

def plot_sample(z, wide_c, deep_c, wide_som, deep_som, zkey='', log_scale=False, path='', **kwargs):
    '''Plots 1d histogram of z and how it matches up with the wide and deep SOM

    Parameters
    ----------
    z : array of values
    deep_c, wide_c : the cell coordinates of each row for z in deep and wide
    deep_som, wide_som : SOM objects
    zkey : the key associated with z
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object
    '''
    wide_cND = wide_som.cell1d_to_cell(wide_c)
    deep_cND = deep_som.cell1d_to_cell(deep_c)
    df = pd.DataFrame({'z': z, 'wide_0': wide_cND[0], 'wide_1': wide_cND[1],
                               'deep_0': deep_cND[0], 'deep_1': deep_cND[1]})
    # 3 col, 3 rows
    # row 1: 1d hist of key
    # further rows: [logN occupation, average in cell, std in cell]
    nrows = 3
    ncols = 3
    cell_size = 4
    figsize_x = cell_size * ncols * 1.3
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max
    fig = plt.figure(figsize=(figsize_x, figsize_y))  # slight adjust for the axes
    gs = gridspec.GridSpec(nrows, ncols)
    # 1d hist
    ax_1dhist = plt.subplot(gs[0, :])
    yi, edges = np.histogram(z, bins='auto', density=True)
    x, y = histogramize(edges, yi)
    ax_1dhist.plot(x, y, linewidth=2)
    ax_1dhist.set_xlabel(zkey)

    # somspace plots
    for i, kind in enumerate(['wide', 'deep']):
        som = {'wide': wide_som, 'deep': deep_som}[kind]
        assign_keys = ['{0}_0'.format(kind), '{0}_1'.format(kind)]
        c0 = df[assign_keys[0]].values
        c1 = df[assign_keys[1]].values
        for j, agg_kind in enumerate(['log10 N', 'mean', 'std']):
            ax = plt.subplot(gs[i + 1, j])
            agg_func = {'log10 N': 'size', 'mean': 'mean', 'std': 'std'}[agg_kind]
            log_scale_j = {'size': True, 'mean': log_scale, 'std': log_scale}[agg_func]
            plot_key_somspace(z, c0, c1, som, fig, ax, agg_func, log_scale_j, **kwargs)

            ax.set_xlabel('{0}_0'.format(kind))
            ax.set_ylabel('{0}_1'.format(kind))
            if agg_kind == 'log10 N':
                ax.set_title(r'{0} : $\log_{{10}} N$'.format(kind))
            elif agg_kind == 'mean':
                ax.set_title(r'{0} : $\langle {1}\rangle$'.format(kind, zkey.lower()))
            else:
                ax.set_title(r'{0} : $\sigma({1})$'.format(kind, zkey.lower()))

    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig

def plot_sample_colors(mags, wide_c, deep_c, wide_som, deep_som, key='', log_scale=False, path='', **kwargs):
    """Plots 1d histogram of mags and how they match up with the wide and deep SOM

    Parameters
    ----------
    mags : array of values
    deep_c, wide_c : the cell coordinates of each row for mags in deep and wide
    deep_som, wide_som : SOM objects
    key : the key associated with mags
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object
    """
    wide_cND = wide_som.cell1d_to_cell(wide_c)
    deep_cND = deep_som.cell1d_to_cell(deep_c)
    df = pd.DataFrame({'wide_0': wide_cND[0], 'wide_1': wide_cND[1],
                       'deep_0': deep_cND[0], 'deep_1': deep_cND[1]})
    df = pd.concat([mags,df], axis=1)

    # 3 col, 3 rows
    # row 1: 1d hist of key
    # further rows: [logN occupation, average in cell, std in cell]
    nrows = 3
    ncols = 3
    cell_size = 4
    figsize_x = cell_size * ncols * 1.3
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max
    fig = plt.figure(figsize=(figsize_x, figsize_y))  # slight adjust for the axes
    gs = gridspec.GridSpec(nrows, ncols)
    # 1d hist
    ax_1dhist = plt.subplot(gs[0, :])
    print(type(mags))
    print(mags.shape)
    for col in mags:
        yi, edges = np.histogram(mags[col], bins='auto', density=True)
        x, y = histogramize(edges, yi)
        ax_1dhist.plot(x, y, linewidth=2, label=col)
        ax_1dhist.set_xlabel(col)
    ax_1dhist.legend()

    # somspace plots
    for i, kind in enumerate(['wide', 'deep']):
        som = {'wide': wide_som, 'deep': deep_som}[kind]
        assign_keys = ['{0}_0'.format(kind), '{0}_1'.format(kind)]
        c0 = df[assign_keys[0]].values
        c1 = df[assign_keys[1]].values

        ax = plt.subplot(gs[i + 1, 0])

        plot_key_somspace(mags['MAG_i'], c0, c1, som, fig, ax, 'median', log_scale, **kwargs)

        ax.set_xlabel('{0}_0'.format(kind))
        ax.set_ylabel('{0}_1'.format(kind))
        ax.set_title(r'{0} : $\langle {1}\rangle$'.format(kind, key[0].lower()))
        
        for j in range(1, len(key)):
            ax = plt.subplot(gs[i + 1, j])
            plotdata = (mags[key[j]] - mags[key[0]]).values
            plot_key_somspace(plotdata, c0, c1, som, fig, ax, 'median', log_scale, **kwargs)
            ax.set_title('{} - {}'.format(key[j],key[0]))
    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig

def plot_som_mean_z(z, wide_c, deep_c, wide_som, deep_som, zkey='', log_scale=False, path='', 
                    kind='deep', **kwargs):
    '''Plots mean z in each cell of SOM

    Parameters
    ----------
    z : array of values
    deep_c, wide_c : the cell coordinates of each row for z in deep and wide
    deep_som, wide_som : SOM objects
    zkey : the key associated with z
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object
    '''
    wide_cND = wide_som.cell1d_to_cell(wide_c)
    deep_cND = deep_som.cell1d_to_cell(deep_c)
    df = pd.DataFrame({'z': z, 'wide_0': wide_cND[0], 'wide_1': wide_cND[1],
                               'deep_0': deep_cND[0], 'deep_1': deep_cND[1]})

    nrows = 1
    ncols = 1
    cell_size = 4
    figsize_x = cell_size * ncols * 1.3
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max
    fig = plt.figure(figsize=(figsize_x, figsize_y))  # slight adjust for the axes
    gs = gridspec.GridSpec(nrows, ncols)

    ax = plt.subplot(gs[0, :])
    
    som = deep_som if kind == 'deep' else wide_som
    assign_keys = ['{0}_0'.format(kind), '{0}_1'.format(kind)]
    c0 = df[assign_keys[0]].values
    c1 = df[assign_keys[1]].values
    agg_func = 'mean'
    
    c1 = (c1 - 25) % som.shape[0]
    plot_key_somspace(z, c0, c1, som, 
                      fig, ax, agg_func, log_scale, 
                      vmin=None, vmax = None, **kwargs)

    ax.set_xlabel('{0}_0'.format(kind))
    ax.set_ylabel('{0}_1'.format(kind))
    ax.set_title(r'{0} : $\langle {1}\rangle$'.format(kind, zkey.lower()))

    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig

def plot_som_diagnostic(z, wide_c, deep_c, wide_som, deep_som,
                        zkey='', log_scale=False, path='', **kwargs):
    '''Plots mean z in each cell of SOM

    Parameters
    ----------
    z : array of values
    deep_c, wide_c : the cell coordinates of each row for z in deep and wide
    deep_som, wide_som : SOM objects
    zkey : the key associated with z
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object
    '''
    wide_cND = wide_som.cell1d_to_cell(wide_c)
    deep_cND = deep_som.cell1d_to_cell(deep_c)
    df = pd.DataFrame({'z': z, 'wide_0': wide_cND[0], 'wide_1': wide_cND[1],
                               'deep_0': deep_cND[0], 'deep_1': deep_cND[1]})

    nrows = 1
    ncols = 2
    cell_size = 4
    figsize_x = cell_size * ncols * 1.1
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max
    fig = plt.figure(figsize=(figsize_x, figsize_y))  # slight adjust for the axes
    gs = gridspec.GridSpec(nrows, ncols)
    
    for i, kind in enumerate(['wide', 'deep']):
        som = {'wide': wide_som, 'deep': deep_som}[kind]
        ax = plt.subplot(gs[0, i])
        assign_keys = ['{0}_0'.format(kind), '{0}_1'.format(kind)]
        c0 = df[assign_keys[0]].values
        c1 = df[assign_keys[1]].values
        agg_func = 'mean'
    
        #c1 = (c1 - 25) % som.shape[0]
        plot_key_somspace(z, c0, c1, som, 
                          fig, ax, agg_func, log_scale, 
                          vmin=None, vmax = None)
        
        ax.set_xlabel('{0}_0'.format(kind))
        ax.set_ylabel('{0}_1'.format(kind))
        ax.set_title(r'{0} : $\langle {1}\rangle$'.format(kind, zkey.lower()))

    if 'row' in kwargs:
        print(kwargs['row'])
    if 'col' in kwargs:
        print(kwargs['col'])
        
    # plot n(z) in cell
    #cm.histogram_wide(key='Z', cells=[group], cell_weights=[1])
    # plot all spec-z in cell
    # plot postage stamp
    # plot lupticolors
    # see everywhere in deep SOM the wide SOM cell gals are coming from

    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig

def plot_som_diagnostic_balrog_vs_wide(balrog_c, wide_c,
                                       balrog_data, wide_data,
                                       wide_som, log_scale=False,
                                       **kwargs):
    '''Plots normalized counts of samples in each cell of SOM

    Parameters
    ----------
    balrog_data, wide_data : the samples assigned to the SOM
    balrog_c, wide_c : the cell coordinates of each row for balrog and wide
    wide_som : SOM object
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object
    '''
    return None

    wide_cND = wide_som.cell1d_to_cell(wide_c)
    balrog_cND = wide_som.cell1d_to_cell(balrog_c)
    df_wide = pd.DataFrame({'wide_0': wide_cND[0], 'wide_1': wide_cND[1]})
    df_balrog = pd.DataFrame({'balrog_0': balrog_cND[0], 'balrog_1': balrog_cND[1]})

    nrows = 1
    ncols = 2
    cell_size = 4
    figsize_x = cell_size * ncols * 1.1
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max
    fig = plt.figure(figsize=(figsize_x, figsize_y))  # slight adjust for the axes
    gs = gridspec.GridSpec(nrows, ncols)
    
    for i, kind in enumerate(['wide', 'balrog']):
        df = {'wide' : df_wide, 'balrog' : df_balrog}[kind]
        ax = plt.subplot(gs[0, i])
        assign_keys = ['{0}_0'.format(kind), '{0}_1'.format(kind)]
        c0 = df[assign_keys[0]].values
        c1 = df[assign_keys[1]].values
        agg_func = 'count'
    
        #c1 = (c1 - 25) % som.shape[0]
        plot_key_somspace(None, c0, c1, wide_som, 
                          fig, ax, agg_func, log_scale, 
                          vmin=None, vmax = None)
        
        ax.set_xlabel('{0}_0'.format(kind))
        ax.set_ylabel('{0}_1'.format(kind))
        ax.set_title(r'{0} : $\sum {1}$'.format(kind, 'n'))

    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig

def plot_cell(z, other_c, other_som, zkey, data=None, log_scale=False, path='', 
              spec_z = None, c = None, som = None, pcchat = None, cell_kind = None, **kwargs):
    '''Take all of sample in certain som cell and plot 1d histogram and
    breakdown of sample in other som.

    Parameters
    ----------
    z : array or dataframe of values
    other_c : cell values in a SOM
    other_som : SOM object
    zkey : the key associated with z
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object
    '''
    other_cND = other_som.cell1d_to_cell(other_c)

    if c is not None and som is not None:
        row, col = som.cell1d_to_cell(c)
    else:
        row, col = None
        
    if type(z) is np.ndarray:
        df = pd.DataFrame({'z': z, 'other_c0': other_cND[0], 'other_c1': other_cND[1]})
    else:
        df = z
        df['other_c0'] = other_cND[0]
        df['other_c1'] = other_cND[1]
        zkey = 'Z'
        z = df[zkey]
        
    # 3 col, 2 rows
    # row 1: 1d hist of key
    # further rows: [logN occupation, average in cell, std in cell]
    nrows = 4
    ncols = 3
    cell_size = 4
    figsize_x = cell_size * ncols * 1.1
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max
    fig = plt.figure(101, figsize=(figsize_x, figsize_y))  # slight adjust for the axes
    fig.suptitle('Cell: row {}, col {}'.format(row,col))

    gs = gridspec.GridSpec(nrows, ncols)

    # 1d hist
    ax_1dhist = plt.subplot(gs[0, :])

    # point estimate
    yi, edges = np.histogram(z, bins='auto', density=True)
    x, y = histogramize(edges, yi)
    ax_1dhist.plot(x, y, linewidth=2, label=r'$z_{COSMOS}$')


    key = 'Z'
    '''
    keys = [key + '{}_{:02}'.format(i,j) for i in range(6) for j in range(100)]

    hist = np.sum(df[keys].values, axis=0)
    hist = hist / float(np.max(hist))
    ax_1dhist.plot(np.linspace(0,6,601)[:-1], hist, '.', label=r'$z_{COSMOS}$')
    #ax_1dhist.set_xlim((0,2))
    if spec_z is not None:
        for val in spec_z:
            ax_1dhist.axvline(val)

    ax_1dhist.set_xlabel(zkey)
    ax_1dhist.legend()
    '''
    # distributions of other parameters in cell
    # i band lupt
    lupt_i = data['wide_lupt_i']
    ax_i_lupt_hist = plt.subplot(gs[2,0])
    ax_i_lupt_hist.set_title('i band lupt')
    ax_i_lupt_hist.hist(lupt_i)
    ax_i_lupt_hist.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # r - i
    lupt_ri = data['wide_lupticolor_r-i']
    ax_ri_lupt_hist = plt.subplot(gs[2,1])
    ax_ri_lupt_hist.set_title('r-i')
    ax_ri_lupt_hist.hist(lupt_ri)
    ax_ri_lupt_hist.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # z - i
    lupt_zi = data['wide_lupticolor_z-i']
    ax_zi_lupt_hist = plt.subplot(gs[2,2])
    ax_zi_lupt_hist.set_title('z-i')
    ax_zi_lupt_hist.hist(lupt_zi)
    ax_zi_lupt_hist.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # somspace plots
    assign_keys = ['other_c0', 'other_c1']
    c0 = df[assign_keys[0]].values
    c1 = df[assign_keys[1]].values
    for j, agg_kind in enumerate(['log10 N', 'mean', 'std']):
        ax = plt.subplot(gs[1, j])
        agg_func = {'log10 N': 'size', 'mean': 'mean', 'std': np.nanstd}[agg_kind]
        log_scale_j = {'log10 N': True, 'mean': log_scale, 'std': log_scale}[agg_kind]
        plot_key_somspace(z, c0, c1, other_som, fig, ax, agg_func, log_scale_j, **kwargs)

        ax.set_xlabel('c_0')
        ax.set_ylabel('c_1')
        ax.set_title('{0}: {1}'.format(zkey, agg_kind))
        if agg_kind == 'log10 N':
                ax.set_title(r'$\log_{{10}} N$')
        elif agg_kind == 'mean':
            ax.set_title(r'$\langle {0}\rangle$'.format(zkey.lower()))
        else:
            ax.set_title(r'$\sigma({0})$'.format(zkey.lower()))

    # hist of chi^2 of gals that make it into cell
    ax_chi2_hist = plt.subplot(gs[3,0])
    ax_chi2_hist.set_title(r'$\chi^2$')
    ax_chi2_hist.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    chi2_vals = data['cell_' + cell_kind + '_chi2']
    ax_chi2_hist.hist(chi2_vals)

    # p(this cell | other cell)
    ax_pcchat = plt.subplot(gs[3,1])
    if pcchat is not None:
        if cell_kind == 'wide':
            title = r'$p(c|\hat{c})$'
            C = pcchat[:,c]
        elif cell_kind == 'deep':
            title = r'$p(\hat{c}|c)$'
            C = pcchat[c,:]
        C = np.reshape(C, other_som.map_shape)
        im = ax_pcchat.pcolor(C)
        plt.colorbar(im, ax=ax_pcchat)
        ax_pcchat.set_title(title)

    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig

"""
def plot_cell(z, other_c, other_som, zkey, log_scale=False, path='', **kwargs):
    '''Take all of sample in certain som cell and plot 1d histogram and
    breakdown of sample in other som.

    Parameters
    ----------
    z : array of values
    other_c : cell values in a SOM
    other_som : SOM object
    zkey : the key associated with z
    path : if specified, save figure to this path

    Returns
    -------
    matplotlib figure object
    '''
    other_cND = other_som.cell1d_to_cell(other_c)
    df = pd.DataFrame({'z': z, 'other_c0': other_cND[0], 'other_c1': other_cND[1]})
    # 3 col, 2 rows
    # row 1: 1d hist of key
    # further rows: [logN occupation, average in cell, std in cell]
    nrows = 2
    ncols = 3
    cell_size = 4
    figsize_x = cell_size * ncols * 1.3
    figsize_y = cell_size * nrows
    figsize_x_max = 20.
    if figsize_x > figsize_x_max:
        figsize_y = figsize_x_max / figsize_x * figsize_y
        figsize_x = figsize_x_max
    fig = plt.figure(figsize=(figsize_x, figsize_y))  # slight adjust for the axes
    gs = gridspec.GridSpec(nrows, ncols)
    # 1d hist
    ax_1dhist = plt.subplot(gs[0, :])
    yi, edges = np.histogram(z, bins='auto', density=True)
    x, y = histogramize(edges, yi)
    ax_1dhist.plot(x, y, linewidth=2)
    ax_1dhist.set_xlabel(zkey)

    # somspace plots
    assign_keys = ['other_c0', 'other_c1']
    c0 = df[assign_keys[0]].values
    c1 = df[assign_keys[1]].values
    for j, agg_kind in enumerate(['log10 N', 'mean', 'std']):
        ax = plt.subplot(gs[1, j])
        agg_func = {'log10 N': 'size', 'mean': 'mean', 'std': 'std'}[agg_kind]
        log_scale_j = {'size': True, 'mean': log_scale, 'std': log_scale}[agg_func]
        plot_key_somspace(z, c0, c1, other_som, fig, ax, agg_func, log_scale_j, **kwargs)

        ax.set_xlabel('c_0')
        ax.set_ylabel('c_1')
        ax.set_title('{0}: {1}'.format(zkey, agg_kind))
        if agg_kind == 'log10 N':
                ax.set_title(r'$\log_{{10}} N$')
        elif agg_kind == 'mean':
            ax.set_title(r'$\langle {0}\rangle$'.format(zkey.lower()))
        else:
            ax.set_title(r'$\sigma({0})$'.format(zkey.lower()))

    fig.tight_layout()

    if len(path) > 0:
        fig.savefig(path)

    return fig
"""
def plot_key_somspace(z, c0, c1, som, fig, ax, agg_func='mean', log_scale=False, colorbar=True, special_cell=None, **kwargs):
    df = pd.DataFrame({'z': z, 'c0': c0, 'c1': c1})
    agg = df.groupby(['c0', 'c1']).agg(agg_func)
    try:
        agg = agg['z']
    except:
        pass
    if log_scale:
        agg_values = np.log10(agg.values)
    else:
        agg_values = agg.values
    #print(agg_values)
    C = agg_matrix(agg, agg_values, map_shape=som.map_shape)
    im = ax.pcolor(C, **kwargs)
    #im = ax.imshow(C, **kwargs)
    #im = ax.pcolormesh(C, **kwargs)
    # set xlim and ylim from map_shape
    ax.set_xlim(0, som.map_shape[0])
    ax.set_ylim(0, som.map_shape[1])
    cb = fig.colorbar(im, ax=ax)
    if not colorbar:
        cb.remove()
    if special_cell is not None:
        pass

def agg_matrix(agg, z=None, size=None, map_shape=[69, 72], min_occupation=0):
    '''Given aggregated data and a key, create 2d masked array
    '''
    levels_x = agg.index.levels[0].values
    levels_y = agg.index.levels[1].values
    indx_x_transform = levels_x[agg.index.codes[0].values()]
    indx_y_transform = levels_y[agg.index.codes[1].values()]

    bins_x = map_shape[0] + 1
    bins_y = map_shape[1] + 1
    C = np.ma.zeros((bins_x, bins_y))
    C.mask = np.ones((bins_x, bins_y))
    if z is None:
        np.add.at(C, [indx_x_transform, indx_y_transform], 1)
    else:
        # filter any nans
        if np.any(~np.isfinite(z)):
            conds = np.isfinite(z)
            indx_x_transform = indx_x_transform[conds]
            indx_y_transform = indx_y_transform[conds]
            z = z[conds]
            if min_occupation > 0:
                size = size[conds]
        np.add.at(C, [indx_x_transform, indx_y_transform], z)
    np.multiply.at(C.mask, [indx_x_transform, indx_y_transform], 0)
    C = C.T

    # mask any C that are nan
    C.mask = np.where(C != C, True, C.mask)

    if min_occupation > 0 and z is not None:
        Hist = np.zeros((bins_x, bins_y))
        np.add.at(Hist, [indx_x_transform, indx_y_transform], size)
        Hist = Hist.T
        Hist_mask = np.where(Hist < min_occupation, True, False)
        C.mask = (C.mask + Hist_mask)

    return C

def plot_cchat(chat0, chat1, cmap, scatter=False):
    chat = cmap.wide_som.cell2d_to_cell1d(chat0, chat1)
    p = cmap.pcchat[:, chat].reshape(cmap.deep_som.map_shape[1], cmap.deep_som.map_shape[0])

    fig = plt.figure(figsize=(10, 8))
    plt.pcolor(p)
    plt.colorbar()
    plt.xlabel('C0')
    plt.ylabel('C1')
    if scatter:
        plt.scatter(chat0 + 0.5, chat1 + 0.5, s=50, color='r', marker='o', alpha=0.5)
    plt.title(r'$p(c|\hat{{c}},\hat{{s}})$ for $\hat{{c}}=({0},{1})$'.format(chat0, chat1))
    return fig

def plot_chi2(indx, cmap, x, ivar):
    # calculate chi2
    chi2 = cmap.som.evaluate(np.array([x[indx]]), np.array([ivar[indx]]))[0]

    # turn into prob
    prob = np.exp(-0.5 * (chi2.reshape(cmap.som.map_shape) - np.min(chi2)))
    prob = prob / prob.sum()

    # plot
    fig = plt.figure(figsize=(10, 8))
    plt.pcolor(prob)

    # also plot assigned and mc draws
    cd0 = cmap.data['cell_deep_0'][indx]
    cd1 = cmap.data['cell_deep_1'][indx]
    plt.plot(cd0, cd1, 'ro', alpha=0.7)
    cw0 = cmap.data['cell_wide_0'][indx]
    cw1 = cmap.data['cell_wide_1'][indx]
    plt.plot(cw0, cw1, 'rs', alpha=0.7)

    plt.colorbar()
    plt.title('p(Chi2) for Object {0}'.format(indx))
    return fig

def histogramize(bins, y):
    '''Return set of points that make it look like a histogram.
    bins is assumed to be one longer than y. You then just "plot" these!'''
    xhist = []
    yhist = []
    for i in range(len(y)):
        xhist.append(bins[i])
        xhist.append(bins[i + 1])
        yhist.append(y[i])
        yhist.append(y[i])
    xhist = np.array(xhist)
    yhist = np.array(yhist)
    return xhist, yhist

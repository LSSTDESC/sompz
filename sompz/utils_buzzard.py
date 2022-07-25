import pdb
import h5py
import sys, os, time
import pickle
import numpy as np
import pandas as pd
import healpy as hp
import yaml, glob
from math import pi
import numba
#from numpy.interpolate import interp1d
from scipy.interpolate import interp1d
import fitsio
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))
import sompz
from sompz import CellMap, CellMapLupticolorLupticolorluptitude

def load_fits_file(path, table=1, columns=['RA', 'DEC'], **kwargs):
    """Load up fits file with fitsio, returns pandas dataframe

    Parameters
    ----------
    path : path to fits file
    table : extension number
    columns : columns to read
    kwargs : any kwargs to pass to fitsio

    Returns
    -------
    cat : pandas dataframe with specified columns

    Notes
    -----
    Has to do some fun byte swap stuff.

    """
    data = fitsio.read(path, ext=table, columns=columns, **kwargs)

    # deal with fact that you might have loaded up vector row
    cat = {}
    for key in columns:
        if len(data[key].shape) == 2:
            # go through each key
            for i in range(data[key].shape[1]):
                cat['{0}_{1:03d}'.format(key, i)] = data[key][:, i].byteswap().newbyteorder()
        else:
            cat[key] = data[key].byteswap().newbyteorder()
    # turn into pandas dataframe
    cat = pd.DataFrame(cat)

    return cat

def get_df_true(true_mag_file, true_pos_file, despzbands_file):

    print('loading true information')
    df_true_pos = load_fits_file(true_pos_file, columns=['ID', 'Z', 'RA', 'DEC', 'SIZE', 'TILE', 'REDSHIFT FIELD']) # 'INDEX'
    df_true_pos.rename({'RA' : 'RA_ORIGINAL', 'DEC' : 'DEC_ORIGINAL'}, inplace=True, axis='columns')

    print('loading despz information')
    df_true_mag = load_fits_file(true_mag_file, columns=['FLUX', 'LMAG'])
    df_despzbands = pd.read_csv(despzbands_file, skiprows=0)
    despzbands = [b[0].split('.')[0].split('/')[-1] for b in df_despzbands.values]
    df_true_mag.columns = ['TRUEFLUX_' + b for b in despzbands] + ['TRUEMAG_' + b for b in despzbands]

    return df_true_pos, df_true_mag

def shift_buzzard_colors_txt(df_sims, mag_shift_file):

    mag_shift_z = np.genfromtxt(mag_shift_file, names=True)

    mag_shift_columns = ['TRUEMAG_des_u', 'TRUEMAG_desy3std_g',
                         'TRUEMAG_VISTA_Filters_at80K_forETC_J',
                         'TRUEMAG_VISTA_Filters_at80K_forETC_H',
                         'TRUEMAG_VISTA_Filters_at80K_forETC_Ks']
    zmeans = mag_shift_z['zmeans']

    splines = [interp1d(zmeans, mag_shift_z[k]) for k in mag_shift_columns]
    spline_dict = dict(zip(mag_shift_columns, splines))


    zs = np.copy(df_sims['Z'])
    zs[zs < zmeans[0]] = zmeans[0]
    zs[zs > zmeans[-1]] = zmeans[-1]

    for k in mag_shift_columns:
        df_sims[k] += spline_dict[k](zs)


    return df_sims


def get_balrog_sample(df_true, obs_mag_files, fmap_file, label,
                      bal_error_model=False):
    all_wl = pd.DataFrame()
    for i, obs_mag_file in enumerate(obs_mag_files):
        print('loading obs information {0}'.format(i))
        if not bal_error_model:
            df_obs = load_fits_file(obs_mag_file, columns=['RA', 'DEC','FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z',
            'IVAR_G', 'IVAR_R', 'IVAR_I', 'IVAR_Z',
            'MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z',
            'MAGERR_G', 'MAGERR_R', 'MAGERR_I', 'MAGERR_Z',
            'PHOTOZ_GAUSSIAN'])
        else:
            df_obs = load_fits_file(obs_mag_file, columns=['RA', 'DEC','FLUX_G', 'FLUX_R', 'FLUX_I', 'FLUX_Z',
            'IVAR_G', 'IVAR_R', 'IVAR_I', 'IVAR_Z',
            'MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z',
            'MAGERR_G', 'MAGERR_R', 'MAGERR_I', 'MAGERR_Z',
            'PHOTOZ_GAUSSIAN', 'METACAL/flux_r', 'METACAL/flux_i',
            'METACAL/flux_z', 'METACAL/flux_ivar_r', 'METACAL/flux_ivar_i', 'METACAL/flux_ivar_z'])


        wl = pd.concat([df_obs, df_true], ignore_index=True, axis=1)
        wl.columns = df_obs.columns.tolist() + df_true.columns.tolist()
        all_wl = pd.concat((all_wl, wl))

    #all_wl = wl_selection(all_wl, d, dhdr, label) # old. used for versions earlier than buzzard v1.9.8
    fmap = h5py.File(fmap_file, mode='r')
    all_wl = make_mcal_selection(all_wl, fmap, label)
    fmap.close()
    return all_wl

def make_mcal_selection(df_all, fmap, wl_label, x_opt=[21.25431103,  2.1260211,   1.05209237]):
    ''' original source: https://github.com/j-dr/pyaddgals/blob/master/bin/skyfactory/make_master_h5_cat.py#L332'''

    psfmap = {'PIXEL': fmap['maps/hpix'][:], 'SIGNAL': fmap['maps/i/fwhm'][:]}

    pidx = psfmap['PIXEL'].argsort()
    psfmap['SIGNAL'] = psfmap['SIGNAL'][pidx]
    psfmap['PIXEL'] = psfmap['PIXEL'][pidx]
    del pidx

    gpix = hp.ang2pix(4096, df_all['RA'], df_all['DEC'],
                      lonlat=True)
    psfidx = psfmap['PIXEL'].searchsorted(gpix)
    del gpix

    gpsf = 0.26 * 0.5 * psfmap['SIGNAL'][psfidx]
    del psfmap['PIXEL'], psfmap['SIGNAL']
    del psfmap

    idx = np.sqrt(df_all['SIZE'].values**2 + gpsf**2) > (x_opt[2] * gpsf)
    del gpsf

    #idx &= np.abs(df_all['E1']) < 1
    #idx &= np.abs(df_all['E2']) < 1
    idx &= df_all['MAGERR_R'] < 0.25
    idx &= df_all['MAGERR_I'] < 0.25
    idx &= df_all['MAGERR_Z'] < 0.25
    idx &= df_all['MAG_I'] < (x_opt[0] + x_opt[1] * df_all['Z'])

    df_all[wl_label] = False
    df_all[wl_label][idx] = True

    return df_all

def select_metacal_buzzard(f, x_opt = [21.25431103,  2.1260211,   1.05209237]):
    select_metacal = f['index']['select']
    print('old len', select_metacal)

    # Apply size cut (the only missing part of the WL selection) to the wide sample.
    #fmap = h5py.File(fmap_file, mode='r')
    psfmap = {'pixel': f['maps/hpix'][:], 'signal': f['maps/i/fwhm'][:]}
    print(len(f['maps/hpix'][:])) # ~(12*4096**2)/8. NSIDE=4096. DES is 1/8 of sky.

    pidx = psfmap['pixel'].argsort()
    psfmap['signal'] = psfmap['signal'][pidx]
    psfmap['pixel'] = psfmap['pixel'][pidx]
    del pidx

    ras = np.array(f['catalog/metacal/unsheared/ra'][:][select_metacal])
    decs = np.array(f['catalog/metacal/unsheared/dec'][:][select_metacal])
    gpix = hp.ang2pix(4096, ras, decs,
                      lonlat=True)
    psfidx = psfmap['pixel'].searchsorted(gpix)
    del gpix

    gpsf = 0.26 * 0.5 * psfmap['signal'][psfidx]
    del psfmap['pixel'], psfmap['signal']
    del psfmap

    sizes =  f['catalog/metacal/unsheared/size'][:][select_metacal]

    idx = np.array(f['catalog/gold/mag_i_true'][:][select_metacal]) < 24.5
    idx &= np.isfinite(f['catalog/gold/mag_i_lensed'][:][select_metacal])

    idx &= np.sqrt(sizes**2 + gpsf**2) > (x_opt[2] * gpsf)

    idx &= np.abs(f['catalog/metacal/unsheared/e1'][:][select_metacal]) < 1
    idx &= np.abs(f['catalog/metacal/unsheared/e2'][:][select_metacal]) < 1
    idx &= f['catalog/gold/mag_err_r'][:][select_metacal] < 0.25
    idx &= f['catalog/gold/mag_err_i'][:][select_metacal] < 0.25
    idx &= f['catalog/gold/mag_err_z'][:][select_metacal] < 0.25
    idx &= f['catalog/gold/mag_i'][:][select_metacal] < (x_opt[0] +
                                                         x_opt[1] * f['catalog/bpz/unsheared/z'][:][select_metacal])


    select_metacal = select_metacal[:][idx]
    print('new len', len(select_metacal))
    return select_metacal

def get_deep_noise(true):
    return 1

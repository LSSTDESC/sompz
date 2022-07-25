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
import numpy as np
import numba
import time
import pdb
"""
.. module:: cell
"""

from .som import SelfOrganizingMap, unravel_index
from .plots import plot_key, plot_sample, plot_sample_colors, plot_cell, histogramize
from .plots import plot_som_mean_z, plot_som_diagnostic
from .plots import plot_som_diagnostic_balrog_vs_wide

class CellMap(object):
    """This class will handle the the cell mapping system. We have two
    different cell assignment schemes (say, deep field and wide field). Call
    the better one "c" and the noisier one "chat". Call "s" our particular
    sample, and "z" some quantity of interest. We then want to learn the
    following pieces of information:

        - p(c | chat, s)
        - p(z | chat, s) = \sum_{c} p(z | c) p(c | chat, s)

    This code will also return these properties for collections of cells. These
    collections will be called "t". This class can construct a "t" for some
    set of requirements or conditions on other quantities "z" by using p(z | c)
    and then propogating for our chat sample:

        - p(z | t, s) = \sum_{c, chat} p(z | c) p(c | chat, s) p(chat | t, s)

    Finally, this code will be able to return samples from the above
    probabilities:

        - p(c | chat, s)
        - p(z | chat, s)
        - p(z | c)
        - p(c | chat, s) and also p(c, chat | s)
        - p(z | t)

    Possible Extensions
    -------------------
    The CellMap does not really care about the SelfOrganizingMap aspect -- that
    is simply the way we build the classification mappings p(c | f) and
    p(chat | fhat). The som object can be any sort of object as long as it has
    some sort of assign function and probability function. So, you could write a
    new CellMap class that fits some other sort of object, and then modify
    assign_deep and assign_wide.


    Summary of the different histogram methods
    ------------------------------------------
    histogram_true : given sample of z, return histogram
    histogram_wide : given wide cells, gets z from wide cells and makes histogram
    histogram_deep : given deep cells, gets z from deep cells and makes histogram
    histogram : given wide cells, gets z from deep cells and histograms based on probability of being in deep cell given wide cell info
    histogram_chi2 : given wide cells and overlap data, does histogram but without creating the p(c|chat) matrix
    """

    def __init__(self, data, overlap_weight, wide_som, deep_som, pcchat, wide_columns, wide_err_columns, deep_columns, deep_err_columns, zp=None, **kwargs):
        """Initialize the CellMap Object

        Parameters
        ----------
        data :              A pandas dataframe of galaxies with both deep and
                            wide observations. It must contain the
                            wide_columns, wide_err_columns, deep_columns, and
                            deep_err_columns.
        overlap_weight :    Weight vector to account for shear response, or uneven number
                            of times the deep galaxies were drawn, in data.
        wide_som :          SelfOrganizingMap object trained on the wide fluxes
        deep_som :          SelfOrganizingMap object trained on the deep fluxes
        pcchat :            An array of shape (n_tomo_bins, prod(*som.map_shape),
                            prod(*som.map_shape)) which gives the probability
                            of a galaxy really being in cell c given that it is
                            in cell chat. This may or may not have come from
                            data -- we could train this on some other sample.
        wide_columns :      Columns which correspond to information used to
                            make wide field assignment to SOM.
        wide_err_columns :  Columns which correspond to errors in the wide
                            field parameters.
        deep_columns :      Columns which correspond to information used to
                            make deep field assignment to SOM.
        deep_err_columns :  Columns which correspond to errors in the deep
                            field parameters.
        """
        import pandas as pd
        self.data = data
        if(overlap_weight is not None):
            data['overlap_weight']=overlap_weight.copy()
        else: # overlap_weight is None
            if 'overlap_weight' in data.columns:
                print("using overlap_weight as stored in CellMap.data")
            else:
                print("setting overlap_weight to a bunch of ones")
                data['overlap_weight']=np.ones(len(data))

        self.wide_som = wide_som
        self.deep_som = deep_som

        self.pcchat = pcchat

        self.wide_columns = wide_columns
        self.wide_err_columns = wide_err_columns
        self.deep_columns = deep_columns
        self.deep_err_columns = deep_err_columns

        self.zp = zp

        if 'cell_deep' not in self.data:
            self.data['cell_deep'] = self.assign_deep(self.data)
        self._deep_groups = self.data.groupby('cell_deep')
        self._number_deep_cells = self.deep_som.size
        if 'cell_wide' not in self.data:
            self.data['cell_wide'] = self.assign_wide(self.data)
        self._wide_groups = self.data.groupby('cell_wide')
        self._number_wide_cells = self.wide_som.size

        # precompute bins if possible - now deprecated!
        #if 'Z' in data:
        #    self._hist_bins = np.arange(0, 1.6, 0.02)
        #    all_cells = np.arange(self._number_deep_cells)
        #    hists = []
        #    for c in all_cells:
        #        try:
        #            df = self._deep_groups.get_group(c)
        #            z = df['Z'].values
        #            hist = np.histogram(z, self._hist_bins, density=True)[0]
        #            hists.append(hist)
        #        except KeyError:
        #            hists.append(np.zeros(len(self._hist_bins) - 1))
        #    hists = np.array(hists)  # (n_deep_cells, n_bins)
        #    self._deep_hists = hists

        self.kwargs = {}
        self.kwargs.update(kwargs)



    @property
    def som(self):
        import warnings
        warnings.warn('The som property is deprecated in favor of deep_som and wide_som. We are going to give you the deep_som', DeprecationWarning)
        return self.deep_som

    @classmethod
    def read(cls, path, name='cellmap'):
        import sompz
        import pandas as pd
        import h5py

        print('reading pcchat...')
        # pcchat
        try:
            with h5py.File(path, 'r') as h5f:
                pcchat = h5f['{0}/pcchat'.format(name)][:]
                print('...success')
        except AttributeError:
            try:
                print('key exists but not in h5py format. trying pandas of the past')
                pcchat = pd.read_hdf(path, '{0}/pcchat'.format(name)).values
            except KeyError:
                print('found no pcchat. so just putting None')
                pcchat = None
        except KeyError:
            print('found no pcchat. so just putting None')
            pcchat = None
        except ValueError as e:
            print(e)
            print('just putting None')
            pcchat = None

        print('reading zp...')
        # zp
        try:
            with h5py.File(path, 'r') as h5f:
                zp = h5f['{0}/zp'.format(name)][()]
                print('...success')
        except Exception as e:
            print(e)
            print('found no zp. just putting None')
            zp = None
        print('zeropoint =',zp)

        data = pd.read_hdf(path, '{0}/data'.format(name))

        overlap_weight = None

        print('reading columns...')
        try:
            # try reading with h5py instead
            with h5py.File(path, 'r') as h5f:
                deep_columns = h5f['{0}/deep_columns'.format(name)][:].tolist()
                deep_columns = [c.decode('utf-8') for c in deep_columns]
                deep_err_columns = h5f['{0}/deep_err_columns'.format(name)][:].tolist()
                deep_err_columns = [c.decode('utf-8') for c in deep_err_columns]
                wide_columns = h5f['{0}/wide_columns'.format(name)][:].tolist()
                wide_columns = [c.decode('utf-8') for c in wide_columns]
                wide_err_columns = h5f['{0}/wide_err_columns'.format(name)][:].tolist()
                wide_err_columns = [c.decode('utf-8') for c in wide_err_columns]
                kind = h5f['{0}/type'.format(name)][:].tolist()[0]
                kind = kind.decode('utf-8')
                print('...success')
        except:
            print('reading columns with h5 failed. trying hdf')
            wide_columns = pd.read_hdf(path, '{0}/wide_columns'.format(name)).values
            deep_columns = pd.read_hdf(path, '{0}/deep_columns'.format(name)).values

            wide_err_columns = pd.read_hdf(path, '{0}/wide_err_columns'.format(name)).values
            deep_err_columns = pd.read_hdf(path, '{0}/deep_err_columns'.format(name)).values

            kind = pd.read_hdf(path, '{0}/type'.format(name))['type'].values[0]

        # parse kwargs
        kwargs = {}
        # hdf = pd.HDFStore(path)
        # for col in hdf.keys():
        #     slash_split = col.split('/')
        #     if '{0}'.format(name) not in col:
        #         continue
        #     if slash_split[2] == 'kwargs':
        #         key = '/'.join(slash_split[2:])
        #         kwargs[key] = hdf[col].values[0]
        # hdf.close()

        print('reading SOMs...')
        try:
            wide_som = SelfOrganizingMap.read(path, name='{0}/wide_som'.format(name))
            deep_som = SelfOrganizingMap.read(path, name='{0}/deep_som'.format(name))
            print('...success')
        except KeyError:
            print('compatibility fix: we put the som not with the cellmap')
            try:
                wide_som = SelfOrganizingMap.read(path, name='wide_som')
                deep_som = SelfOrganizingMap.read(path, name='deep_som')
            except KeyError:
                print('compatibility: we have only one SOM')
                wide_som = SelfOrganizingMap.read(path, name='som')
                deep_som = wide_som.copy()

        cm_class = getattr(sompz, kind)
        cm = cm_class(data, overlap_weight, wide_som, deep_som, pcchat, wide_columns, wide_err_columns, deep_columns, deep_err_columns, zp, **kwargs)
        return cm

    def write(self, path, name='cellmap'):
        import h5py
        import pandas as pd
        self.data.to_hdf(path, '{0}/data'.format(name))

        for key in self.kwargs:
            pd.Series([self.kwargs[key]]).to_hdf(path, '{0}/kwargs/{1}'.format(name, key))

        self.wide_som.write(path, name='{0}/wide_som'.format(name))
        self.deep_som.write(path, name='{0}/deep_som'.format(name))

        with h5py.File(path, 'r+') as h5f:
            if self.pcchat is not None:
                try:
                    h5f.create_dataset('{0}/pcchat'.format(name), data=self.pcchat)
                except RuntimeError:
                    # path already exists
                    h5f['{0}/pcchat'.format(name)][...] = self.pcchat

            if self.zp is not None:
                print(self.zp)
                print(type(self.zp))
                try:
                    h5f.create_dataset('{0}/zp'.format(name), data=self.zp)
                except RuntimeError:
                    # path already exists
                    h5f['{0}/zp'.format(name)][...] = self.zp

            for ci in [['wide_columns', self.wide_columns], ['deep_columns', self.deep_columns],
                       ['wide_err_columns', self.wide_err_columns], ['deep_err_columns', self.deep_err_columns],
                       ['type', [self.__class__.__name__]]]:
                label, data_raw = ci
                data = [x.encode('utf-8') for x in data_raw]
                col = '{0}/{1}'.format(name, label)
                try:
                    h5f.create_dataset(col, data=data)
                except RuntimeError:
                    # path already exists. But we don't want to save because we have no idea if the column is long enough.
                    del h5f[col]
                    h5f.create_dataset(col, data=data)

    @classmethod
    def one_point_statistics(cls, y, bins):
        """Given a histogram and its bins return summary statistics

        Parameters
        ----------
        y :     A histogram of values
        bins :  The bins of the histogram

        Returns
        -------
        normalization, mean, sigma
        """
        dx = np.diff(bins)
        x = 0.5 * (bins[1:] + bins[:-1])
        normalization = np.trapz(y, x=x, dx=dx)
        mean = np.trapz(x * y, x=x, dx=dx) / normalization
        var = np.trapz((x - mean) ** 2 * y, x=x, dx=dx) / normalization
        sigma = np.sqrt(var)
        return normalization, mean, sigma

    @classmethod
    def overlap(cls, hists, weight, normalized=True):
        """Given a set of (normalized) histograms and their relative weight, compute the overlap of the histograms

        Parameters
        ----------
        hists :   An array of histograms (usually the histogram in each tomographic bin)
        weight :  The relative weight between the histogramms (usually the number of galaxies in each bin)
        normalized : True if the histograms are normalized

        Returns
        -------
        overlap
        """

        if hists.ndim < 2:
            raise ValueError("You must enter an array of at least two histograms.")
        elif hists.shape[0] != weight.shape[0]:
            raise ValueError("A weight must be provided for each histogram.")
        #Ignore the empty histogramms
        non_zero = (hists.sum(axis=1) != 0)
        hists = hists[non_zero]
        weight = weight[non_zero]

        hists_unormalised = (hists/hists.sum(axis=1)[:, None])
        if normalized:
            hists_unormalised *= weight[:, None]
        overlap = (hists_unormalised.sum(axis=0)-hists_unormalised.max(axis=0)).sum()/hists_unormalised.sum()

        return overlap

    @classmethod
    def bin_count(cls, cells, tomo_bins):
        """Given the assignement of data to cells and the assignement of cells to tomographic bins, return the number of objects in each tomographic bin

        Parameters
        ----------
        cells :   A list of cell assignement
        tomo_bins :  A dictionnary containing as keys the bin and as values an array of the cells assigned to this bin

        Returns
        -------
        n_object_in_bin : The number of object assigned to each bin
        """

        n_object_in_bin = []
        for key in tomo_bins:
            cells_use = tomo_bins[key]
            cells_conds = np.in1d(cells, cells_use)
            n_object_in_bin.append(np.sum(cells_conds))
            print('Bin {0}: {1:.3e}, {2:.3f}'.format(key, np.sum(cells_conds), np.sum(cells_conds) * 1. / len(cells_conds)))

        n_object_in_bin = np.array(n_object_in_bin)

        return n_object_in_bin


    def histogram_true(self, z, bins, weights=None):
        """Given data z, return histogram. Very simple wrapper.

        Parameters
        ----------
        z :     Parameter we histogram
        bins :  The bins of the histogram
        weights: the weights of the parameters, default to ones

        Returns
        -------
        hist :  A histogram of the values from z
        """
        if str(type(weights)) == "<type 'NoneType'>":
            weights = np.ones(len(z))

        hist, edges = np.histogram(z, bins,weights=weights)

        # I am pretty sure this isn't necessary, but just in case
        dx = np.diff(bins)
        normalization = np.sum(dx * hist)
        if normalization != 0:
            hist = hist / normalization
        return hist

    @staticmethod
    def histogram_from_fullpz(df, key, overlap_weighted, bin_edges, full_pz_end = 6.00, full_pz_npts=601):
        '''Preserve bins from Laigle'''
        dz_laigle = full_pz_end / (full_pz_npts - 1)
        condition = np.sum(~np.equal(bin_edges, np.arange(0 - dz_laigle/2.,
                                      full_pz_end + dz_laigle,
                                      dz_laigle)))
        assert condition == 0
        # bin_edges: [-0.005, 0.005], (0.005, 0.015], ... (5.995, 6.005]

        single_cell_hists = np.zeros((len(df), len(key)))

        overlap_weights = np.ones(len(df))
        if(overlap_weighted):
            overlap_weights = df['overlap_weight'].values

        single_cell_hists[:,:] = df[key].values

        # normalize sompz p(z) to have area 1
        dz = 0.01
        area = np.sum(single_cell_hists, axis=1) * dz
        area[area == 0] = 1 # some galaxies have pz with only one non-zero point. set these galaxies' histograms to have area 1
        area = area.reshape(area.shape[0], 1)
        single_cell_hists = single_cell_hists / area

        # response weight normalized p(z)
        single_cell_hists = np.multiply(overlap_weights, single_cell_hists.transpose()).transpose()

        # sum individual galaxy p(z) to single cell p(z)
        hist = np.sum(single_cell_hists, axis=0)

        # renormalize p(z|c)
        area = np.sum(hist) * dz
        hist = hist / area

        return hist

        """ take a discretely, evenly sampled p(z) and turn it into a histogram with specified bin_edges """
        '''
        assert bin_edges[-1] == full_pz_end, 'The bin range of the data doesn\'t match the bins you have input.'

        # There is one more bin edge than there are bins in the histogram.
        # The left-most bin edge is the lowest value in which our p(z) are estimated (0.00)
        # the right-most bin edge is the highest value in which or p(z) are estimated (full_pz_end)
        single_cell_hists = np.zeros((len(df), len(bin_edges) - 1)) # nedges-1=nbins

        histogram_bin_width_ratio = float(full_pz_npts - 1) / (len(bin_edges) - 1)
        assert histogram_bin_width_ratio.is_integer(), 'the bin width ratio must be an integer'
        histogram_bin_width_ratio = int(histogram_bin_width_ratio)

        overlap_weights = np.ones(len(df))
        if(overlap_weighted):
            overlap_weights = df['overlap_weight'].values

        # 1.  put unambiguous COSMOS p(z) points into bins
        # 1.1 first and last bins. These are handled differently because they are technically 'edge' points that would be split, but aren't because they are at the ends of the pz range.
        single_cell_hists[:, 0] = np.multiply(overlap_weights,df[key].values[:, 0])
        single_cell_hists[:,-1] = np.multiply(overlap_weights,df[key].values[:,-1])
        # 1.2 rest of bins
        # The number of unambiguous points that fall into the bins depends on the ratio of the bin width of the histogram to the width of the pz samples
        # If you have a histogram bin width of 0.02 and pz samples at every 0.01, then every other point falls onto an edge, and for each edge point you have one unambiguous point
        # If you have a histogram bin width of 0.03 and pz samples at every 0.01, then every 3rd point falls onto an edge, and for each edge point you have two unambiguous points
        # So we need to loop from 0 to this ratio to put the unambiguous points into bins
        for i in range(histogram_bin_width_ratio - 1):
            single_cell_hists[:,:] += np.multiply(overlap_weights,df[key].values[:,i+1::histogram_bin_width_ratio].transpose()).transpose()

        # 2. split COSMOS p(z) points on edges between bins into bins
        # the division by 2 indicates we are splitting the pz into two neighboring bins.
        single_cell_hists[:,:-1] += np.multiply(overlap_weights,(df[key].values[:,histogram_bin_width_ratio:-histogram_bin_width_ratio:histogram_bin_width_ratio] / 2.).transpose()).transpose()
        single_cell_hists[:, 1:] += np.multiply(overlap_weights,(df[key].values[:,histogram_bin_width_ratio:-histogram_bin_width_ratio:histogram_bin_width_ratio] / 2.).transpose()).transpose()

        # normalize sompz p(z) to have area 1
        dz = bin_edges[1] - bin_edges[0]
        area = np.sum(single_cell_hists, axis=1) * dz
        area[area == 0] = 1 # some galaxies have pz with only one non-zero point. set these galaxies' histograms to have area 1
        area = area.reshape(area.shape[0], 1)
        single_cell_hists = single_cell_hists / area

        # sum individual galaxy p(z) to single cell p(z) and renormalize
        hist = np.sum(single_cell_hists, axis=0)
        area = np.sum(hist) * dz
        hist = hist / area
        '''


    def histogram_wide(self, key, overlap_weighted, cells, cell_weights, bins=None):
        """Return histogram from values that live in specified wide cells

        Parameters
        ----------
        key : Parameter(s) to extract from dataframe
        overlap_weighted : Whether or not to weight the histogram by overlap weight.
        cells : A list of wide cells to return sample from, or a single int.
        cell_weights : How much we weight each wide cell. This is the array p(chat | sample), possibly overlap weighted if you like
        bins : Bins we histogram the values into

        Returns
        -------
        hist : a histogram of the values from self.data[key]

        """
        # p(z|chat)
        hists = []
        for chat in cells:
            try:
                df = self._wide_groups.get_group(chat)
                if type(key) is str:
                    z = df[key].values
                    hist = np.histogram(z, bins, density=True)[0]
                    hists.append(hist)
                elif type(key) is list:
                    # use full p(z)
                    assert(bins is not None)
                    hist = self.histogram_from_fullpz(df, key, overlap_weighted, bins)
                    hists.append(hist)
            except KeyError:
                hists.append(np.zeros(len(bins) - 1))
        hists = np.array(hists)

        # p(z|s) = sum_chat p(z|chat) p(chat|s)
        weights = cell_weights
        conds = (weights != 0) & np.all(np.isfinite(hists), axis=1)
        hist = np.sum((hists[conds] * weights[conds, None]), axis=0)

        dx = np.diff(bins)
        normalization = np.sum(dx * hist)
        if normalization != 0:
            hist = hist / normalization
        return hist

    def get_deep_histograms(self, key, cells, overlap_weighted_pzc, bins, overlap_key= 'overlap_weight', interpolate_kwargs={}):
        """Return individual deep histograms for each cell. Can interpolate for empty cells.

        Parameters
        ----------
        key   : Parameter to extract from dataframe
        cells : A list of deep cells to return sample from, or a single int.
        overlap_weighted_pzc : Use overlap_weights in p(z|c) histogram if True. Also required if you want to bin conditionalize
        overlap_key : column name for the overlap weights in the dataframe, default to 'overlap_weight'
        bins  : Bins we histogram the values into
        interpolate_kwargs : arguments to pass in for performing interpolation
        between cells for redshift hists using a 2d gaussian of sigma
        scale_length out to max_length cells away. The two kwargs are:
        'scale_length' and 'max_length'
        data : subset of redshift information to conditionalize on. i.e. for use with computing p(z|c,is_in_data)
        Returns
        -------
        hists : a histogram of the values from self.data[key] for each deep cell
        """

        if len(interpolate_kwargs) > 0:
            cells_keep = cells
            cells = np.arange(self._number_deep_cells)
        else:
            cells_keep = cells

        hists = []
        missing_cells = []
        populated_cells = []
        for ci, c in enumerate(cells):
            try:
                df = self._deep_groups.get_group(c)
                if type(key) is str:
                    z = df[key].values
                    if(overlap_weighted_pzc==True):
                        #print("WARNING: You are using a deprecated point estimate Z. No overlap weighting enabled. You're on your own now.")#suppress
                        weights = df[overlap_key].values
                    else:
                        weights = np.ones(len(z))
                    hist = np.histogram(z, bins, weights=weights, density=True)[0] #make weighted histogram by overlap weights
                    populated_cells.append([ci, c])
                elif type(key) is list:
                    # use full p(z)
                    assert(bins is not None)
                    hist = self.histogram_from_fullpz(df, key, overlap_weighted=overlap_weighted_pzc, bin_edges=bins)
                hists.append(hist)
            except KeyError:
                missing_cells.append([ci, c])
                hists.append(np.zeros(len(bins) - 1))
        hists = np.array(hists)

        if len(interpolate_kwargs) > 0:
            # print('Interpolating {0} missing histograms'.format(len(missing_cells)))
            missing_cells = np.array(missing_cells)
            populated_cells = np.array(populated_cells)
            hist_conds = np.isin(cells, populated_cells[:, 1]) & np.all(np.isfinite(hists), axis=1)
            for ci, c in missing_cells:
                if c not in cells_keep:
                    # don't worry about interpolating cells we won't use anyways
                    continue

                central_index = np.zeros(len(self.deep_som.map_shape), dtype=int)
                unravel_index(c, self.deep_som.map_shape, central_index)  # fills central_index
                cND = np.zeros(len(self.deep_som.map_shape), dtype=int)
                weight_map = np.zeros(self._number_deep_cells)
                gaussian_rbf(weight_map, central_index, cND, self.deep_som.map_shape, **interpolate_kwargs)  # fills weight_map
                hists[ci] =  np.sum(hists[hist_conds] * (weight_map[hist_conds] / weight_map[hist_conds].sum())[:, None], axis=0)

            # purge hists back to the ones we care about
            hists = hists[cells_keep]

        return hists

    def histogram_deep(self, key, cells, cell_weights, overlap_weighted, bins=None, interpolate_kwargs={}):
        """Return histogram from values that live in specified deep cells

        Parameters
        ----------
        key                : Parameter(s) to extract from dataframe
        cells              : A list of deep cells to return sample from, or a single int.
        cell_weights       : How much we weight each wide cell. This is the array p(c | sample)
        overlap_weighted   : If True, deep cell histogram is weighted by stored overlap_weight
        bins               : Bins we histogram the values into
        interpolate_kwargs : arguments to pass in for performing interpolation between cells for redshift hists using a 2d gaussian of sigma scale_length out to max_length cells away. The two kwargs are: 'scale_length' and 'max_length'

        Returns
        -------
        hist : a histogram of the values from self.data[key]

        """
        # p(z|c)
        hists = self.get_deep_histograms(key=key, cells=cells, overlap_weighted_pzc=overlap_weighted, bins=bins, interpolate_kwargs=interpolate_kwargs)
        # p(z|s) = sum_c p(z|c) p(c|s)
        weights = cell_weights
        conds = (weights != 0) & np.all(np.isfinite(hists), axis=1)
        hist = np.sum((hists[conds] * weights[conds, None]), axis=0)
        dx = np.diff(bins)
        normalization = np.sum(dx * hist)
        if normalization != 0:
            hist = hist / normalization
        return hist

    def histogram(self, key, cells, cell_weights, overlap_weighted_pzc, bins=None, individual_chat=False, interpolate_kwargs={}):
        """Return histogram from values that live in specified wide cells by querying deep cells that contribute

        Parameters
        ----------
        key                  : Parameter(s) to extract from dataframe
        cells                : A list of wide cells to return sample from, or a single int.
        cell_weights         : How much we weight each wide cell. This is the array p(chat | sample)
        overlap_weighted_pzc : Weight contribution of galaxies within c by overlap_weight, if True. Weighting for p(c|chat) is done using stored transfer matrix.
        bins                 : Bins we histogram the values into
        individual_chat      : If True, compute p(z|chat) for each individual cell in cells. If False, compute a single p(z|{chat}) for all cells.
        interpolate_kwargs   : arguments to pass in for performing interpolation between cells for redshift hists using a 2d gaussian of sigma scale_length out to max_length cells away. The two kwargs are: 'scale_length' and 'max_length'

        Returns
        -------
        hist : a histogram of the values from self.data[key]

        Notes
        -----
        This method tries to marginalize wide assignments into what deep assignments it has

        """
        # get sample, p(z|c)
        all_cells = np.arange(self._number_deep_cells)
        hists_deep = self.get_deep_histograms(key=key, cells=all_cells, overlap_weighted_pzc=overlap_weighted_pzc, bins=bins, interpolate_kwargs=interpolate_kwargs)
        if individual_chat: # then compute p(z|chat) for each individual cell in cells and return histograms
            hists = []
            for i, (cell, cell_weight) in enumerate(zip(cells, cell_weights)):
                # p(c|chat,s)p(chat|s) = p(c,chat|s)
                possible_weights = self.pcchat[:, [cell]] * np.array([cell_weight])[None]  # (n_deep_cells, 1)
                # sum_chat p(c,chat|s) = p(c|s)
                weights = np.sum(possible_weights, axis=-1)
                conds = (weights != 0) & np.all(np.isfinite(hists_deep), axis=1)
                # sum_c p(z|c) p(c|s) = p(z|s)
                hist = np.sum((hists_deep[conds] * weights[conds, None]), axis=0)

                dx = np.diff(bins)
                normalization = np.sum(dx * hist)
                if normalization != 0:
                    hist = hist / normalization
                hists.append(hist)
            return hists
        else: # compute p(z|{chat}) and return histogram
            # p(c|chat,s)p(chat|s) = p(c,chat|s)
            possible_weights = self.pcchat[:, cells] * cell_weights[None]  # (n_deep_cells, n_cells)
            # sum_chat p(c,chat|s) = p(c|s)
            weights = np.sum(possible_weights, axis=-1)
            conds = (weights != 0) & np.all(np.isfinite(hists_deep), axis=1)
            # sum_c p(z|c) p(c|s) = p(z|s)
            hist = np.sum((hists_deep[conds] * weights[conds, None]), axis=0)

            dx = np.diff(bins)
            normalization = np.sum(dx * hist)
            if normalization != 0:
                hist = hist / normalization
            return hist

    @classmethod
    def get_cell_weights(cls, data, overlap_weighted, key):
        """Given data, get cell weights and indices

        Parameters
        ----------
        data :  Dataframe we extract parameters from
        overlap_weighted : If True, use mean overlap weights of cells.
        key :   Which key we are grabbing

        Returns
        -------
        cells :         The names of the cells
        cell_weights :  The fractions of the cells
        """
        if(overlap_weighted):
            cws = data.groupby(key)['overlap_weight'].sum()
        else:
            cws = data.groupby(key).size()

        cells = cws.index.values.astype(int)
        cws = cws / cws.sum()

        cell_weights = cws.values
        return cells, cell_weights

    def get_cell_weights_deep(self, data, overlap_weighted_pc, cell_key='cell_deep', force_assignment=True, **kwargs):
        """Given data, get cell weights p(c) and indices from deep SOM

        Parameters
        ----------
        data :  Dataframe we extract parameters from
        overlap_weighted_pc : If True, return weights proportional to the sum of overlap weights in a deep cell (rather than number of objects).
        cell_key :   Which key we are grabbing. Default: cell_deep
        force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True

        Returns
        -------
        cells :         The names of the cells
        cell_weights :  The fractions of the cells
        """
        if force_assignment:
            data[cell_key] = self.assign_deep(data, **kwargs)
        return self.get_cell_weights(data, overlap_weighted_pc, cell_key)

    def get_cell_weights_wide(self, data, overlap_weighted_pchat, cell_key='cell_wide', force_assignment=True, **kwargs):
        """Given data, get cell weights p(chat) and indices from wide SOM

        Parameters
        ----------
        data             : Dataframe we extract parameters from
        overlap_weighted_pchat : If True, use mean overlap weights of wide cells in p(chat)
        cell_key         : Which key we are grabbing. Default: cell_wide
        force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True

        Returns
        -------
        cells        :  The names of the cells
        cell_weights :  The fractions of the cells
        """
        if force_assignment:
            data[cell_key] = self.assign_wide(data, **kwargs)
        return self.get_cell_weights(data, overlap_weighted_pchat, cell_key)

    def define_tomo_bins_deep(self, data, overlap_weighted, n_bins=5, key='Z', cell_key='cell_deep', force_assignment=True, from_val=None, fullpzbins = np.arange(-0.005, 6.01, 0.01), interpolate_kwargs={}):
        """Returns which wide bins go into which tomographic sample. We order sample by key and the add cells until we have 1 / n_bins of the sample.

        Parameters
        ----------
        data :      Data sample of interest with deep data
        overlap_weighted : Use overlap weights for tomo bin definition, i.e. in p(z|c)
        n_bins :    Number of tomographic bins
        key :       Key that we use to order cells
        cell_key :   Which key we are grabbing. Default: cell_deep
        force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True
        from_val :  Minimum value of binning
        interpolate_kwargs : arguments to pass in for performing interpolation between cells for mean spec redshift using a 2d gaussian of sigma scale_length out to max_length cells away. The two kwargs are: 'scale_length' and 'max_length'

        Returns
        -------
        deep_bins : A dictionary of deep cell assignments

        """

        cell_indices = np.arange(self._number_deep_cells)  # this can probably be done in a smarter fashion
        cell_assignments = np.zeros(self._number_deep_cells, dtype=int) - 1

        # get mean z of spec data
        spec_cells = self._deep_groups.size().index.values
        if type(key) is str:
            #if(overlap_weighted==True): #warning suppressed
                #print("WARNING: You are using the deprecated point-estimate Z. Overlap weighting not implemented. You're on your own now.")
            spec_cells_z = self._deep_groups.agg('mean')[key].values
            spec_z = np.zeros(self._number_deep_cells) + np.nan
            spec_z[spec_cells] = spec_cells_z
        elif type(key) is list:
            spec_cells_pz = self.get_deep_histograms(key, spec_cells, overlap_weighted_pzc=overlap_weighted, bins=fullpzbins)
            #spec_cells_z = np.array([np.sum(fullpzbins * hist) / np.sum(hist) if (np.sum(hist) > 0) else np.nan for hist in spec_cells_pz])
            spec_cells_z = np.array([np.sum((hist / np.sum(hist)) * (fullpzbins[1:] + fullpzbins[:-1]) / 2.) if (np.sum(hist) > 0) else np.nan for hist in spec_cells_pz])
            spec_z = np.zeros(self._number_deep_cells) + np.nan
            spec_z[spec_cells] = spec_cells_z

        if len(spec_cells) != self._number_deep_cells:
            print('Warning! We have {0} deep cells, but our spec sample only occupies {1}! We are {3} {2} cells'.format(self._number_deep_cells, len(spec_cells), self._number_deep_cells - len(spec_cells), ['cutting out', 'interpolating'][len(interpolate_kwargs) > 0]))
            if len(interpolate_kwargs) > 0:
                # get which cells are missing spec
                missing_cells = cell_indices[np.isin(cell_indices, spec_cells, invert=True)]
                for c in missing_cells:
                    central_index = np.zeros(len(self.deep_som.map_shape), dtype=int)
                    unravel_index(c, self.deep_som.map_shape, central_index)
                    cND = np.zeros(len(self.deep_som.map_shape), dtype=int)
                    weight_map = np.zeros(self._number_deep_cells)
                    gaussian_rbf(weight_map, central_index, cND, self.deep_som.map_shape, **interpolate_kwargs)
                    spec_z[c] = np.sum(spec_z[spec_cells] * weight_map[spec_cells] / weight_map[spec_cells].sum())

        # get occupation of cells from your data
        sample_cells, sample_cell_weights = self.get_cell_weights_deep(data, overlap_weighted_pc=overlap_weighted, cell_key=cell_key, force_assignment=force_assignment)
        # OK to not be overlap_weighted - will only use for occupation statistics
        sample_occupation = np.zeros(self._number_deep_cells)
        sample_occupation[sample_cells] = sample_cell_weights
        # OK to not be overlap_weighted - will only use for occupation statistics

        # rank sort by mean z
        ordering_all = np.argsort(spec_z)  # nan to go end of the list
        # cut from ordering the nans
        ordering = ordering_all[np.isfinite(spec_z[ordering_all])]
        if from_val != None:
            cells_in_bin_0 = ordering[spec_z[ordering] < from_val]
            cell_assignments[cells_in_bin_0] = 0
            ordering = ordering[spec_z[ordering] >= from_val]

        # cumsum the occupation
        cumsum_occupation = np.cumsum(sample_occupation[ordering])
        # OK to not be overlap_weighted - will only use for occupation statistics
        if cumsum_occupation[-1] < 1.:
            print('Warning! We only have {0} of the sample in {1} cells with spec_z.'.format(cumsum_occupation[-1], len(ordering)))
            cumsum_occupation = cumsum_occupation / cumsum_occupation[-1]
        ordered_indices = cell_indices[ordering]

        if from_val==None:
            j=0
        else:
            j=1
        # assign to groups based on percentile
        for i in np.arange(j, n_bins, 1):
            lower = (i-j) / (n_bins-j)
            upper = (i + 1-j) / (n_bins-j)
            conds = (cumsum_occupation >= lower) * (cumsum_occupation <= upper)
            if upper==1:
                conds = (cumsum_occupation >= lower)
            cells_in_bin = ordered_indices[conds]
            cell_assignments[cells_in_bin] = i

        # convert into tomo_bins
        tomo_bins = {}
        for i in np.unique(cell_assignments):
            tomo_bins[i] = np.where(cell_assignments == i)[0]
        return tomo_bins

    def define_tomo_bins_wide(self, deep_bins, dfilter=0.0):
        """Returns which wide bins go into which tomographic sample.

        Parameters
        ----------
        deep_bins : A dictionary of deep cell assignments
        dfilter :   Require the probability of belonging to this bin to be
                    dfilter more likely than the second most likely bin.
                    Default is to disable this with dfilter = 0.0

        Returns
        -------
        wide_bins : Same as deep, only for wide assignments

        Notes
        -----
        For each deep bin, calculates sum_{c \in bin b} p(c | chat)
        chat then goes into bin b with largest value
        """

        keys = deep_bins.keys()
        # sum_{c\in bin} p(c|chat)
        probabilities = []
        for key in keys:
            cells = deep_bins[key]
            prob = np.sum(self.pcchat[cells], axis=0)
            probabilities.append(prob)
        probabilities = np.array(probabilities)

        # dfilter
        sorted_probabilities = np.sort(probabilities, axis=0)
        dprob = sorted_probabilities[-1] - sorted_probabilities[-2]

        # group assignments
        # binhat = argmax_bin sum_{c\in bin} p(c|chat)
        assignments = np.argmax(probabilities, axis=0)
        wide_bins = {}
        for key_i, key in enumerate(keys):
            wide_bins[key] = np.where((assignments == key_i) * (dprob >= dfilter))[0]
        return wide_bins

    def define_tomo_bins_wide_pz(self, data, overlap_weighted_pchat, overlap_weighted_pzc, bins, n_bins=5, key='Z', cell_key='cell_wide', force_assignment=True, from_val=None, interpolate_kwargs={}, **kwargs):
        """Returns which wide bins go into which tomographic sample.

        Parameters
        ----------
        data    :   Sample for which we want approximately equal fractions in each tomographic bin.
        overlap_weighted_pchat  : If True, use overlap_weight for p(chat).
        overlap_weighted_pzc    : If True, use overlap_weight for p(z|c).

        Returns
        -------
        wide_bins : Same as deep, only for wide assignments

        Notes
        -----
        Calculate
        """

        hists = []
        # compute p(z|chat)
        cells, cell_weights = self.get_cell_weights_wide(data, overlap_weighted_pchat=overlap_weighted_pchat, cell_key=cell_key, force_assignment=force_assignment, **kwargs)
        cell_indices = np.arange(len(cells)) # this can probably be done in a smarter fashion
        cell_assignments = np.zeros(len(cells), dtype=int) - 1
        hists = self.histogram(key, cells, cell_weights, overlap_weighted_pzc=overlap_weighted_pzc, bins=bins, individual_chat=True, interpolate_kwargs=interpolate_kwargs)

        # compute mean(z)|chat
        meanz = np.array([np.sum((hist / np.sum(hist)) * (bins[1:] + bins[:-1]) / 2.) if (np.sum(hist) > 0) else np.nan for hist in hists])

        # get occupation of cells from your data
        sample_occupation = np.zeros(len(cells)) # np.zeros(self._number_wide_cells) # allow cell -1
        sample_occupation[cells] = cell_weights

        # rank sort by mean z
        ordering_all = np.argsort(meanz)  # nan to go end of the list

        # cut from ordering the nans
        ordering = ordering_all[np.isfinite(meanz[ordering_all])]
        if from_val != None:
            cells_in_bin_0 = ordering[meanz[ordering] < from_val]
            cell_assignments[cells_in_bin_0] = 0
            ordering = ordering[meanz[ordering] >= from_val]

        # cumsum the occupation
        cumsum_occupation = np.cumsum(sample_occupation[ordering])
        ordered_indices = cell_indices[ordering]

        if from_val==None:
            j=0
        else:
            j=1
        # assign to groups based on percentile
        for i in np.arange(j, n_bins, 1):
            lower = (i-j) / (n_bins-j)
            upper = (i + 1-j) / (n_bins-j)
            conds = (cumsum_occupation >= lower) * (cumsum_occupation <= upper)
            if upper==1:
                conds = (cumsum_occupation >= lower)
            cells_in_bin = ordered_indices[conds]
            cell_assignments[cells_in_bin] = i

        # convert into tomo_bins
        tomo_bins = {}
        for i in np.unique(cell_assignments):
            tomo_bins[i] = np.where(cell_assignments == i)[0]
        return tomo_bins

    def redshift_distributions_wide(self, data, overlap_weighted_pchat, overlap_weighted_pzc, bins, tomo_bins={}, key='Z', force_assignment=True, interpolate_kwargs={}, **kwargs):
        """Returns redshift distribution for sample

        Parameters
        ----------
        data :      Data sample of interest with wide data
        overlap_weighted_pchat  : If True, use overlap weights for p(chat)
        overlap_weighted_pzc : If True, use overlap weights for p(z|c)
                    Note that whether p(c|chat) is overlap weighted depends on how you built pcchat earlier.
        bins :      bin edges for redshift distributions data[key]
        tomo_bins : Which cells belong to which tomographic bins. First column is
                    cell id, second column is an additional reweighting of galaxies in cell.
                    If nothing is passed in, then we by default just use all cells
        key :       redshift key
        force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True
        interpolate_kwargs : arguments to pass in for performing interpolation
        between cells for redshift hists using a 2d gaussian of sigma
        scale_length out to max_length cells away. The two kwargs are:
        'scale_length' and 'max_length'

        Returns
        -------
        hists : Either a single array (if no tomo_bins) or multiple arrays

        """
        if len(tomo_bins) == 0:
            cells, cell_weights = self.get_cell_weights_wide(data, overlap_weighted_pchat=overlap_weighted_pchat, force_assignment=force_assignment, **kwargs)
            if cells.size == 0:
                hist = np.zeros(len(bins) - 1)
            else:
                hist = self.histogram(key=key, cells=cells, cell_weights=cell_weights, overlap_weighted_pzc=overlap_weighted_pzc, bins=bins, interpolate_kwargs=interpolate_kwargs)
            return hist
        else:
            cells, cell_weights = self.get_cell_weights_wide(data, overlap_weighted_pchat, force_assignment=force_assignment, **kwargs)
            cellsort = np.argsort(cells)
            cells = cells[cellsort]
            cell_weights = cell_weights[cellsort]

            # break up hists into the different bins
            hists = []
            for tomo_key in tomo_bins:
                cells_use     = tomo_bins[tomo_key][:,0]
                cells_binweights = tomo_bins[tomo_key][:,1]
                cells_conds   = np.searchsorted(cells, cells_use,side='left')
                if len(cells_conds) == 0:
                    hist = np.zeros(len(bins) - 1)
                else:
                    hist = self.histogram(key=key, cells=cells[cells_conds], cell_weights=cell_weights[cells_conds]*cells_binweights, overlap_weighted_pzc=overlap_weighted_pzc, bins=bins, interpolate_kwargs=interpolate_kwargs)
                hists.append(hist)
            hists = np.array(hists)
            return hists

    def redshift_distributions_deep(self, data, bins, overlap_weighted_pc, overlap_weighted_pzc, tomo_bins={}, key='Z', force_assignment=True, interpolate_kwargs={}, **kwargs):
        """Returns redshift distribution for sample defined by set of deep cells, from deep information

        Parameters
        ----------
        data :      Data sample of interest with wide data
        bins :      bin edges for redshift distributions data[key]
        overlap_weighted_pc  : If True, weight contributing cells by their mean overlap_weight
        overlap_weighted_pzc : If True, weight galaxy contribution within c by their overlap_weight
        tomo_bins : Which cells belong to which tomographic bins. First column is
                    cell id, second column is an additional reweighting of galaxies in cell.
                    If nothing is passed in, then we by default just use all cells
        key :       redshift key
        force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True
        interpolate_kwargs : arguments to pass in for performing interpolation
        between cells for redshift hists using a 2d gaussian of sigma
        scale_length out to max_length cells away. The two kwargs are:
        'scale_length' and 'max_length'

        Returns
        -------
        hists : Either a single array (if no tomo_bins) or multiple arrays

        """
        if len(tomo_bins) == 0:
            # given data, gets cell weights, returns histogram
            cells, cell_weights = self.get_cell_weights_deep(data, overlap_weighted_pc=overlap_weighted_pc, force_assignment=force_assignment, **kwargs)
            if cells.size == 0:
                hist = np.zeros(len(bins) - 1)
            else:
                hist = self.histogram_deep(key=key, cells=cells, cell_weights=cell_weights, overlap_weighted=overlap_weighted_pzc, bins=bins, interpolate_kwargs=interpolate_kwargs)
            return hist
        else:
            cells, cell_weights = self.get_cell_weights_deep(data, overlap_weighted_pc=overlap_weighted_pc, force_assignment=force_assignment, **kwargs)
            cellsort = np.argsort(cells)
            cells = cells[cellsort]
            cell_weights = cell_weights[cellsort]

            hists = []
            for tomo_key in tomo_bins:
                cells_use     = tomo_bins[tomo_key][:,0]
                cells_binweights = tomo_bins[tomo_key][:,1]
                cells_conds   = np.searchsorted(cells, cells_use, side='left')

                if cells[cells_conds].size == 0:
                    hist = np.zeros(len(bins) - 1)
                else:
                    hist = self.histogram_deep(key=key, cells=cells[cells_conds], cell_weights=cell_weights[cells_conds]*cells_binweights, overlap_weighted=overlap_weighted_pzc, bins=bins, interpolate_kwargs=interpolate_kwargs)
                hists.append(hist)
            hists = np.array(hists)
            return hists

    def redshift_distributions_true(self, data, bins, tomo_bins={}, key='Z', overlap_weight = True, overlap_key='overlap_weight', force_assignment=True, cell_key='cell_deep', **kwargs):
        """Returns redshift distribution for sample from true information. Basically just a wrapper for histogram_true and the numpy histogram function

        Parameters
        ----------
        data :      Data sample of interest with wide data
        bins :      bin edges for redshift distributions data[key]
        tomo_bins : Which cells belong to which tomographic bins. First column is
                    cell id, second column is an additional reweighting of galaxies in cell.
                    If nothing is passed in, then we by default just use all cells
        key :       redshift key
        overlap_weight : boolean, use shear response or injection number as weighted in dataframe
        overlap_key : key for weighting of histogram, default to 'overlap_weight', ignore if overlap_weight = False
        force_assignment : Calculate cell assignments. If False, then will use whatever value is in the cell_key field of data. Default: True
        cell_key :   Which key we are grabbing. Default: cell_deep

        Returns
        -------
        hists : Either a single array (if no tomo_bins) or multiple arrays

        """
        if len(tomo_bins) == 0:
            if overlap_weight == True:
                hist = self.histogram_true(data[key].values, bins, weights=data[overlap_key].values)
            else:
                hist = self.histogram_true(data[key].values, bins)
            return hist
        else:
            # get cell assignments from data, break up, make hists
            hists = []
            if force_assignment:
                # try to be smart about what we really mean by "force_assignment"
                if 'wide' in cell_key:
                    cells = self.assign_wide(data, **kwargs)
                elif 'deep' in cell_key:
                    cells = self.assign_deep(data, **kwargs)
                else:
                    cells = self.assign_deep(data, **kwargs)
            elif cell_key in data.columns:
                cells = data[cell_key].values
            else:
                if 'wide' in cell_key:
                    cells = self.assign_wide(data, **kwargs)
                elif 'deep' in cell_key:
                    cells = self.assign_deep(data, **kwargs)
                else:
                    cells = self.assign_deep(data, **kwargs)
            for tomo_key in tomo_bins:
                cellsort = np.argsort(tomo_bins[tomo_key][:,0])
                cells_use     = tomo_bins[tomo_key][cellsort,0]
                cells_binweights = tomo_bins[tomo_key][cellsort,1]
                cells_conds_left  = np.searchsorted(cells_use, cells, side='left') # index of used cell / binweight for each galaxy
                cells_conds_right = np.searchsorted(cells_use, cells, side='right')
                cells_conds = (cells_conds_right-cells_conds_left==1) # True are the galaxies that are in the used cells

                try:
                    if overlap_weight == True:
                        zs = data[key].values[cells_conds]
                        weights = data[overlap_key].values[cells_conds]*cells_binweights[cells_conds_left]
                        hist = self.histogram_true(zs, bins, weights=weights)
                    else:
                        zs = data[key].values[cells_conds]
                        weights = cells_binweights[cells_conds_left]
                        hist = self.histogram_true(zs, bins, weights=weights)
                except KeyError:
                    hist = np.zeros(len(bins) - 1)
                hists.append(hist)
            hists = np.array(hists)
            return hists

    """
    # source: Alex Alarcon.
    # hacked to work with overlap weights.
    """
    def nz_bin_conditioned(self, wfdata, overlap_weighted_pchat, overlap_weighted_pzc, tomo_cells, zbins, cell_wide_key='cell_wide', zkey='Z'):
        """ Function to obtain p(z|bin,s): the redshift distribution of a tomographic bin
        including the tomographic selection effect in p(z|chat).

        Implementation note:
        This is going to sneak the bin conditionalization into the overlap weights, and then divide them back out.
        This is a simple way of achieving to not completely lose cells c that contribute to p(c|chat) but don't have a z in b.
        Not the cleanest code written by a human.

            Parameters
            ----------
            wfdata : Wide field data
            overlap_weighted_pchat : If True, weight chat by the sum of overlap weights, not number of galaxies, in wide field data.
            tomo_cells : Which cells belong to this tomographic bin. First column is
                         cell id, second column is an additional reweighting of galaxies in that cell.
            zbins : redshift bin edges.
            cell_wide_key : key for wide SOM cell id information in spec_data.
            cell_deep_key : key for wide SOM cell id information in spec_data.
            #cells : A list of deep cells to return sample from, or a single int.
            #cell_weights : How much we weight each wide cell. This is the array p(c | sample)
        """
        print('full redshift sample:', len(self.data))
        #print('cell_wide_key: ', cell_wide_key)
        #print('self.data[cell_wide_key].shape', self.data[cell_wide_key].shape)
        #print('tomo_cells', tomo_cells)
        bl=len(self.data[self.data[cell_wide_key].isin(tomo_cells[:,0])])
        
        print('subset of reshift sample in bin:', bl)

        f=1.e9 # how much more we weight the redshift of a galaxy that's in the right bin

        stored_overlap_weight = self.data['overlap_weight'].copy() # save for later

        if(overlap_weighted_pzc == False): # we need to use it, but you don't want to
            self.data['overlap_weight'] = np.ones(len(self.data))

        self.data.loc[self.data[cell_wide_key].isin(tomo_cells[:,0]),'overlap_weight'] *= f

        nz = self.redshift_distributions_wide(data=wfdata, overlap_weighted_pchat=overlap_weighted_pchat, overlap_weighted_pzc=True,
                                         bins=zbins, tomo_bins={"mybin" : tomo_cells}, key=zkey, force_assignment=False, cell_key=cell_wide_key)

        self.data['overlap_weight'] = stored_overlap_weight.copy() # open jar

        return nz[0]

    def redshift_histograms_stats(self, hists_true, hists_estimated, bins, legend_estimated):
        """Compute some statistics for the set of (true, estimated) histograms in each tomographic bin
        Parameters
        ----------
        hists_true :       An array of normalized histogramms for the different tomographic bins (the truth)
        hists_estimated :  An array of normalized histogramms for the different tomographic bins (the estimation)
        bins :             The bins corresponding to hists_true and hists_estimated
        legend_estimated:  The legend of the estimated histogram, must be 'deep' or 'wide'.
        Returns
        -------
        results :          Normalisation, mean, sigma for each tomographic bin for truth and estimated
        deltas :           A DataFrame containing for each bin the difference between the truth and estimation in mean z and sigma z
        """
        import pandas as pd

        if hists_true.ndim == 2:
            if hists_true.shape[0] != hists_estimated.shape[0]:
                raise ValueError('hists_true must contain the same number of histograms as hists_estimated')
            if hists_true.shape[1] != hists_estimated.shape[1]:
                raise ValueError('hists_true must have the same number of bins as hists_estimated')
            if hists_true.shape[1] != (bins.shape[0]-1):
                raise ValueError('the number of bins must correspond to the length of the histograms')
            #Ignore the empty histogramms
            non_zero_true = (hists_true.sum(axis=1) != 0)
            non_zero_estimated = (hists_estimated.sum(axis=1) != 0)
            hists_true = hists_true[non_zero_true * non_zero_estimated]
            hists_estimated = hists_estimated[non_zero_true * non_zero_estimated]
        elif hists_true.ndim == 1:
            if hists_true.shape[0] != hists_estimated.shape[0]:
                raise ValueError('hists_true must have the same number of bins as hists_estimated')
            if hists_true.shape[0] != (bins.shape[0]-1):
                raise ValueError('the number of bins must correspond to the length of the histograms')
            hists_true = [hists_true]
            hists_estimated = [hists_estimated]
        else:
            raise ValueError('hists_true has not the correct dimension')

        results = {'norm': [], 'mean': [], 'sigma': [], 'label': [], 'tomo' : []}

        for tomo, (hist_true_deep, hist_deep) in enumerate(zip(hists_true, hists_estimated)):
            # color = 'C{0}'.format(tomo)
            for hist, label in zip([hist_true_deep, hist_deep], ['true', legend_estimated]):
                norm, mean, sigma = self.one_point_statistics(hist, bins)
                results['norm'].append(norm)
                results['mean'].append(mean)
                results['sigma'].append(sigma)
                results['label'].append(label)
                results['tomo'].append(tomo)
        results = pd.DataFrame(results)

        delta_mean_z = results[results['label']== 'true']['mean'].values -  results[results['label'] == legend_estimated]['mean'].values
        delta_sigma_z = results[results['label']== 'true']['sigma'].values -  results[results['label'] == legend_estimated]['sigma'].values
        deltas = pd.concat((pd.Series(delta_mean_z, name='delta <z>'), pd.Series(delta_sigma_z, name='delta sigma(z)')), axis=1)

        return results, deltas


    @staticmethod
    def get_mean_sigma(zmeans, hists):
        """Returns means and sigmas for each tomo bin
        """
        means = np.zeros(4)
        sigmas = np.zeros(4)

        for i in range(4):
            means[i] = np.sum(hists[i]*zmeans)/np.sum(hists[i])
            sigmas[i] = np.sqrt(np.sum(hists[i]*(zmeans-means[i])**2)/np.sum(hists[i]))

        return means,sigmas

    @staticmethod
    def pileup(hists,zs,zmeans,z_pileup,dz,weight,nbins):
        """Cuts off z, zmean and Nz at a pileup-z, and stacks tail on pileup-z and renormalises"""
        ## Pile up very high z in last bin
        import copy
        #print(hists)
        hists_piled=copy.copy(hists)
        zbegin=int(z_pileup/dz)
        print("Dz, new-end-z,weight: ", dz, z_pileup,weight)
        for b in range(nbins):
            s=np.sum(hists[b,zbegin:])
            hists_piled[b,zbegin-1]+=s*weight
            hists_piled[b,zbegin:]=0.

        #print(hists_piled)
        zs=zs[:zbegin+1]
        zmeans_piled=zmeans[:zbegin]
        hists_piled=hists_piled[:,:zbegin]
        #print(hists_piled)

        for b in range(nbins):
            hists_piled[b,:]=hists_piled[b,:]/np.sum(hists_piled[b,:]*dz)

        #print(hists_piled)
        return zs, zmeans_piled, hists_piled

    @staticmethod
    def to2point(lastnz, templatef, runname,label,data_dir):
        from scipy.stats import norm
        import twopoint
        import scipy.interpolate as interp
        from scipy.signal import savgol_filter
        import fitsio

        outfile=data_dir+'2pt_'+label+'_'+runname+'.fits'

        #open-up the saved final fits
        nz=fitsio.read(lastnz)

        #open the template
        oldnz=twopoint.TwoPointFile.from_fits(templatef)

        #puts the nzs into 2pt file
        bins=['BIN1','BIN2','BIN3','BIN4']
        for i,bin in enumerate(bins):
            #print(oldnz.kernels[0].nzs[i])
            oldnz.kernels[0].zlow = nz['Z_LOW']
            oldnz.kernels[0].z = nz['Z_MID']
            oldnz.kernels[0].zhigh = nz['Z_HIGH']
            oldnz.kernels[0].nzs[i] = nz[bin]
            #print(oldnz.kernels[0].nzs[i])
        oldnz.to_fits(outfile,clobber=True,overwrite=True)

    def smooth(self,twoptfile, nzsmoothfile, runname,label,data_dir, oldnz):
        from scipy.stats import norm
        import twopoint
        import scipy.interpolate as interp
        from scipy.signal import savgol_filter
        import matplotlib.pyplot as plt

        outfilesmooth=data_dir+'2pt_'+label+'_'+runname+'_smooth.fits'

        #Troxel's smoothing adapted
        nosmooth=twopoint.TwoPointFile.from_fits(twoptfile)
        z = nosmooth.kernels[0].z
        for i in range(4):
            b = savgol_filter(nosmooth.kernels[0].nzs[i],25,2)
            f = interp.interp1d(nosmooth.kernels[0].z,b,bounds_error=False,fill_value=0.)
            nosmooth.kernels[0].nzs[i] = f(z)
        nosmooth.to_fits(outfilesmooth,clobber=True,overwrite=True)
        np.savetxt(nzsmoothfile,np.vstack((nosmooth.kernels[0].zlow,nosmooth.kernels[0].nzs[0],nosmooth.kernels[0].nzs[1],nosmooth.kernels[0].nzs[2],nosmooth.kernels[0].nzs[3])).T)

        oldnz=twopoint.TwoPointFile.from_fits(twoptfile)
        means_smooth, sigmas_smooth = self.get_mean_sigma(nosmooth.kernels[0].z, nosmooth.kernels[0].nzs)
        means_bc_piled, sigmas_bc_piled = self.get_mean_sigma(oldnz.kernels[0].z, oldnz.kernels[0].nzs)

        plt.figure(figsize=(16.,9.))
        colors=['blue','orange','green','red']
        for i in range(4):
            plt.fill_between(oldnz.kernels[0].z, oldnz.kernels[0].nzs[i], color= colors[i],alpha=0.3)#,label="fiducial")
            plt.axvline(means_smooth[i], linestyle='-.', color= colors[i],label=str(i)+' %.3f'%(means_smooth[i]))
            plt.plot(nosmooth.kernels[0].z, nosmooth.kernels[0].nzs[i], color= colors[i])#,label="smooth")
            plt.axvline(means_bc_piled[i], linestyle='-', color= colors[i],label=str(i)+' smooth: %.3f'%(means_bc_piled[i]))
        plt.xlabel(r'$z$')
        plt.ylabel(r'$p(z)$')
        plt.xlim(0,3)
        plt.ylim(-0.5,6)
        plt.legend(loc='upper right')
        plt.title('Wide n(z)')
        plt.savefig(data_dir+'smooth_wide_nz.png')


    def smooth_response_weight(self, snr, size_ratio,file):
        snmin=10
        snmax=300
        sizemin=0.5
        sizemax=5
        steps=20
        r = np.genfromtxt(file)
        def assign_loggrid( x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
            # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps
            x = np.maximum(x, xmin)
            x = np.minimum(x, xmax)
            y = np.maximum(y, ymin)
            y = np.minimum(y, ymax)
            logstepx = np.log10(xmax/xmin)/xsteps
            logstepy = np.log10(ymax/ymin)/ysteps
            indexx = (np.log10(x/xmin)/logstepx).astype(int)
            indexy = (np.log10(y/ymin)/logstepy).astype(int)
            indexx = np.minimum(indexx, xsteps-1)
            indexy = np.minimum(indexy, ysteps-1)
            return indexx,indexy
        def apply_loggrid(x, y, grid, xmin=snmin, xmax=snmax, xsteps=steps, ymin=sizemin, ymax=sizemax, ysteps=steps):
            indexx,indexy = assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps)
            res = np.zeros(len(x))
            res = grid[indexx,indexy]
            return res

        smoothresponse = apply_loggrid(snr, size_ratio, r)
        return smoothresponse


    def newz(self,newspec_data, cm, cfg):
        import copy
        print("length of new spec_data", len(newspec_data))
        if 'SOURCE' in newspec_data.columns:
            print(newspec_data.groupby('SOURCE')['RAspec'].nunique())
        cm_new=copy.copy(cm)
        cm_new.data=newspec_data

        wide_columns = open(cfg['wide_bands_file']).read().splitlines()
        wide_flux_columns = [b.replace('MAG', 'flux') for b in wide_columns]
        wide_flux_ivar_columns = [b.replace('flux', 'flux_ivar') for b in wide_flux_columns]
        wide_METACAL_flux_columns = wide_flux_columns
        wide_METACAL_flux_ivar_columns = wide_flux_ivar_columns
        rename_dict_flux = {col.replace("METACAL", "unsheared") : col for col in wide_METACAL_flux_columns}
        rename_dict_ivar = {col.replace("METACAL", "unsheared") : col for col in wide_METACAL_flux_ivar_columns}
        rename_dict = {}
        rename_dict.update(rename_dict_flux)
        rename_dict.update(rename_dict_ivar)
        newspec_data_renamed = newspec_data.rename(columns=rename_dict)

        cm_new.data['cell_deep'] = cm_new.assign_deep(newspec_data_renamed)
        cm_new.data['cell_wide'] = cm_new.assign_wide(newspec_data_renamed)

        cm_new._deep_groups = cm_new.data.groupby('cell_deep') #df = self._deep_groups.get_group(c)

        return cm_new

    def save_des_nz(self, hists, zbins, n_bins, outdir, run_name, suffix):
        ### output n(z) to fits  in y1 format ###

        import astropy.io.fits as fits
        import os

        bin_spacing = (zbins[1] - zbins[0]) / 2.
        z_low = zbins[:-1]
        z_mid = z_low + bin_spacing
        z_upper = zbins[1:]

        (bin_1, bin_2, bin_3, bin_4) = hists[:n_bins]

        col1 = fits.Column(name='Z_LOW', format='D', array=z_low)
        col2 = fits.Column(name='Z_MID', format='D', array=z_mid)
        col3 = fits.Column(name='Z_HIGH', format='D', array=z_upper)
        col4 = fits.Column(name='BIN1', format='D', array=bin_1)
        col5 = fits.Column(name='BIN2', format='D', array=bin_2)
        col6 = fits.Column(name='BIN3', format='D', array=bin_3)
        col7 = fits.Column(name='BIN4', format='D', array=bin_4)

        hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6, col7])

        print('mkdir -p ' + outdir)
        os.system('mkdir -p ' + outdir)
        os.system('chmod -R a+rx ' + outdir)

        nz_out = outdir + 'y3_redshift_distributions_{}_{}.fits'.format(run_name, suffix)
        print('write ' + nz_out)
        hdu.writeto(nz_out, overwrite=True)
        os.system('chmod a+r ' + nz_out)


    def summary_stats(self, data, key='Z', assign_names='cell_deep'):
        """Returns summary statistics of some data, including their mean, median, std, number, and 25th and 75th percentile.
        """
        return data.groupby(assign_names).agg(['mean', 'median', 'std', 'size', percentile(25), percentile(75)])[key]
        """Compute some statistics for the set of (true, estimated) histograms in each tomographic bin

        Parameters
        ----------
        hists_true :       An array of normalized histogramms for the different tomographic bins (the truth)
        hists_estimated :  An array of normalized histogramms for the different tomographic bins (the estimation)
        bins :             The bins corresponding to hists_true and hists_estimated
        legend_estimated:  The legend of the estimated histogram, must be 'deep' or 'wide'.

        Returns
        -------
        results :          Normalisation, mean, sigma for each tomographic bin for truth and estimated
        deltas :           A DataFrame containing for each bin the difference between the truth and estimation in mean z and sigma z
        """
        import pandas as pd

        if hists_true.ndim == 2:
            if hists_true.shape[0] != hists_estimated.shape[0]:
                raise ValueError('hists_true must contain the same number of histograms as hists_estimated')
            if hists_true.shape[1] != hists_estimated.shape[1]:
                raise ValueError('hists_true must have the same number of bins as hists_estimated')
            if hists_true.shape[1] != (bins.shape[0]-1):
                raise ValueError('the number of bins must correspond to the length of the histograms')
            #Ignore the empty histogramms
            non_zero_true = (hists_true.sum(axis=1) != 0)
            non_zero_estimated = (hists_estimated.sum(axis=1) != 0)
            hists_true = hists_true[non_zero_true * non_zero_estimated]
            hists_estimated = hists_estimated[non_zero_true * non_zero_estimated]
        elif hists_true.ndim == 1:
            if hists_true.shape[0] != hists_estimated.shape[0]:
                raise ValueError('hists_true must have the same number of bins as hists_estimated')
            if hists_true.shape[0] != (bins.shape[0]-1):
                raise ValueError('the number of bins must correspond to the length of the histograms')
            hists_true = [hists_true]
            hists_estimated = [hists_estimated]
        else:
            raise ValueError('hists_true has not the correct dimension')

        results = {'norm': [], 'mean': [], 'sigma': [], 'label': [], 'tomo' : []}

        for tomo, (hist_true_deep, hist_deep) in enumerate(zip(hists_true, hists_estimated)):
            # color = 'C{0}'.format(tomo)
            for hist, label in zip([hist_true_deep, hist_deep], ['true', legend_estimated]):
                norm, mean, sigma = self.one_point_statistics(hist, bins)
                results['norm'].append(norm)
                results['mean'].append(mean)
                results['sigma'].append(sigma)
                results['label'].append(label)
                results['tomo'].append(tomo)
        results = pd.DataFrame(results)

        delta_mean_z = results[results['label']== 'true']['mean'].values -  results[results['label'] == legend_estimated]['mean'].values
        delta_sigma_z = results[results['label']== 'true']['sigma'].values -  results[results['label'] == legend_estimated]['sigma'].values
        deltas = pd.concat((pd.Series(delta_mean_z, name='delta <z>'), pd.Series(delta_sigma_z, name='delta sigma(z)')), axis=1)

        return results, deltas

    def plot(self, key, som_keys=['cell_deep_0', 'cell_deep_1'], selection=None, **kwargs):
        """Plot for each selection and key the 1d hist and breakdown by SOM cell

        Parameters
        ----------
        key :       key of interest
        som_keys :  The two variables that define the 0-th and 1-th SOM
                    dimensions. By default, [cell_deep_0, cell_deep_1]
        selection : a list of lists of indicies of which objects in self.data
                    go into each selection. If None, then we just use the full
                    dataset.
        path :      Save the figure to this file, if specified.
        kwargs :    Any additional kwargs to go into sompz.plot_key

        Returns
        -------
        fig :       A matplotlib figure of this plot

        """
        if self.deep_som.ndim != 2:
            raise NotImplementedError("Don't know how to plot SOMs of dimensionality {0}!".format(self.deep_som.ndim))

        if selection is None:
            dfs = [self.data]
        else:
            dfs = [self.data.iloc[s] for s in selection]

        fig = plot_key(dfs=dfs, key=key, assign_names=som_keys, map_shape=self.deep_som.map_shape, **kwargs)

        return fig

    def plot_sample(self, data, key, **kwargs):
        """Plot 1d histogram and then breakdown of sample in wide and deep soms

        Parameters
        ----------
        data :      dataframe containing key of interest and cell assignments
        key :       key of interest
        kwargs :    any additional kwargs to go into sompz.plots.plot_sample

        Returns
        -------
        fig :       A matplotlib figure of this plot

        """
        z = data[key].values
        if 'cell_deep' not in data:
            deep_c = self.assign_deep(data)
        else:
            deep_c = data['cell_deep'].values
        if 'cell_wide' not in data:
            wide_c = self.assign_wide(data)
        else:
            wide_c = data['cell_wide'].values

        # cut on finite z
        conds = np.isfinite(z)
        z = z[conds]
        wide_c = wide_c[conds]
        deep_c = deep_c[conds]

        fig = plot_sample(z, wide_c, deep_c, self.wide_som, self.deep_som, key, **kwargs)

        return fig
    def plot_sample_colors(self, data, keys, **kwargs):
        """Plot 1d histogram and then breakdown of sample in wide and deep soms

        Parameters
        ----------
        data :      dataframe containing key of interest and cell assignments
        keys :      key of interest
        kwargs :    any additional kwargs to go into sompz.plots.plot_sample

        Returns
        -------
        fig :       A matplotlib figure of this plot

        """
        z = data[keys]
        if 'cell_deep' not in data:
            deep_c = self.assign_deep(data)
        else:
            deep_c = data['cell_deep'].values
        if 'cell_wide' not in data:
            wide_c = self.assign_wide(data)
        else:
            wide_c = data['cell_wide'].values

        fig = plot_sample_colors(z, wide_c, deep_c, self.wide_som, self.deep_som, keys, **kwargs)

        return fig

    def plot_som_mean_z(self, data, key, **kwargs):
        """Plot 1d histogram and then breakdown of sample in wide and deep soms

        Parameters
        ----------
        data :      dataframe containing key of interest and cell assignments
        key :       key of interest
        kwargs :    any additional kwargs to go into sompz.plots.plot_sample

        Returns
        -------
        fig :       A matplotlib figure of this plot

        """
        z = data[key].values
        if 'cell_deep' not in data:
            deep_c = self.assign_deep(data)
        else:
            deep_c = data['cell_deep'].values
        if 'cell_wide' not in data:
            wide_c = self.assign_wide(data)
        else:
            wide_c = data['cell_wide'].values

        # cut on finite z
        conds = np.isfinite(z)
        z = z[conds]
        wide_c = wide_c[conds]
        deep_c = deep_c[conds]

        fig = plot_som_mean_z(z, wide_c, deep_c, self.wide_som, self.deep_som, key, **kwargs)

        return fig

    def plot_som_diagnostic(self, data, key, **kwargs):
        """Plot 1d histogram and then breakdown of sample in wide and deep soms

        Parameters
        ----------
        data :      dataframe containing key of interest and cell assignments
        key :       key of interest
        kwargs :    any additional kwargs to go into sompz.plots.plot_sample

        Returns
        -------
        fig :       A matplotlib figure of this plot

        """
        z = data[key].values
        if 'cell_deep' not in data:
            deep_c = self.assign_deep(data)
        else:
            deep_c = data['cell_deep'].values
        if 'cell_wide' not in data:
            wide_c = self.assign_wide(data)
        else:
            wide_c = data['cell_wide'].values

        # cut on finite z
        conds = np.isfinite(z)
        z = z[conds]
        wide_c = wide_c[conds]
        deep_c = deep_c[conds]

        fig = plot_som_diagnostic(z, wide_c, deep_c, self.wide_som, self.deep_som, key, **kwargs)

        return fig

    def plot_som_diagnostic_balrog_vs_wide(self, balrog_data, wide_data, **kwargs):
        '''Plot

        Parameters
        ----------
        data :      dataframe containing key of interest and cell assignments
        key :       key of interest
        kwargs :    any additional kwargs to go into sompz.plots.plot_sample

        Returns
        -------
        fig :       A matplotlib figure of this plot

        '''
        return None

        if 'cell_wide' not in balrog_data:
            wide_c = self.assign_wide(balrog_data)
        else:
            balrog_c = balrog_data['cell_wide'].values

        if 'cell_wide' not in wide_data:
            wide_c = self.assign_wide(wide_data)
        else:
            wide_c = wide_data['cell_wide'].values

        fig = plot_som_diagnostic_balrog_vs_wide(balrog_c, wide_c,
                                                 balrog_data, wide_data,
                                                 self.wide_som, **kwargs)

        return fig

    def plot_cell(self, data, key, cell, cell_kind, pcchat = None, **kwargs):
        """Take all of sample in certain som cell and plot 1d histogram and
        breakdown of sample in other som.

        Parameters
        ----------
        data :      dataframe containing key of interest and cell assignments
        key :       key of interest
        cell :      cell number to cut on
        cell_kind : wide or deep; selects which SOM we take our cell from
        kwargs :    any additional kwargs to go into sompz.plots.plot_sample

        Returns
        -------
        fig :       A matplotlib figure of this plot

        """
        if type(key) is list:
            z = data[key]
        else:
            z = data[key].values

        if 'cell_deep' not in data:
            deep_c = self.assign_deep(data)
        else:
            deep_c = data['cell_deep'].values
        if 'cell_wide' not in data:
            wide_c = self.assign_wide(data)
        else:
            wide_c = data['cell_wide'].values

        if cell_kind == 'wide':
            conds = wide_c == cell
            other_c = deep_c[conds]
            other_som = self.deep_som
            som = self.wide_som
        elif cell_kind == 'deep':
            conds = deep_c == cell
            other_c = wide_c[conds]
            other_som = self.wide_som
            som = self.deep_som
        else:
            raise KeyError('Unacceptable cell_kind: {0}. Please choose \'wide\' or \'deep\'.'.format(cell_kind))
        z = z[conds]
        data = data[conds]

        # cut on finite z
        if type(key) is list:
            conds = np.isfinite(z['Z'])
        else:
            conds = np.isfinite(z)

        z = z[conds]
        other_c = other_c[conds]
        data = data[conds]

        wide_flux_columns = ['mcal_flux_i_out', 'mcal_flux_r_out', 'mcal_flux_z_out']
        wide_flux_ivar_columns = [b.replace('flux','flux_ivar') for b in wide_flux_columns]
        # TODO save lupt in dataframe after computing it
        wide_lupt_columns = [_.replace("flux","lupt") for _ in wide_flux_columns]

        fluxes = data[wide_flux_columns].values
        lupt_i, lupt_r, lupt_z = None, None, None #luptize_wide_y3(fluxes, 0)[0]
        data[wide_lupt_columns] = lupt_i, lupt_r, lupt_z
        fig = plot_cell(z, other_c, other_som, key, data=data,
                        c=cell, som=som, pcchat=pcchat, cell_kind=cell_kind, **kwargs)

        return fig

    '''
    def plot_cell(self, data, key, cell, cell_kind, **kwargs):
        """Take all of sample in certain som cell and plot 1d histogram and
        breakdown of sample in other som.

        Parameters
        ----------
        data :      dataframe containing key of interest and cell assignments
        key :       key of interest
        cell :      cell number to cut on
        cell_kind : wide or deep; selects which SOM we take our cell from
        kwargs :    any additional kwargs to go into sompz.plots.plot_sample

        Returns
        -------
        fig :       A matplotlib figure of this plot

        """
        z = data[key].values
        if 'cell_deep' not in data:
            deep_c = self.assign_deep(data)
        else:
            deep_c = data['cell_deep'].values
        if 'cell_wide' not in data:
            wide_c = self.assign_wide(data)
        else:
            wide_c = data['cell_wide'].values

        if cell_kind == 'wide':
            conds = wide_c == cell
            c = deep_c[conds]
            som = self.deep_som
        elif cell_kind == 'deep':
            conds = deep_c == cell
            c = wide_c[conds]
            som = self.wide_som
        else:
            raise KeyError('Unacceptable cell_kind: {0}. Please choose \'wide\' or \'deep\'.'.format(cell_kind))
        z = z[conds]

        # cut on finite z
        conds = np.isfinite(z)
        z = z[conds]
        c = c[conds]

        fig = plot_cell(z, c, som, key, **kwargs)

        return fig
    '''

    def plot_redshift_histograms(self, hists_true, hists_estimated, bins, title=None, legend_estimated='estimated', legend_true='true', max_pz=3.5, max_z=2.0):
        """Plot the set of (true, estimated) histograms in each tomographic bin

        Parameters
        ----------
        hists_true :       An array of normalized histogramms for the different tomographic bins (the truth)
        hists_estimated :  An array of normalized histogramms for the different tomographic bins (the estimation)
        bins :             The bins corresponding to hists_true and hists_estimated
        title :            The title of the figure
        legend_estimated:  The legend of the estimated histogram

        Returns
        -------
        fig :       A matplotlib figure of this plot
        """
        from matplotlib.pyplot import subplots
        if hists_true.ndim == 2:
            if hists_true.shape[0] != hists_estimated.shape[0]:
                raise ValueError('hists_true must contain the same number of histograms as hists_estimated')
            if hists_true.shape[1] != hists_estimated.shape[1]:
                raise ValueError('hists_true must have the same number of bins as hists_estimated')
            if hists_true.shape[1] != (bins.shape[0]-1):
                raise ValueError('the number of bins must correspond to the length of the histograms')
            #Ignore the empty histogramms
            non_zero_true = (hists_true.sum(axis=1) != 0)
            non_zero_estimated = (hists_estimated.sum(axis=1) != 0)
            hists_true = hists_true[non_zero_true * non_zero_estimated]
            hists_estimated = hists_estimated[non_zero_true * non_zero_estimated]
        elif hists_true.ndim == 1:
            if hists_true.shape[0] != hists_estimated.shape[0]:
                raise ValueError('hists_true must have the same number of bins as hists_estimated')
            if hists_true.shape[0] != (bins.shape[0]-1):
                raise ValueError('the number of bins must correspond to the length of the histograms')
            hists_true = [hists_true]
            hists_estimated = [hists_estimated]
        else:
            raise ValueError('hists_true does not have the correct dimensions')

        # if legend_estimated not in ['wide', 'deep']:
        #     raise ValueError('The legend_estimated must be either wide or deep.')

        fig, ax = subplots(figsize=(12, 8))

        for tomo, (hist_true, hist_estimated) in enumerate(zip(hists_true, hists_estimated)):
            color = 'C{0}'.format(tomo)
            for hist, label, linestyle in zip([hist_true, hist_estimated], [legend_true, legend_estimated], ['-', '--']):
                xtd, ytd = histogramize(bins, hist)
                if tomo == 0:
                    plotlabel = label
                else:
                    plotlabel = None
                ax.plot(xtd, ytd, color=color, linestyle=linestyle, linewidth=2, label=plotlabel)
                if label=='true':
                    ax.fill_between(xtd, 0, ytd, color=color, alpha=0.5, label=None)
        ax.set_xlabel('$z$', fontsize=20)
        ax.set_ylabel('$P(z)$', fontsize=20)
        ax.set_xlim(0, max_z)
        ax.set_ylim(0, max_pz)
        ax.legend(fontsize=20)
        ax.tick_params(labelsize=14)
        if title is not None:
            ax.set_title(title, fontsize=20)
        fig.tight_layout()

        return fig

    def calculate_pcchat(self, balrog_data, balrog_overlap_weight, max_iter=0, wide_cell_key='cell_wide', deep_cell_key='cell_deep', force_assignment=True, replace=False):
        """With a given balrog_data (no Z required), calculate a new pcchat. Returns a new cmap object, which ***has the same spec data as before (albeit a copy in memory), just a new pcchat***"""

        t0 = time.time()
        log('Starting construction of new p(c|chat,s) from new s. Loading data', t0)

        if force_assignment:
            log('Assigning SOM Deep', t0)
            cell_deep = self.assign_deep(balrog_data)
        elif deep_cell_key in balrog_data.columns:
            cell_deep = balrog_data[deep_cell_key].values
        else:
            log('Assigning SOM Deep', t0)
            cell_deep = self.assign_deep(balrog_data)

        if force_assignment:
            log('Assigning SOM wide', t0)
            cell_wide = self.assign_wide(balrog_data)
        elif wide_cell_key in balrog_data.columns:
            cell_wide = balrog_data[wide_cell_key].values
        else:
            log('Assigning SOM wide', t0)
            cell_wide = self.assign_wide(balrog_data)

        pcchat = self.build_pcchat(cell_wide, cell_deep, balrog_overlap_weight, wide_som=self.wide_som, deep_som=self.deep_som, max_iter=max_iter, replace=replace, t0=t0)

        log('Creating class object', t0)
        new_cm = self.update(data=self.data, pcchat=pcchat)
        return new_cm

    def update(self, data=None, overlap_weight=None, pcchat=None, wide_som=None, deep_som=None, wide_columns=None, deep_columns=None, wide_err_columns=None, deep_err_columns=None):
        """Returns a new cellmap with new data and pcchat. Everything is copied
        """
        # checking all the variables. There probably is a better way
        if data is None:
            data = self.data
        if pcchat is None:
            if self.pcchat is None:
                # None can't copy itself
                pcchat = None
            else:
                pcchat = self.pcchat
        if wide_som is None:
            wide_som = self.wide_som
        if deep_som is None:
            deep_som = self.deep_som
        if wide_columns is None:
            wide_columns = self.wide_columns
        if wide_err_columns is None:
            wide_err_columns = self.wide_err_columns
        if deep_columns is None:
            deep_columns = self.deep_columns
        if deep_err_columns is None:
            deep_err_columns = self.deep_err_columns

        # in read, we can just do cls(kwargs), but the self is the actual object, so we have to do self.__class__. Note that you can NOT do cls.__class__
        new_cm = self.__class__(data.copy(), overlap_weight, wide_som.copy(), deep_som.copy(), pcchat.copy(), wide_columns, wide_err_columns, deep_columns, deep_err_columns)
        return new_cm

    @classmethod
    def fit(cls, data, overlap_weight,
            wide_columns, wide_err_columns, deep_columns, deep_err_columns,
            data_train_deep=None, data_train_wide=None, data_train_overlap=None,
            wide_som=None, deep_som=None, n_mcmc_draws=0, share_soms=False, deep_kwargs={}, wide_kwargs={}, force_assignment=True, **kwargs):
        """Make a SOM (if none given), assign data to SOM cells, build p(c|chat,data).

        Parameters
        ----------
        data :              A pandas dataframe of galaxies with both deep and
                            wide observations that will be used for calculating
                            histograms. It must contain the wide_columns,
                            wide_err_columns, deep_columns, and
                            deep_err_columns.
        overlap_weight :    Vector of weights for each entry in the sample, e.g.
                            shear response, or the inverse of the number of times
                            a deep galaxy was drawn, or the product, or just 1.
        wide_columns :      Columns which correspond to information used to
                            make wide field assignment to SOM.
        wide_err_columns :  Columns which correspond to errors in the wide
                            field parameters.
        deep_columns :      Columns which correspond to information used to
                            make deep field assignment to SOM.
        deep_err_columns :  Columns which correspond to errors in the deep
                            field parameters.
        data_train_deep :   Data used for training the deep SOM. If None
                            specified, use data. [default: None]
        data_train_wide :   Data used for training the wide SOM. If None
                            specified, use data. [default: None]
        data_train_overlap: Data used for calculating p(c | chat) between two
                            SOMs. If None specified, use data. [default: None]
        wide_som :          SelfOrganizingMap object, if already made
        deep_som :          SelfOrganizingMap object, if already made
        wide/deep_kwargs :  Arguments to pass for making the SelfOrganizingMap
        force_assignment :  Calculate cell assignments. If False, then will use
                            whatever value is in the 'cell_deep' and
                            'cell_wide' fields of data, if they are present.
                            Default: True
        share_soms :        If True, then the wide and deep soms are the same. Default: False
        n_mcmc_draws :      Number of MCMC draws on wide field data. Default 0 (skip)


        If no SOM is provided, then the following kwargs may be provided for
        fitting the SOM to the given deep data:

        map_shape :     desired output map shape = [dim1, dim2]. (n_out_dim,)
        learning_rate : float
        max_iter :      maximum number of steps in algorithm fit
        min_val :       minimum parameter difference we worry about in updating
                        SOM. This in practice usually doesn't come up, as we
                        limit the range of cells a SOM may update to be less
                        than one wrap around the map.

        Returns
        -------
        CellMap : A CellMap (or inherited class)

        Additional kwargs go into SelfOrganizingMap.fit, if no som is passed in
        """

        t0 = time.time()

        deep_kwargs.update(kwargs)
        wide_kwargs.update(kwargs)

        deep_diag_ivar = True

        if deep_som is None:
            log('Fitting SOM to deep data', t0)
            if data_train_deep is None:
                data_train_deep = data
            deep_x = cls.get_x_deep(data_train_deep, deep_columns)
            deep_ivar = cls.get_ivar_deep(data_train_deep, deep_columns, deep_err_columns)
            deep_som = SelfOrganizingMap.fit(deep_x, deep_ivar, diag_ivar=deep_diag_ivar, **deep_kwargs)
        else:
            log('Already have deep SOM', t0)
        if share_soms:
            if wide_som is None:
                wide_som = deep_som
            elif wide_som == deep_som:
                pass
            else:
                raise Exception('Both share_soms is True AND we specified a different wide_som?')
        else:
            if wide_som is None:
                log('Fitting SOM to wide data', t0)
                if data_train_wide is None:
                    data_train_wide = data
                wide_x = cls.get_x_wide(data_train_wide, wide_columns)
                wide_ivar = cls.get_ivar_wide(data_train_wide, wide_columns, wide_err_columns)
                wide_som = SelfOrganizingMap.fit(wide_x, wide_ivar, **wide_kwargs)
            else:
                log('Already have wide SOM', t0)

        # get columns
        log('Loading data', t0)
        deep_x = cls.get_x_deep(data, deep_columns)
        deep_ivar = cls.get_ivar_deep(data, deep_columns, deep_err_columns)
        wide_x = cls.get_x_wide(data, wide_columns)
        wide_ivar = cls.get_ivar_wide(data, wide_columns, wide_err_columns)

        if 'cell_deep' in data and not force_assignment:
            log('Grabbing SOM Deep', t0)
            cell_deep = data['cell_deep']
        else:
            # do assignments
            log('Assigning SOM Deep', t0)
            cell_deep = deep_som.assign(deep_x, deep_ivar, diag_ivar=deep_diag_ivar)
            data['cell_deep'] = cell_deep
            cell_deep_ND = deep_som.cell1d_to_cell(cell_deep)
            for i, ci in enumerate(cell_deep_ND):
                data['cell_deep_{0}'.format(i)] = ci

        if 'cell_wide' in data and not force_assignment:
            log('Grabbing SOM Wide', t0)
            cell_wide = data['cell_wide']
        else:
            log('Assigning SOM Wide', t0)
            cell_wide = wide_som.assign(wide_x, wide_ivar)
            data['cell_wide'] = cell_wide
            cell_wide_ND = wide_som.cell1d_to_cell(cell_wide)
            for i, ci in enumerate(cell_wide_ND):
                data['cell_wide_{0}'.format(i)] = ci

            if n_mcmc_draws > 0:
                log('Assigning {0} SOM Wide MC Draws'.format(n_mcmc_draws), t0)
                cell_wide_mcs = wide_som.assign_probabilistic(wide_x, wide_ivar, n_mcmc=n_mcmc_draws)
                for i in range(n_mcmc_draws):
                    if i > 0:
                        mcmc_key = 'cell_wide_mc__{0:02d}'.format(i)
                    else:
                        mcmc_key = 'cell_wide_mc'
                    cell_wide_mc = cell_wide_mcs[:, i]
                    data[mcmc_key] = cell_wide_mc
                    cell_wide_mc_ND = wide_som.cell1d_to_cell(cell_wide_mc)
                    for i, ci in enumerate(cell_wide_mc_ND):
                        data[mcmc_key + '_{0}'.format(i)] = ci


        log('Building p(c | chat, data) Matrix', t0)
        if data_train_overlap is None:
            data_train_overlap = data
            # and we have already defined wide_x, etc
        else:
            wide_x = cls.get_x_wide(data_train_overlap, wide_columns)
            wide_ivar = cls.get_ivar_wide(data_train_overlap, wide_columns, wide_err_columns)
            deep_x = cls.get_x_deep(data_train_overlap, deep_columns)
            deep_ivar = cls.get_ivar_deep(data_train_overlap, deep_columns, deep_err_columns)

            if 'cell_deep' in data_train_overlap and not force_assignment:
                cell_deep = data_train_overlap['cell_deep']
            else:
                cell_deep = deep_som.assign(deep_x, deep_ivar, diag_ivar=deep_diag_ivar)
            if 'cell_wide' in data_train_overlap and not force_assignment:
                cell_wide = data_train_overlap['cell_wide']
            else:
                cell_wide = wide_som.assign(wide_x, wide_ivar)

        pcchat = cls.build_pcchat(cell_wide, cell_deep, overlap_weight, wide_som=wide_som, deep_som=deep_som, max_iter=0, replace=False, t0=t0)

        log('Creating class object', t0)
        return cls(data, overlap_weight, wide_som, deep_som, pcchat, wide_columns, wide_err_columns, deep_columns, deep_err_columns, **kwargs)

    @classmethod
    def fitDESY3(cls, spec_data, overlap_weight, wide_columns, wide_err_columns, deep_columns, deep_err_columns, data_train_deep, data_train_wide, zp, deep_kwargs={}, wide_kwargs={}, **kwargs):
        # overlap_weight: Weights of galaxies in spec_data, to account for shear response, or uneven number of times these galaxies were drawn in the wide data

        t0 = time.time()

        deep_kwargs.update(kwargs)
        wide_kwargs.update(kwargs)

        deep_diag_ivar = True
        cls.zp = zp
        log('Fitting SOM to deep data', t0)
        deep_x = cls.get_x_deep(data_train_deep, deep_columns, zp)
        deep_ivar = cls.get_ivar_deep(data_train_deep, deep_columns, deep_err_columns)
        deep_som = SelfOrganizingMap.fit(deep_x, deep_ivar, diag_ivar=deep_diag_ivar, **deep_kwargs)

        log('Fitting SOM to wide data', t0)
        wide_x = cls.get_x_wide(data_train_wide, wide_columns, zp)
        wide_ivar = cls.get_ivar_wide(data_train_wide, wide_columns, wide_err_columns, zp)
        wide_som = SelfOrganizingMap.fit(wide_x, wide_ivar, **wide_kwargs)

        # get deep columns
        log('Loading spec_data', t0)
        deep_x = cls.get_x_deep(spec_data, deep_columns, zp)
        deep_ivar = cls.get_ivar_deep(spec_data, deep_columns, deep_err_columns)
        # do deep assignments
        log('Assigning SOM Deep', t0)
        cell_deep = deep_som.assign(deep_x, deep_ivar, diag_ivar=deep_diag_ivar)
        spec_data['cell_deep'] = cell_deep

        pcchat = 0

        log('Creating class object', t0)
        return cls(spec_data, overlap_weight, wide_som, deep_som, pcchat, wide_columns, wide_err_columns, deep_columns, deep_err_columns, zp, **kwargs)

    @classmethod
    def fitDESY3_deep_only(cls, spec_data, overlap_weight, deep_columns, deep_err_columns, data_train_deep, deep_kwargs={}, **kwargs):

        t0 = time.time()

        deep_kwargs.update(kwargs)

        deep_diag_ivar = True

        log('Fitting SOM to deep data', t0)
        deep_x = cls.get_x_deep(data_train_deep, deep_columns)
        deep_ivar = cls.get_ivar_deep(data_train_deep, deep_columns, deep_err_columns)
        deep_som = SelfOrganizingMap.fit(deep_x, deep_ivar, diag_ivar=deep_diag_ivar, **deep_kwargs)

        wide_x = None
        wide_ivar = None
        wide_som = SelfOrganizingMap(w=np.zeros((64, 64)),shape=(64,64))
        wide_columns = None
        wide_err_columns = None

        # get deep columns
        log('Loading spec_data', t0)
        deep_x = cls.get_x_deep(spec_data, deep_columns)
        deep_ivar = cls.get_ivar_deep(spec_data, deep_columns, deep_err_columns)

        # do deep assignments
        log('Assigning SOM Deep', t0)
        cell_deep = deep_som.assign(deep_x, deep_ivar, diag_ivar=deep_diag_ivar)
        spec_data['cell_deep'] = cell_deep
        spec_data['cell_wide'] = np.zeros(len(cell_deep))
        pcchat = 0

        log('Creating class object', t0)
        return cls(spec_data, overlap_weight, wide_som, deep_som, pcchat, wide_columns, wide_err_columns, deep_columns, deep_err_columns, **kwargs)

    def assign_deep(self, data, **kwargs):
        """Given data that has appropriate columns, will return deep field assignments as 1d array
        """
        # convert to right form
        x = self.get_x_deep(data, self.deep_columns, self.zp)
        ivar = self.get_ivar_deep(data, self.deep_columns, self.deep_err_columns)

        # assign
        assignment = self.deep_som.assign(x, ivar, **kwargs)  # 1d assignment

        return assignment

    def assign_wide(self, data, **kwargs):
        """Given data that has appropriate columns, will return wide field assignments as 1d array
        """
        # convert to right form
        x = self.get_x_wide(data, self.wide_columns, self.zp)
        ivar = self.get_ivar_wide(data, self.wide_columns, self.wide_err_columns, self.zp)

        # assign
        assignment = self.wide_som.assign(x, ivar, **kwargs)  # 1d assignment

        return assignment

    def probability_wide(self, data):
        """Given data that has appropriate columns, will return 2d array that is probability of galaxy being in wide cell
        """
        x = self.get_x_wide(data, self.wide_columns)
        ivar = self.get_ivar_wide(data, self.wide_columns, self.wide_err_columns)
        prob = self.wide_som.evaluate_probability(x, ivar)
        return prob

    def probability_deep(self, data):
        """Given data that has appropriate columns, will return 2d array that is probability of galaxy being in deep cell
        """
        x = self.get_x_deep(data, self.deep_columns)
        ivar = self.get_ivar_deep(data, self.deep_columns, self.deep_err_columns)
        prob = self.deep_som.evaluate_probability(x, ivar)
        return prob

    @classmethod
    def build_pcchat(cls, wide, deep, overlap_weight, wide_som, deep_som, t0=0, **kwargs):
        """Given indices of assignment to deep and wide, build the matrix p(c | chat, sel)

        Parameters
        ----------
        wide :      array (n_samples) of integer assignments of each galaxy to
                    SOM based on wide fluxes
        deep :      array (n_samples) of integer assignments of each galaxy to
                    SOM based on deep fluxes
        overlap_weight : weight of a galaxy in the transfer function; can be used
                    to correct for inhomogeneous response to shear, or for
                    inhomogeneous numbers for having drawn each deep galaxy
                    from the sample
        wide_som :  A SelfOrganizingMap object
        deep_som :  A SelfOrganizingMap object

        Returns
        -------
        pcchat :    An array of shape (prod(*som.map_shape),
                    prod(*som.map_shape)) which gives the probability of a
                    galaxy really being in cell c given that it is in cell
                    chat.

        """
        log('Building pcchat', t0)
        # I think numba doesn't like np.zeros, which will make it run in python (slow) mode
        pcchat_num = np.zeros((deep_som.size, wide_som.size))

        np.add.at(pcchat_num, [deep, wide], overlap_weight)  # sum(w)[c, chat]
        pcchat_denom = pcchat_num.sum(axis=0)
        pcchat = pcchat_num / pcchat_denom[None]

        log('Checking for non-finites, which are changed to 0 probability', t0)
        # any nonfinite in pcchat are to be treated as 0 probabilty
        pcchat = np.where(np.isfinite(pcchat), pcchat, 0)

        return pcchat

    def get_deep_wide(self, data=None):
        """Returns deep and wide values from catalog
        """
        if data is None:
            data = self.data
        deep_x = self.get_x_deep(data, self.deep_columns)
        deep_ivar = self.get_ivar_deep(data, self.deep_columns, self.deep_err_columns)
        wide_x = self.get_x_wide(data, self.wide_columns)
        wide_ivar = self.get_ivar_wide(data, self.wide_columns, self.wide_err_columns)

        return deep_x, deep_ivar, wide_x, wide_ivar

    @classmethod
    def get_x(cls, data, columns, kind='wide'):
        if kind == 'wide':
            return cls.get_x_wide(data, columns)
        elif kind == 'deep':
            return cls.get_x_deep(data, columns)
        else:
            raise TypeError("{0} is an invalide kind. Please use either 'wide' or 'deep'".format(kind))

    @classmethod
    def get_x_wide(cls, data, columns):
        x = data[columns].values
        return x

    @classmethod
    def get_x_deep(cls, data, columns):
        x = data[columns].values
        return x

    @classmethod
    def get_ivar(cls, data, columns, kind='wide'):
        if kind == 'wide':
            return cls.get_ivar_wide(data, columns)
        elif kind == 'deep':
            return cls.get_ivar_deep(data, columns)
        else:
            raise TypeError("{0} is an invalide kind. Please use either 'wide' or 'deep'".format(kind))

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns):
        err = data[err_columns].values
        ivar = fill_ivar_from_error_diag(err)
        return ivar

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values
        ivar = fill_ivar_from_error_diag(err)
        return ivar

@numba.jit(nopython=True)
def gaussian_rbf(weight_map, central_index, cND, map_shape, scale_length=1, max_length=0, **kwargs):
    # fills weight map with gaussian kernel exp(-0.5 (distance / scale_length) ** 2)

    w_dims = len(weight_map)
    map_dims = len(map_shape)
    inv_scale_length_square = scale_length ** -2.

    if max_length <= 0:
        max_length_square = np.inf
    else:
        max_length_square = max_length ** 2

    # update all cells
    for c in range(w_dims):
        # convert to ND
        unravel_index(c, map_shape, cND)

        # get distance, including accounting for toroidal topology
        diff2 = 0.0
        for di in range(map_dims):
            best_c = central_index[di]
            ci = cND[di]
            i_dims = map_shape[di]
            diff = (ci - best_c)
            while diff < 0:
                diff += i_dims
            while diff >= i_dims * 0.5:
                diff -= i_dims
            diff2 += diff * diff
            if diff2 > max_length_square:
                continue

        if diff2 <= max_length_square:
            weight_map[c] = np.exp(-0.5 * diff2 * inv_scale_length_square)

def log(string, t0=0):
    print('{0}: {1}'.format(time.time() - t0, string))

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

@numba.jit
def fill_ivar_from_error_diag(x):
    # one of those cases where if you have to think about it too much, just numba it
    n_samples, n_dim = x.shape
    ivar = np.zeros((n_samples, n_dim, n_dim), dtype=x.dtype)
    for i in range(n_samples):
        for j in range(n_dim):
            ivar[i, j, j] = x[i, j] ** -2.
            """
            if ~np.isfinite(ivar[i, j, j]):
                print(i,j, x[i,j], ivar[i, j, j], x[i, j] ** -2)
            """
    return ivar

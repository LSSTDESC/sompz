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

"""
.. module:: cellprob

Replaces many common cell operations that used just the max assignment with chi2 probability based methods
"""

from .cell import CellMap

class CellMapWideProbabilistic(object):
    """This class is like CellMap, except now many operations that once used tuple cell assignments have been replaced with assignments based on p(chat | fluxes, errors). For only the wide
    """

    def assign_wide(self, data):
        """Given data that has appropriate columns, will return wide field assignments drawing from the assignment probability
        """
        # convert to right form
        x = self.get_x_wide(data, self.wide_columns)
        ivar = self.get_ivar_wide(data, self.wide_columns, self.wide_err_columns)

        # assign
        assignment = self.wide_som.assign_probabilistic(x, ivar)  # 1d assignment

        return assignment

    def get_cell_weights_wide(self, data, max_iter=0, replace=False, **kwargs):
        """Given data, get cell weights and indices via chi2 probability

        Parameters
        ----------
        data :  Dataframe we extract parameters from

        Returns
        -------
        cells :         The names of the cells
        cell_weights :  The fractions of the cells
        """
        t0 = time.time()
        log('Starting fit. Loading wide data', t0)
        # get wide x and ivar
        wide_x = self.get_x_wide(data, self.wide_columns)
        wide_ivar = self.get_ivar_wide(data, self.wide_columns, self.wide_err_columns)

        log('Building p(chat | data) Matrix from {0} samples'.format(len(wide_x)), t0)
        cells, cell_weights = self.wide_som.fractional_occupation(wide_x, wide_ivar, max_iter=max_iter, replace=replace)

        return cells, cell_weights

    def histogram(self, key, cells, cell_weights, data_overlap, bins=None, deep_cell_key='cell_deep', max_iter=0, interpolate_kwargs={}):
        """ Return histogram calculating chi2 occupation probability from deep overlap

        Parameters
        ----------
        key : Parameter to extract from datafram
        cells : A list of wide cells to return sample from, or a single int.
        cell_weights : How much we weight each wide cell. This is the array p(chat | sample)
        bins : Bins we histogram the values into
        data_overlap : data with both wide and deep information

        Returns
        -------
        hist : a histogram of the values from self.data[key]

        Notes
        -----
        This calculates p(c|chat)p(chat|xhat,sigmahat) for all xhat, sigmahat passed. I have made this function because p(c|chat) for e.g. 256 x 256 matrix is a 32 gig array. That's a bit unwieldy, and actually can't be loaded in memory!

        """
        t0 = time.time()
        all_cells = np.arange(self._number_deep_cells)
        hists = self.get_deep_histograms(key=key, cells=all_cells, bins=bins, interpolate_kwargs=interpolate_kwargs)

        # get columns
        wide_x = self.get_x_wide(data_overlap, self.wide_columns)
        wide_ivar = self.get_ivar_wide(data_overlap, self.wide_columns, self.wide_err_columns)

        if deep_cell_key in data_overlap.columns:
            cell_deep = data_overlap[deep_cell_key].values
        else:
            log('Assigning SOM Deep', t0)
            cell_deep = self.assign_deep(data_overlap)

        # p(c|s)
        pc = self.wide_som.divided_fractional_occupation(wide_x, wide_ivar, cell_deep, self._number_deep_cells, cells, cell_weights, max_iter)
        weights = pc

        conds = (weights != 0) & np.all(np.isfinite(hists), axis=1)
        # sum_c p(z|c) p(c|s) = p(z|s)
        hist = np.sum((hists[conds] * weights[conds, None]), axis=0)

        dx = np.diff(bins)
        normalization = np.sum(dx * hist)
        if normalization != 0:
            hist = hist / normalization
        return hist

    def redshift_distributions_wide(self, data, bins, tomo_bins={}, key='Z', force_assignment=True, interpolate_kwargs={}, **kwargs):
        """Returns redshift distribution for sample

        Parameters
        ----------
        data :      Data sample of interest with wide data
        bins :      bin edges for redshift distributions data[key]
        tomo_bins : Which cells belong to which tomographic bins. If nothing is
                    passed in, then we by default just use all cells
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
            cells, cell_weights = self.get_cell_weights_wide(data, force_assignment=force_assignment, **kwargs)
            if cells.size == 0:
                hist = np.zeros(len(bins) - 1)
            else:
                if 'cell_deep' not in self.data.columns:
                    self.data['cell_deep'] = self.assign_deep(self.data, **kwargs)
                # skip using the pcchat
                hist = self.histogram(key=key, cells=cells, cell_weights=cell_weights, bins=bins, data_overlap=self.data)
            return hist
        else:
            cells, cell_weights = self.get_cell_weights_wide(data, force_assignment=force_assignment, **kwargs)
            # break up hists into the different bins
            hists = []
            for tomo_key in tomo_bins:
                cells_use = tomo_bins[tomo_key]
                cells_conds = np.in1d(cells, cells_use)
                if cells[cells_conds].size == 0:
                    hist = np.zeros(len(bins) - 1)
                else:
                    if 'cell_deep' not in self.data.columns:
                        self.data['cell_deep'] = self.assign_deep(self.data, **kwargs)
                    # skip using the pcchat
                    hist = self.histogram(key=key, cells=cells[cells_conds], cell_weights=cell_weights[cells_conds], bins=bins, data_overlap=self.data)
                hists.append(hist)
            hists = np.array(hists)
            return hists

    def calculate_pcchat(self, data, max_iter=0, deep_cell_key='cell_deep', force_assignment=True, replace=False):
        """With a given data, calculate a new pcchat. Returns a new cmap object, which ***has the same data as before (albeit a copy in memory), just a new pcchat***"""

        t0 = time.time()
        log('Starting construction of new p(c|chat,s) from new s. Loading data', t0)

        if force_assignment:
            log('Assigning SOM Deep', t0)
            cell_deep = self.assign_deep(data)
        elif deep_cell_key in data.columns:
            cell_deep = data[deep_cell_key].values
        else:
            log('Assigning SOM Deep', t0)
            cell_deep = self.assign_deep(data)

        # get columns
        wide_x = self.get_x_wide(data, self.wide_columns)
        wide_ivar = self.get_ivar_wide(data, self.wide_columns, self.wide_err_columns)
        if max_iter > 0 and max_iter < len(wide_x):
            n_iter = max_iter
        else:
            n_iter = len(wide_x)
        log('Building p(c | chat, data) Matrix from {0} samples'.format(n_iter), t0)
        pcchat = self.build_pcchat(wide_x, wide_ivar, cell_deep, wide_som=self.wide_som, deep_som=self.deep_som, max_iter=max_iter, replace=replace, t0=t0)

        log('Creating class object', t0)
        # in read, we can just do cls(kwargs), but the self is the actual object, so we have to do self.__class__. Note that you can NOT do cls.__class__
        new_cm = self.update(self.data, pcchat)
        return new_cm

    @classmethod
    def build_pcchat_chi2(cls, wide_x, wide_ivar, deep, wide_som, deep_som, max_iter=0, replace=False, t0=0):
        """Given indices of assignment to deep and fluxes in the wide, build
        the matrix p(c | chat, sel)

        Parameters
        ----------
        deep :      array (n_samples) of integer assignments of each galaxy to
                    SOM based on deep fluxes
        wide_x :    parameters for the wide field
        ivar_x :    inverse variance of the wide field params
        wide_som :  A SelfOrganizingMap object
        deep_som :  A SelfOrganizingMap object
        choices :   indices of samples to draw from x

        Returns
        -------
        pcchat :    An array of shape (prod(*deep_som.map_shape),
                    prod(*wide_som.map_shape)) which gives the probability of a
                    galaxy really being in cell c given that it is in cell
                    chat.
        """
        log('Building pcchat', t0)
        pcchat = wide_som.build_pcchat_hist_chi2(wide_x, wide_ivar, deep, deep_som.size, max_iter=max_iter, replace=replace)

        return pcchat

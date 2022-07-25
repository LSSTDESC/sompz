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

"""
.. module:: som
"""

import numba

class SelfOrganizingMap(object):

    def __init__(self, w, shape=None):
        """A class to encapsulate the SOM functions written out below, and to
        hold together the resultant variables

        Parameters
        ----------
        w :     self organizing map weights (*map_shapes, input_dims)
        shape : shape of the self organizing map (map_dim_1, map_dim_2, ... input_dim). If not specified, will try to figure out the shape from w

        """

        if shape is not None:
            self._shape = shape
            self._map_shape = shape[:-1]
            self._ndim = len(self._map_shape)
            self._input_shape = shape[-1]
            self._size = np.prod(shape[:-1])

            self.w = w
            self._w_shape = w.reshape(shape)
        else:
            # interpret from the SOM w if we can
            self._shape = w.shape
            self._map_shape = w.shape[:-1]
            self._ndim = len(self._map_shape)
            self._input_shape = w.shape[-1]
            self._size = np.prod(w.shape[:-1])

            # flatten w
            self._w_shape = w
            self.w = w.reshape((self._size, self._input_shape))

        assert self.w.shape == (self._size, self._input_shape), "SOM assumes cells flattened to 1d indexing"

    def __getitem__(self, *args, **kwargs):
        return self._w_shape.__getitem__(*args, **kwargs)

    @property
    def ndim(self):
        return self._ndim

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def map_shape(self):
        return self._map_shape

    @property
    def input_shape(self):
        return self._input_shape

    @classmethod
    def read(cls, path, name='som'):
        import sompz
        import pandas as pd
        import h5py
        w = pd.read_hdf(path, '{0}/weight'.format(name)).values
        try:
            shape = pd.read_hdf(path, '{0}/shape'.format(name)).values
        except KeyError:
            shape = None

        try:
            # old version
            kind = pd.read_hdf(path, '{0}/type'.format(name))['type'].values[0]
        except TypeError:
            with h5py.File(path, 'r') as h5f:
                kind = h5f['{0}/type'.format(name)][:].tolist()[0]
                kind = kind.decode('utf-8')

        som_class = getattr(sompz, kind)
        som = som_class(w, shape=shape)
        return som

    def write(self, path, name='som'):
        import pandas as pd
        import h5py
        pd.DataFrame(self.w).to_hdf(path, '{0}/weight'.format(name))
        pd.Series(self.shape).to_hdf(path, '{0}/shape'.format(name))
        # pd.DataFrame({'type': [self.__class__.__name__]}).to_hdf(path, '{0}/type'.format(name))
        with h5py.File(path, 'r+') as h5f:
            try:
                h5f.create_dataset('{0}/type'.format(name), data=[self.__class__.__name__.encode('utf-8')])
            except RuntimeError:
                del h5f['{0}/type'.format(name)]
                h5f.create_dataset('{0}/type'.format(name), data=[self.__class__.__name__.encode('utf-8')])

    def copy(self):
        return self.__class__(self.w.copy(), self._shape)

    def evaluate(self, x, ivar):
        """Return chi2 of input x's and ivar's

        Parameters
        ----------
        x :         Input vector (n_samples, input_shape)
        ivar :      Input inverse variance of x (n_samples, input_shape, input_shape)

        Returns
        -------
        chi2 :      Chi2 fit of each sample to each map cell (n_samples, som_size)

        Notes
        -----
        The input x does not have to have the full span of self.input_shape.
        The chi2 will be evaluated only up to the shape of x. This means that
        you can use a subset of the dimensions as follows: if we trained on
        dimensions griz, then you could pass in gr, but not gi or iz.

        """
        # check number of dims required
        dim = x.shape[1]
        if dim != self.input_shape:
            print('Warning! Trying to evaluate SOM with shape {0} using input of shape {1}. I hope you meant to do this!'.format(self.input_shape, dim))

        n_samples, n_dims = x.shape
        chi2 = np.zeros((n_samples, self.size), dtype=np.float64)
        evaluate_som(x, ivar, self.w, chi2)  # operates on chi2
        return chi2

    def evaluate_probability(self, x, ivar):
        """Return p(cell) of input x's and ivar's

        Parameters
        ----------
        x :         Input vector (n_samples, input_shape)
        ivar :      Input inverse variance of x (n_samples, input_shape, input_shape)

        Returns
        -------
        prob :      Probability of each galaxy belonging to each cell (n_samples, som_size)

        Notes
        -----
        The input x does not have to have the full span of self.input_shape.
        The chi2 will be evaluated only up to the shape of x. This means that
        you can use a subset of the dimensions as follows: if we trained on
        dimensions griz, then you could pass in gr, but not gi or iz.

        """
        # check number of dims required
        dim = x.shape[1]
        if dim != self.input_shape:
            print('Warning! Trying to evaluate SOM with shape {0} using input of shape {1}. I hope you meant to do this!'.format(self.input_shape, dim))

        n_samples, n_dims = x.shape
        prob = np.zeros((n_samples, self.size), dtype=np.float64)
        pc_t = np.zeros(self.size, dtype=np.float64)
        evaluate_som_probability(x, ivar, self.w, prob, pc_t)  # operates on chi2
        return prob

    def __call__(self, x, ivar):
        return self.evaluate(x, ivar)

    @classmethod
    def fit(cls, x, ivar, map_shape=[10, 10],
            learning_rate=0.5, max_iter=2e6, min_val=1e-4,
            verbose=True, diag_ivar=False, replace=True):
        """Calculate Self Organizing Map

        Parameters
        ----------
        x :             input data of shape (n_samples, n_dim)
        ivar :          inverse variance of input data (n_samples, n_dim, n_dim)
        map_shape :     desired output map shape = [*map_dims]. (n_out_dim,)
        learning_rate : float usually between 0 and 1. Sets how large of a
                        change we can effect in the weights at each step by
                        multiplying the change by:
                            learning_rate ** (step_t / total_t)
        max_iter :      maximum number of steps in algorithm fit
        min_val :       minimum parameter difference we worry about in updating
                        SOM. This in practice usually doesn't come up, as we
                        limit the range of cells a SOM may update to be less
                        than one wrap around the map.

        Returns
        -------
        w : self organizing map weights (*map_dims * n_dims)

        """
        # check shapes of input values
        assert x.shape[0] == ivar.shape[0] and x.shape[1] == ivar.shape[1] and x.shape[1] == ivar.shape[2]

        w = train_som(x, ivar, np.array(map_shape), learning_rate, max_iter, min_val, verbose, diag_ivar, replace)
        return cls(w, map_shape + [x.shape[1]])

    def cell1d_to_cell(self, c):
        """Takes 1d assignment vector and turns into Nd

        Parameters
        ----------
        c :     A list of integers (n_samples)

        Returns
        -------
        cND :   Cell assignments (len(map_shape), n_samples)
        """
        cND = np.unravel_index(c, self.map_shape)
        return cND

    def cell_to_cell1d(self, cND):
        """Takes Nd assignment vector and turns into 1d

        Parameters
        ----------
        cND :   Cell assignments (len(map_shape), n_samples)

        Returns
        -------
        c :     A list of integers (n_samples)
        """
        c = np.ravel_multi_index(cND, self.map_shape)
        return c

    def cell1d_to_cell2d(self, c):
        """Takes 1d assignment vector and turns into 2d

        Parameters
        ----------
        c :     A list of integers (n_samples)

        Returns
        -------
        c0, c1: Two dimensional versions of c

        """
        # NOTE: should be exactly same as cell1d_to_cell
        c0 = c % self.map_shape[0]
        c1 = c // self.map_shape[0]
        return c0, c1

    def cell2d_to_cell1d(self, c0, c1):
        """Takes 1d assignment vector and turns into 2d

        Parameters
        ----------
        c0, c1: Two dimensional versions of c

        Returns
        -------
        c :     A list of integers (n_samples)

        """
        # NOTE: should be exactly same as cell_to_cell1d
        c = c1 * self.map_shape[0] + c0
        return c

    def assign(self, x, ivar, verbose=True, diag_ivar=False):
        """Assign sample to a som cell

        Parameters
        ----------
        x :         input data of shape (n_samples, n_dim)
        ivar :      inverse variance of input data (n_samples, n_dim, n_dim)
        verbose :   Print extra information?

        Returns
        -------
        cell :      Best matching cell (by chi2) for each sample (n_samples) in
                    1d coordinate

        """
        # check number of dims required
        dim = x.shape[1]
        if dim != self.input_shape and verbose:
            print('Warning! Trying to assign to SOM with shape {0} using input of shape {1}. I hope you meant to do this!'.format(self.input_shape, dim))

        n_samples = len(x)
        cell = np.zeros(n_samples, dtype=np.int32)
        # cell = assign_bmu(x, ivar, self.w, cell, verbose, diag_ivar)
        assign_bmu(x, ivar, self.w, cell, verbose, diag_ivar)
        return cell

    def assign_probabilistic(self, x, ivar, n_mcmc=1, verbose=True, diag_ivar=False):
        """For each galaxy, evaluate p(cell | x, ivar) and then return an assignment drawing from p(c|x,ivar)

        Parameters
        ----------
        x :         input data of shape (n_samples, n_dim)
        ivar :      inverse variance of input data (n_samples, n_dim, n_dim)
        n_mcmc :    Number of mcmc draws [default: 1]
        verbose :   Print extra information?

        Returns
        -------
        cell :      Best matching cell (by chi2) for each sample (n_samples) in
                    1d coordinate

        """
        # check number of dims required
        dim = x.shape[1]
        if dim != self.input_shape and verbose:
            print('Warning! Trying to assign to SOM with shape {0} using input of shape {1}. I hope you meant to do this!'.format(self.input_shape, dim))

        n_samples = len(x)
        cell = np.zeros((n_samples, n_mcmc), dtype=np.int32)
        cumsum_pchat_t = np.zeros(self.size, dtype=np.float64)
        cell = assign_probabilistic(x, ivar, self.w, n_mcmc, cell, cumsum_pchat_t, verbose, diag_ivar)
        return cell

    def fractional_occupation(self, x, ivar, max_iter=0, replace=False, diag_ivar=False):
        """Get fraction of x that reside in each cell by weighting by chi2.
        This is p(chat | s)
        """
        choices = self.build_choices(x, max_iter, replace)

        # print('Building pchat', t0)
        pchat = np.zeros(self.size)
        pchat_t = np.zeros(self.size)
        # print('Starting chi2 hist calculation', t0)
        _build_weights_chi2(x, ivar, self.w, pchat, pchat_t, choices)

        # print('Checking for non-finites, which are changed to 0 probability', t0)
        # any nonfinite in pcchat are to be treated as 0 probabilty
        pchat = np.where(np.isfinite(pchat), pchat, 0)
        # print('normalizing pchat', t0)
        pchat = pchat / np.nansum(pchat, axis=0)[None]

        cell_weights = pchat
        cells = np.arange(self.size)
        return cells, cell_weights

    def divided_fractional_occupation(self, x, ivar, cell_deep, number_deep_cells, cells, cell_weights, max_iter=0, replace=False, diag_ivar=False):
        """Get fraction of x that reside in some deep cell c based on chi2 weighting with current SOM and p(c|chat,s)
        This is p(c | s)
        """

        choices = self.build_choices(x, max_iter, replace)

        pc = np.zeros(number_deep_cells)
        pchat_t = np.zeros(self.size)
        _build_pc_hist_chi2(x, ivar, cell_deep, self.w, pc, pchat_t, choices, cells, cell_weights)

        return pc

    def build_pcchat_hist_chi2(self, x, ivar, deep, number_deep_cells, max_iter=0, replace=False):

        choices = self.build_choices(x, max_iter, replace)

        # print('Building pcchat', t0)
        pcchat = np.zeros((number_deep_cells, self.size))
        pchat_t = np.zeros((self.size))
        # print('Starting chi2 hist assignment', t0)
        _build_pcchat_hist_chi2(x, ivar, deep, self.w, pcchat, pchat_t, choices)

        # print('Checking for non-finites, which are changed to 0 probability', t0)
        # any nonfinite in pcchat are to be treated as 0 probabilty
        pcchat = np.where(np.isfinite(pcchat), pcchat, 0)
        # print('normalizing pcchat', t0)
        pcchat = pcchat / np.nansum(pcchat, axis=0)[None]

        return pcchat

    @classmethod
    def build_choices(cls, x, max_iter=0, replace=False):
        return build_choices(x, max_iter, replace)

    def umatrix(self, N_diff=1):
        """Look at average distance within adjacent cells

        Parameters
        ----------
        N_diff : distance away to look

        Returns
        -------
        u: average distance with adjacent cells

        """
        if self.ndim != 2:
            raise NotImplementedError("Don't know (yet) how to calculate umatrix of SOMs of dimensionality {0}!".format(self.ndim))
        return umatrix(self._w_shape, N_diff)

    def plot(self, filename=None, fig=None, axs=None, **kwargs):
        """Plot the SOM vectors

        Parameters
        ----------
        filename :  Save the figure to this file, if specified.
        fig, axs :  Use these matplotlib figures and axes, if specified
        kwargs :    Any other kwargs to into ax.pcolor for each axis

        Returns
        -------
        fig :       A matplotlib figure of this plot

        """
        if self.ndim != 2:
            raise NotImplementedError("Don't know how to plot SOMs of dimensionality {0}!".format(self.ndim))

        import matplotlib.pyplot as plt
        if not fig:
            fig, axs = plt.subplots(nrows=self._w_shape.shape[-1], ncols=1, figsize=(4, 3 * (self._w_shape.shape[-1])), sharex=True)
            for i in range(len(axs)):
                axs[i].set_ylabel('C1')
                if i == len(axs) - 1:
                    axs[i].set_xlabel('C0')
                axs[i].set_title('X{0}'.format(i))

        # plot the weight vectors with pcolor
        for i in range(self._w_shape.shape[-1]):
            wi = self._w_shape[:, :, i]
            ax = axs[i]
            IM = ax.pcolor(wi, **kwargs)
            fig.colorbar(IM, ax=ax)

        fig.tight_layout()
        if filename:
            fig.savefig(filename)

        return fig

def train_som(x, ivar, map_shape=[10, 10], learning_rate=0.5, max_iter=2e6, min_val=1e-4, verbose=False, diag_ivar=False, replace=False):
    """Calculate Self Organizing Map

    Parameters
    ----------
    x :             input data of shape (n_samples, n_dim)
    ivar :          inverse variance of input data (n_samples, n_dim, n_dim)
    map_shape :     desired output map shape = [dim1, dim2]. (n_out_dim,)
    learning_rate : float usually between 0 and 1. Sets how large of a
                    change we can effect in the weights at each step by
                    multiplying the change by:
                        learning_rate ** (step_t / total_t)
    max_iter :      maximum number of steps in algorithm fit
    min_val :       minimum parameter difference we worry about in updating
                    SOM. This in practice usually doesn't come up, as we limit
                    the range of cells a SOM may update to be less than one
                    wrap around the map.
    verbose :       Print updates?

    Returns
    -------
    w : self organizing map weights (dim1, dim2, n_dims)

    Notes
    -----
    Suggest whitening your data to span 0-1 range before training

    """

    # initialize w
    # WARNING: initial w is between 0 and 1, not the full range of values in x
    if verbose:
        print('Initializing SOM weight map with shape ({0}, {1})'.format(np.prod(map_shape), x.shape[1]))
    w = np.random.random(size=(np.prod(map_shape), x.shape[1]))

    # select what x and ivar we cycle through
    max_iter = int(max_iter)
    if verbose:
        print('Choosing {0} draws from {1} training samples'.format(max_iter, len(x)))
    choices = build_choices(x, max_iter, replace)

    # update w
    # these arrays cannot be made inside numba functions with nopython=True, so we make them here:
    cND = np.zeros(len(map_shape), dtype=int)
    best_cellND = np.zeros(len(map_shape), dtype=int)
    sigma2_s = np.max(map_shape) ** 2
    update_som_weights(x, ivar, w, choices, cND, best_cellND, learning_rate, sigma2_s, min_val, map_shape, verbose, diag_ivar)

    return w

@numba.jit(nopython=True)  # use nopython to make sure we aren't dropping back to python objects
def update_som_weights(x, ivar, w, choices, cND, best_cellND, learning_rate, sigma2_s, min_val, map_shape, verbose=False, diag_ivar=False):
    """Update Self Organizing Map

    Parameters
    ----------
    x : input data of shape (n_samples, n_dim)
    ivar : inverse variance of input shape (n_samples, n_dim, n_dim)
    w : initial self organizing map weights (w_dims, n_dims)
    choices : indices of samples to draw from x
    learning_rate : float usually between 0 and 1. Sets how large of a
                    change we can effect in the weights at each step by
                    multiplying the change by:
                        learning_rate ** (step_t / total_t)
    min_val : minimum parameter difference we worry about in updating SOM
    map_shape : shape of map w is supposed to be, not including n_dims
    verbose : print updates?

    Returns
    -------
    w : self organizing map weights (dim1, dim2, n_dims)

    Notes
    -----
    n_dims is decided based on x, not w

    """
    n_samples, n_dims = x.shape
    w_dims = w.shape[0]
    map_dims = len(map_shape)
    n_iter = len(choices)

    # sigma2_s = (np.min([j_dims, k_dims])) ** 2
    j_dims = 0
    for i in range(map_dims):
        ji = map_shape[i]
        if ji > j_dims:
            j_dims = ji
    # set the limit in distance the SOM will update
    max_N_diff = np.int(0.5 * j_dims + 1)

    n_print_iter = 10000
    # print(n_samples, n_dims, j_dims, k_dims, n_iter, max_N_diff)

    one_over_n_iter = 1. / n_iter
    one_over_n_dims = 1. / n_dims

    for t in range(n_iter):
        index = choices[t]
        # find index of minimal chi2
        chi2_min, c_min = find_minimum_chi2(x, ivar, w, w_dims, n_dims, one_over_n_dims, index, diag_ivar)
        if c_min == -1:
            raise Exception('Must assign a cell when fitting SOM')

        # if we are only taking one step, then use the values we stuck in
        if t == 0 and n_iter == 1:
            t_ratio = t * one_over_n_iter
            # learning rate
            a = learning_rate
            sigma_neg2_t = sigma2_s ** -1
        else:
            t_ratio = t * one_over_n_iter
            # learning rate
            a = learning_rate ** t_ratio
            sigma_neg2_t = sigma2_s ** (t_ratio - 1)

        # window the indices we update
        N_diff = np.int(np.sqrt(np.abs(np.log(min_val / a) / sigma_neg2_t)))
        if verbose:
            if t % n_print_iter == 0:
                print('step t: ', t, ' . Fraction done: ', t_ratio)
                # print('Distance away we we would update based on min_val and time step: ', N_diff, ' . Max distance away we would ever look : ', max_N_diff)
                # print('t / total steps: ', t_ratio, ' learning rate ** tratio: ', a, ' j_dims ** 2 ** (tratio - 1) :', sigma_neg2_t)
                print('index: ', index, ' current best cell: ', c_min, ' chi2 of best cell: ', chi2_min)
        if N_diff > max_N_diff:
            N_diff = max_N_diff

        if N_diff == 0:
            # we are done, so stop!
            # print('Stopping SOM at Iteration {0}'.format(t))
            print('Stopping SOM because N_diff == 0')
            print('step: ', t)
            return w

        # get j, k, etc of best matching cell
        unravel_index(c_min, map_shape, best_cellND)

        # update all cells
        for c in range(w_dims):
            # convert to ND
            unravel_index(c, map_shape, cND)

            # get distance, including accounting for toroidal topology
            diff2 = 0.0
            for di in range(map_dims):
                best_c = best_cellND[di]
                ci = cND[di]
                i_dims = map_shape[di]
                diff = (ci - best_c)
                while diff < 0:
                    diff += i_dims
                while diff >= i_dims:
                    diff -= i_dims
                diff2 += diff * diff

            # get Hbk
            Hbk = np.exp(-sigma_neg2_t * diff2)

            # update
            for i in range(n_dims):
                w[c, i] += a * Hbk * (x[index, i] - w[c, i])

    # return w

@numba.jit(nopython=True)
def assign_bmu(x, ivar, w, bmu, verbose=True, diag_ivar=False):
    """Assign best matching unit from self organizing map weights

    Parameters
    ----------
    x : input data of shape (n_samples, n_dim)
    ivar : inverse variance of input shape (n_samples, n_dim, n_dim)
    w : self organizing map weights (dim1, dim2, n_dim)

    Returns
    -------
    bmu : best matching cell that each x goes into (n_samples,)

    """

    n_samples, n_dims = x.shape
    w_dims = w.shape[0]

    n_print_iter = 10000
    one_over_n_dims = 1. / n_dims

    for t in range(n_samples):
        if t % n_print_iter == 0:
            if verbose:
                print('assigning sample ', t, ' out of ', n_samples)

        chi2_min, bmu_t = find_minimum_chi2(x, ivar, w, w_dims, n_dims, one_over_n_dims, t, diag_ivar)

        # hooray, now save the best matching unit for input t
        bmu[t] = bmu_t

    # return bmu

@numba.jit(nopython=True)
def assign_probabilistic(x, ivar, w, n_mcmc, cell, cumsum_pchat_t, verbose=True, diag_ivar=False):
    """Assign sample to cell based on probability of being in that cell.

    Parameters
    ----------
    x : input data of shape (n_samples, n_dim)
    ivar : inverse variance of input shape (n_samples, n_dim, n_dim)
    w : self organizing map weights (dim1, dim2, n_dim)
    cell : array containing cell assignments
    cumsum_pchat_t : temporary holder for p(c|x, ivar) that is cumulatively summed!


    Returns
    -------
    cell : cell assignments

    """
    n_samples, n_dims = x.shape
    w_dims = w.shape[0]

    n_print_iter = 10000
    one_over_n_dims = 1. / n_dims

    for t in range(n_samples):
        if t % n_print_iter == 0:
            if verbose:
                print('assigning sample ', t, ' out of ', n_samples)

        # construct p(chat | fhat)
        cumsum_pchat_t_sum = 0.0
        for c in range(w_dims):
            chi2 = evaluate_chi2(x, ivar, w, n_dims, one_over_n_dims, index=t, cell=c, diag_ivar=diag_ivar)
            cumsum_pchat_t_jk = np.exp(-0.5 * chi2)
            cumsum_pchat_t[c] = cumsum_pchat_t_jk
            if c > 0:
                cumsum_pchat_t[c] += cumsum_pchat_t[c - 1]
            cumsum_pchat_t_sum += cumsum_pchat_t_jk
            # print(c, cumsum_pchat_t[c])

        # if we have 0 probability (not sure what that really means) then we
        # assign it to whatever find_minimum_chi2 finds
        if cumsum_pchat_t_sum == 0:
            # print('Warning! Failed MC draw for galaxy. This sometimes happens when one cell is _a much better fit_ than other cells. Therefore we will fill this galaxy with the minimum chi2 cell.')
            # print(t)
            chi2_min, c = find_minimum_chi2(x, ivar, w, w_dims, n_dims, one_over_n_dims, t)
            for mcmc_indx in range(n_mcmc):
                cell[t, mcmc_indx] = c
        else:
            for mcmc_indx in range(n_mcmc):
                draw = np.random.random()
                # print(draw)
                # normalize the cumsum probability to 1
                draw_t = cumsum_pchat_t_sum * draw

                indx = 0
                found_cell = False
                # figure out which indx by stepping through cells
                for c in range(w_dims):
                    if cumsum_pchat_t[c] > draw_t:
                        indx = c
                        # print(indx)
                        found_cell = True
                    if found_cell:
                        break

                # hooray, now save the best matching unit for input t
                cell[t, mcmc_indx] = indx

    return cell

@numba.jit(nopython=True)
def find_minimum_chi2(x, ivar, w, w_dims, n_dims, one_over_n_dims, index, diag_ivar=False):
    """Find which cell has the minimum chi2

    Parameters
    ----------
    x : input data of shape (n_samples, n_dims)
    ivar : inverse variance of input shape (n_samples, n_dims, n_dims)
    w : self organizing map weights (w_dims, n_dims)
    w_dims : number of map cells
    n_dims : input vector dims
    index : which object in x we are looking at

    Returns
    -------
    chi2_min : minimum chi2 of best match
    cell_min : index of best match

    """

    chi2_min = 1e100
    cell_min = -1
    # find index of minimal chi2
    for c in range(w_dims):
        chi2 = evaluate_chi2(x, ivar, w, n_dims, one_over_n_dims, index, cell=c, chi2_break=chi2_min, diag_ivar=diag_ivar)
        if chi2 < chi2_min:
            chi2_min = chi2
            cell_min = c

    if cell_min == -1:
        print('No minimum cell found for object:', index)
        print('flux is:', x[index])
        print('ivar is:', ivar[index])

    return chi2_min, cell_min

@numba.jit(nopython=True)
def evaluate_chi2(x, ivar, w, n_dims, one_over_n_dims, index, cell, chi2_break=2e1000, diag_ivar=False):
    """Get chi2. Break if chi2 is larger than chi2_break.
    """
    chi2 = 0.0
    give_up = False
    for i in range(n_dims):
        chi2 += (x[index, i] - w[cell, i]) * (x[index, i] - w[cell, i]) * ivar[index, i, i] * one_over_n_dims
        if chi2 > chi2_break:
            # it only gets worse
            give_up = True
            break
        if not diag_ivar:
            # take advantage of symmetry
            for i2 in range(i + 1, n_dims):
                chi2 += 2 * (x[index, i] - w[cell, i]) * (x[index, i2] - w[cell, i2]) * ivar[index, i, i2] * one_over_n_dims
                if chi2 > chi2_break:
                    # it only gets worse
                    give_up = True
                    break
        if give_up:
            break
    return chi2

@numba.jit(nopython=True)
def evaluate_chi2_probability(pchat_t, x, ivar, w, w_dims, n_dims, one_over_n_dims, index, diag_ivar=False):
    chi2_min = 1e100
    pchat_t_sum = 0.0

    for c in range(w_dims):
        chi2 = evaluate_chi2(x, ivar, w, n_dims, one_over_n_dims, index=index, cell=c, diag_ivar=diag_ivar)
        if chi2 < chi2_min:
            chi2_min = chi2
        pchat_t[c] = chi2

    # we have to find chi2_min before we can calculate the probabilities,
    # so we have to go through the som cells again
    for c in range(w_dims):
        # to prevent *flow errors, subtract the best chi2. This works
        # out to be a constant factor in the ratio for pchat_t
        chi2 = pchat_t[c]
        pchat_t_jk = np.exp(-0.5 * (chi2 - chi2_min))
        pchat_t[c] = pchat_t_jk
        pchat_t_sum += pchat_t_jk

    # OK one final time to multiply the probabilities in, which seems silly
    if pchat_t_sum == 0:
        one_over_pchat_t_sum = 0.
    else:
        one_over_pchat_t_sum = 1. / pchat_t_sum

    for c in range(w_dims):
        pchat_t[c] *= one_over_pchat_t_sum
    # return pchat_t

@numba.jit(nopython=True)
def evaluate_som(x, ivar, w, chi2, diag_ivar=False):
    """Get chi2 of x with each cell in w

    Parameters
    ----------
    x : input data of shape (n_samples, n_dim)
    ivar : inverse variance of input shape (n_samples, n_dim, n_dim)
    w : self organizing map weights (dim1, dim2, n_dim)
    chi2 : chi2 of each cell with each input (n_samples, dim1 * dim2)

    chi2 is filled by this operation

    """
    n_samples, n_dims = x.shape
    w_dims = w.shape[0]
    one_over_n_dims = 1. / n_dims

    for t in range(n_samples):
        for c in range(w_dims):
            chi2i = evaluate_chi2(x, ivar, w, n_dims, one_over_n_dims, index=t, cell=c, diag_ivar=diag_ivar)
            chi2[t, c] = chi2i

    # return chi2


@numba.jit(nopython=True)
def evaluate_som_probability(x, ivar, w, prob, pchat_t, diag_ivar=False):
    """Get chi2 of x with each cell in w

    Parameters
    ----------
    x : input data of shape (n_samples, n_dim)
    ivar : inverse variance of input shape (n_samples, n_dim, n_dim)
    w : self organizing map weights (dim1 * dim2, n_dim)
    prob : prob of each cell with each input (n_samples, dim1 * dim2)
    pchat_t : prob of cell occupation (som size)

    prob is filled by this operation

    """
    n_samples, n_dims = x.shape
    w_dims = w.shape[0]
    one_over_n_dims = 1. / n_dims

    for t in range(n_samples):
        evaluate_chi2_probability(pchat_t, x, ivar, w, w_dims, n_dims, one_over_n_dims, index=t, diag_ivar=diag_ivar)
        for c in range(w_dims):
            prob[t, c] = pchat_t[c]

@numba.jit(nopython=True)
def _build_pcchat_hist_chi2(x, ivar, deep, w, pcchat, pchat_t, choices, diag_ivar=False):
    """hist: for the fact that the deeps are histogram assigned. chi2: because the wide are chi2 assigned

    Here we calculate p(c|chat, sample) = sum_{i} p(c_i|chat)p(chat|xhat_i, sigmahat_i)
    """
    n_samples, n_dims = x.shape
    w_dims = w.shape[0]
    one_over_n_dims = 1. / n_dims

    n_iter = len(choices)

    for ti in range(n_iter):
        t = choices[ti]
        if ti % 10000 == 0:
            print(ti, n_iter)

        evaluate_chi2_probability(pchat_t, x, ivar, w, w_dims, n_dims, one_over_n_dims, index=t, diag_ivar=diag_ivar)

        # now add to pcchat[deep_t, wide]
        deep_t = deep[t]
        for wide in range(w_dims):
            pcchat[deep_t, wide] += pchat_t[wide]

    # return pcchat

@numba.jit(nopython=True)
def _build_pc_hist_chi2(x, ivar, deep, w, pc, pchat_t, choices, cells, cell_weights, diag_ivar=False):
    """hist: for the fact that the deeps are histogram assigned. chi2: because the wide are chi2 assigned.

    Here we calculate p(c|sample) = sum_{chat, i} p(c_i|chat)p(chat|xhat_i, sigmahat_i)
    """
    n_samples, n_dims = x.shape
    w_dims = w.shape[0]
    one_over_n_dims = 1. / n_dims

    chat_dims = len(cells)

    n_iter = len(choices)

    for ti in range(n_iter):
        t = choices[ti]
        if ti % 10000 == 0:
            print(ti, n_iter)

        evaluate_chi2_probability(pchat_t, x, ivar, w, w_dims, n_dims, one_over_n_dims, index=t, diag_ivar=diag_ivar)

        # now add to pc[deep_t]
        deep_t = deep[t]
        for k in range(chat_dims):
            pc[deep_t] += cell_weights[k] * pchat_t[cells[k]]

    # return pc

# this is also slow for the same reasons. Only difference with build_pcchat is that we do not divide by deep cell assignment
@numba.jit(nopython=True)
def _build_weights_chi2(x, ivar, w, pchat, pchat_t, choices, diag_ivar=False):
    """chi2: because the wide are chi2 assigned"""
    n_samples, n_dims = x.shape
    w_dims = w.shape[0]
    one_over_n_dims = 1. / n_dims

    n_iter = len(choices)

    for ti in range(n_iter):
        t = choices[ti]
        if ti % 10000 == 0:
            print(ti, n_iter)

        evaluate_chi2_probability(pchat_t, x, ivar, w, w_dims, n_dims, one_over_n_dims, index=t, diag_ivar=diag_ivar)

        # now add to pchat[wide]
        for wide in range(w_dims):
            pchat[wide] += pchat_t[wide]

    # return pchat


@numba.jit
def umatrix(w, N_diff=1):
    """Look at average distance within adjacent cells

    Parameters
    ----------
    w : self organizing map weights (dim1, dim2, n_dim)
    N_diff : distance away to look

    Returns
    -------
    u: average distance with adjacent cells

    """
    j_dims, k_dims, n_dims = w.shape
    u = np.zeros((j_dims, k_dims), dtype=np.float64)

    for m in range(j_dims):
        for n in range(k_dims):
            umn = 0
            # handle how far away we are willing to look
            for j_i in range(2 * N_diff + 1):
                if j_i > N_diff:
                    j_diff = -(j_i - N_diff)
                else:
                    j_diff = j_i
                j = m + j_diff
                while j < 0:
                    j += j_dims
                while j >= j_dims:
                    j -= j_dims
                for k_i in range(2 * N_diff + 1):
                    if k_i > N_diff:
                        k_diff = -(k_i - N_diff)
                    else:
                        k_diff = k_i
                    k = n + k_diff
                    while k < 0:
                        k += k_dims
                    while k >= k_dims:
                        k -= k_dims

                    for i in range(n_dims):
                        umn += (w[m, n, i] - w[j, k, i]) ** 2
            # now turn into distance instead of distance squared, average
            u[m, n] = np.sqrt(umn) / n_dims

    return u

def whiten(x, ivar, cmin, cmax):
    """Whiten vectors with cmin, cmax
    """
    dim = x.shape[1]
    x = (x - cmin[:dim]) / (cmax - cmin)[:dim]
    ivar = ivar * (cmax - cmin)[None, :dim, None] * (cmax - cmin)[None, None, :dim]
    return x, ivar

def build_choices(x, max_iter=0, replace=False):
    if max_iter == 0:
        choices = np.random.choice(len(x), size=len(x), replace=replace)
    elif max_iter <= len(x):
        choices = np.random.choice(len(x), size=max_iter, replace=replace)
    else:
        # max_iter > len(x). assume intentional
        choices = np.random.choice(len(x), size=max_iter, replace=True)

    return choices

# I do not believe I use this but eh
@numba.jit(nopython=True)
def ravel_index(c, map_shape):
    ndim = len(map_shape)
    power = 1
    val = 0
    for i in range(ndim - 1, 0 - 1, -1):
        val += power * c[i]
        power *= map_shape[i]
    return val

@numba.jit(nopython=True)
def unravel_index(c, map_shape, cout):
    ndim = len(map_shape)
    ci = c + 0
    for i in range(ndim - 1, 0 - 1, -1):
        cj = ci % map_shape[i]
        cout[i] = cj
        ci -= cj
        ci //= map_shape[i]

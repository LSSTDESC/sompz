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
.. module:: fluxmodes

We have found that we tested a fair number of different kinds of CellMaps. We felt it best to move that bit over to its own file, so that it can be a little more manageable.
"""

from .cell import CellMap, fill_ivar_from_error_diag

class CellMapColor(CellMap):
    """CellMap which takes as x_keys magnitudes and error_keys magerr, and then
    creates x, ivar based on the colors relative to the 0th band.

    Note: get_ivar_deep is still diagonal!
    """

    @classmethod
    def get_x_wide(cls, data, columns):
        x = data[columns].values
        return mag_to_color(x)

    @classmethod
    def get_x_deep(cls, data, columns):
        x = data[columns].values
        return mag_to_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns):
        err = data[err_columns].values
        ivar = fill_ivar_color(err)
        return ivar

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapMagColor(CellMap):
    """CellMap which takes as x_keys magnitudes and error_keys magerr, and then
    creates x, ivar based on the colors relative to the 0th band, keeping the
    0th band as magnitudes.

    Note: get_ivar_deep is still diagonal!
    """

    @classmethod
    def get_x_wide(cls, data, columns):
        x = data[columns].values
        return mag_to_mag_color(x)

    @classmethod
    def get_x_deep(cls, data, columns):
        x = data[columns].values
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns):
        err = data[err_columns].values
        ivar = fill_ivar_magcolor(err)
        return ivar

class CellMapColorColorMagnitude(CellMap):
    @classmethod
    def get_x_wide(cls, data, columns):
        x = data[columns].values
        return mag_to_mag_color(x)

    @classmethod
    def get_x_deep(cls, data, columns):
        x = data[columns].values
        return mag_to_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns):
        err = data[err_columns].values
        ivar = fill_ivar_magcolor(err)
        return ivar

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapFluxFluxRatio(CellMap):
    """x_keys are fluxes. error_keys are flux errors. Creates xi, ivar based
    on flux ratio relative to 0th band, keeping 0th band as flux.

    Note: get_ivar_deep is still diagonal!
    """

    @classmethod
    def get_x_wide(cls, data, columns):
        x = data[columns].values
        return flux_to_flux_fluxratio(x)

    @classmethod
    def get_x_deep(cls, data, columns):
        x = data[columns].values
        return flux_to_flux_fluxratio(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns):
        flux = data[columns].values
        err = data[err_columns].values
        ivar = fill_ivar_flux_fluxratio(err, flux)
        return ivar

class CellMapLnFluxFluxRatio(CellMap):
    """x_keys are fluxes. error_keys are flux errors. Creates xi, ivar based
    on flux ratio relative to 0th band and transforming 0th bandn to ln flux

    Note: get_ivar_deep is still diagonal!
    """

    @classmethod
    def get_x_wide(cls, data, columns):
        x = data[columns].values
        return flux_to_lnflux_fluxratio(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns):
        flux = data[columns].values
        err = data[err_columns].values
        ivar = fill_ivar_lnflux_fluxratio(err, flux)
        return ivar

class CellMapColorLuptitcolor(CellMap):
    """CellMap which takes as x_deep_keys magnitudes and error_deep_keys magerr, and then creates x, ivar based on the colors relative to the 0th band for the deep data. It takes as x_wide_keys fluxes and error_wide_keys inverse variance on fluxes, and then creates x, ivar based on the lupticolors relative to the 0th band for the wide data.
    """
    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolor(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns):
        x = data[columns].values
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapLupticolor(CellMap):
    """CellMap which takes as x_keys fluxes and error_keys inverse variance on fluxes, and then
    creates x, ivar based on the lupticolors relative to the 0th band.
    """
    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolor(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapLupticolorluptitude(CellMap):
    """CellMap which takes as x_keys fluxes and error_keys inverse variance on fluxes, and then
    creates x, ivar based on the lupticolors relative to the 0th band, keeping 0th band as luptitude in both wide and deep.
    Deep errors are (for now) nominal diagonal
    """

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapLupticolorLupticolorluptitude(CellMap):
    """CellMap which takes as x_keys fluxes and error_keys inverse variance on fluxes, and then
    creates x, ivar based on the lupticolors relative to the 0th band, keeping 0th band as luptitude.
    """

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapLupticolorLupticolorluptitude_softb(CellMap):
    """Same as CellMapLupticolorLupticolorluptitude, but the deep luptitudes have different softening parameter b"""

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep_softb(x, 0, 1, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapLupticolorLupticolorluptitude_wideIRZ(CellMap):
    """Same as CellMapLupticolorLupticolorluptitude without g band in wide, the wide columns must be in order IRZ
    """

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide_irz(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide_irz(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapLupticolorLupticolorluptitude_withoutY(CellMap):
    """Same as CellMapLupticolorLupticolorluptitude, but there is no Y band in deep columns
    """

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep_withoutY(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

    
class CellMapLupticolorLupticolorluptitude_wideIRZ_withoutY(CellMap):
    """Same as CellMapLupticolorLupticolorluptitude, but there is no Y band in deep columns
    """

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide_irz(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide_irz(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep_withoutY(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar
    
class CellMapLupticolorLupticolorluptitude_wideIGRZ_withoutY(CellMap):
    """Same as CellMapLupticolorLupticolorluptitude, but there is no Y band in deep columns and g in wide
    """

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide_igrz(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide_igrz(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep_withoutY(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

class CellMapLupticolorLupticolorluptitudeIvarDeep(CellMap):
    """Same as CellMapLupticolorLupticolorluptitude, but expect erros in the deep_err_columns, fill ivar accordingly
    """

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_deep(x, 0, zp)[0]
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_deep(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolor(lupt_var)
        return ivar

class CellMapColorLupticolorluptitude(CellMap):
    """CellMap which takes as x_deep_keys magnitudes and error_deep_keys magerr, and then creates x, ivar based on the colors relative to the 0th band for the deep data. It takes as x_wide_keys fluxes and error_wide_keys inverse variance on fluxes, and then creates x, ivar based on the lupticolors relative to the 0th band for the wide data,  keeping 0th band as luptitude.
    """

    @classmethod
    def get_x_wide(cls, data, columns, zp):
        x = data[columns].values
        x = luptize_wide(x, 0, zp)[0]
        return mag_to_mag_color(x)

    @classmethod
    def get_ivar_wide(cls, data, columns, err_columns, zp):
        flux = data[columns].values
        flux_ivar = data[err_columns].values
        lupt_var = luptize_wide(flux, 1/flux_ivar, zp)[1]
        ivar = fill_ivar_lupticolorluptitude(lupt_var)
        return ivar

    @classmethod
    def get_x_deep(cls, data, columns):
        x = data[columns].values
        return mag_to_color(x)

    @classmethod
    def get_ivar_deep(cls, data, columns, err_columns):
        err = data[err_columns].values[:, 1:]
        ivar = fill_ivar_from_error_diag(err)
        return ivar

# a series of useful inverse variance calculators for the above classes. Maybe
# these should live inside their objects, especially since we always write
# methods that use them?

def mag_to_color(x):
    x = x[:, 1:] - x[:, 0][:, None]
    return x

def mag_to_mag_color(x):
    x[:, 1:] = x[:, 1:] - x[:, 0][:, None]
    return x

def flux_to_fluxratio(x):
    x = x[:, 1:] / x[:, 0][:, None]
    return x

def flux_to_flux_fluxratio(x):
    x[:, 1:] = x[:, 1:] / x[:, 0][:, None]
    return x

def flux_to_lnflux_fluxratio(x):
    x[:, 1:] = x[:, 1:] / x[:, 0][:, None]
    # hacky as hell: take abs value
    x[:, 0] = np.log(np.abs(x[:, 0]))
    return x

@numba.jit
def fill_ivar_diag(x):
    # one of those cases where if you have to think about it too much, just numba it
    n_samples, n_dim = x.shape
    ivar = np.zeros((n_samples, n_dim, n_dim), dtype=x.dtype)
    for i in range(n_samples):
        for j in range(n_dim):
            ivar[i, j, j] = x[i, j]
    return ivar

@numba.jit
def fill_ivar_color(emag):
    """
    prefactor = 1 / sum_k sigma_k^-2 for k in ALL bands
    off diagonal factor Sigma^-1_{jk} = -sigma_j^-2 sigma_k^-2
    diagonal = off diagonal term + sigma_j^-2
    """
    # assume we do color based on subtracting 0th mag index
    n_gal, n_dim = emag.shape
    icovar = np.zeros((n_gal, n_dim - 1, n_dim - 1))
    for i in range(0, n_gal):
        iprefactor = 0
        for j in range(0, n_dim):
            iprefactor += emag[i, j] ** -2
        iprefactor = iprefactor ** -1

        for j in range(n_dim - 1):
            for k in range(n_dim - 1):
                icovar[i, j, k] = -iprefactor * emag[i, j + 1] ** -2 * emag[i, k + 1] ** -2
                if j == k:
                    icovar[i, j, k] += emag[i, j + 1] ** -2
    return icovar

@numba.jit
def fill_ivar_magcolor(emag):
    # assume we do color based on subtracting 0th mag index
    n_gal, n_dim = emag.shape
    icovar = np.zeros((n_gal, n_dim, n_dim))
    for i in range(0, n_gal):
        icovar[i, 0, 0] = emag[i, 0] ** -2.
        for j in range(1, n_dim):
            ivarmag = emag[i, j] ** -2.
            icovar[i, j, j] = ivarmag
            icovar[i, 0, j] = ivarmag
            icovar[i, j, 0] = ivarmag
            icovar[i, 0, 0] += ivarmag
    return icovar

@numba.jit
def fill_ivar_flux_fluxratio(eflux, flux):
    n_gal, n_dim = eflux.shape
    icovar = np.zeros((n_gal, n_dim, n_dim))
    for i in range(0, n_gal):
        icovar[i, 0, 0] += (flux[i, 0] / eflux[i, 0]) ** 2
        for j in range(1, n_dim):
            icovar[i, 0, 0] += (flux[i, j] / eflux[i, j]) ** 2
            icovar[i, j, 0] = flux[i, 0] * flux[i, j] * eflux[i, j] ** - 2.
            icovar[i, 0, j] = icovar[i, j, 0]
            icovar[i, j, j] = (flux[i, 0] / eflux[i, j]) ** 2

    return icovar

@numba.jit
def fill_ivar_lnflux_fluxratio(eflux, flux):
    n_gal, n_dim = eflux.shape
    icovar = np.zeros((n_gal, n_dim, n_dim))
    for i in range(0, n_gal):
        icovar[i, 0, 0] += eflux[i, 0] ** -2
        for j in range(1, n_dim):
            icovar[i, 0, 0] += (flux[i, j] / eflux[i, j] / flux[i, 0]) ** 2
            icovar[i, j, 0] = flux[i, j] * eflux[i, j] ** - 2.
            icovar[i, 0, j] = icovar[i, j, 0]
            icovar[i, j, j] = (flux[i, 0] / eflux[i, j]) ** 2

    return icovar

@numba.jit
def fill_ivar_color_pinv(emag):
    """if you don't believe the above calculation, you can check manually with this. This is also how you would go about calculating the inverse variances if you do not know how to calculate them analytically."""
    # assume we do color based on subtracting 0th mag index
    n_gal, n_dim = emag.shape
    icovar = np.zeros((n_gal, n_dim - 1, n_dim - 1))
    covar = np.zeros((n_dim - 1, n_dim - 1))
    for i in range(0, n_gal):
        for j in range(n_dim - 1):
            for k in range(n_dim - 1):
                if j == k:
                    covar[j, k] = emag[i, j + 1] ** 2 + emag[i, 0] ** 2
                else:
                    covar[j, k] = emag[i, 0] ** 2
        # create covar from wide colors
        icovar[i] = np.linalg.pinv(covar)
    return icovar

@numba.jit
def fill_ivar_lupticolor(lupt_var):
    n_gal, n_dim = lupt_var.shape
    icovar = np.zeros((n_gal, n_dim - 1, n_dim - 1))
    for i in range(0, n_gal):
        iprefactor = 0
        for j in range(0, n_dim):
            iprefactor += lupt_var[i, j] ** -1
        iprefactor = iprefactor ** -1

        for j in range(n_dim - 1):
            for k in range(n_dim - 1):
                icovar[i, j, k] = -iprefactor * lupt_var[i, j + 1] ** -1 * lupt_var[i, k + 1] ** -1
                if j == k:
                    icovar[i, j, k] += lupt_var[i, j + 1] ** -1
    return icovar

@numba.jit
def fill_ivar_lupticolorluptitude(lupt_var):
    n_gal, n_dim = lupt_var.shape
    icovar = np.zeros((n_gal, n_dim, n_dim))
    for i in range(0, n_gal):
        icovar[i, 0, 0] = lupt_var[i, 0] ** -1.
        for j in range(1, n_dim):
            ivarband = lupt_var[i, j] ** -1.
            icovar[i, j, j] = ivarband
            icovar[i, 0, j] = ivarband
            icovar[i, j, 0] = ivarband
            icovar[i, 0, 0] += ivarband
    return icovar

@numba.jit
def fill_ivar_lupticolor_pinv(lupt_var):
    n_gal, n_dim = lupt_var.shape
    icovar = np.zeros((n_gal, n_dim - 1, n_dim - 1))
    covar = np.zeros((n_dim - 1, n_dim - 1))
    for i in range(0, n_gal):
        for j in range(n_dim - 1):
            for k in range(n_dim - 1):
                if j == k:
                    covar[j, k] = lupt_var[i, 0] + lupt_var[i, j + 1]
                else:
                    covar[j, k] = lupt_var[i, 0]
        icovar[i] = np.linalg.pinv(covar)
    return icovar

@numba.jit
def fill_ivar_lupticolorluptitude_pinv(lupt_var):
    n_gal, n_dim = lupt_var.shape
    icovar = np.zeros((n_gal, n_dim, n_dim))
    covar = np.zeros((n_dim, n_dim))
    for i in range(0, n_gal):
        covar[0, 0] = lupt_var[i, 0]
        for j in range(n_dim - 1):
            covar[j+1, 0] = -lupt_var[i, 0]
            covar[0, j+1] = -lupt_var[i, 0]
            for k in range(n_dim - 1):
                if j == k:
                    covar[j+1, k+1] = lupt_var[i, 0] + lupt_var[i, j + 1]
                else:
                    covar[j+1, k+1] = lupt_var[i, 0]
        icovar[i] = np.linalg.pinv(covar)
    return icovar

def luptize(flux, var, s, zp):
    # s: measurement error (variance) of the flux (with zero pt zp) of an object at the limiting magnitude of the survey
    # a: Pogson's ratio
    # b: softening parameter that sets the scale of transition between linear and log behavior of the luptitudes
    a = 2.5 * np.log10(np.exp(1)) 
    b = a**(1./2) * s 
    mu0 = zp -2.5 * np.log10(b)

    # turn into luptitudes and their errors
    lupt = mu0 - a * np.arcsinh(flux / (2 * b))
    lupt_var = a ** 2 * var / ((2 * b) ** 2 + flux ** 2)
    return lupt, lupt_var

def luptize_wide(flux, var=0, zp=22.5):
    """ The flux must be four dimensional and must be given in the order [f_i, f_g, f_r, f_z] to match the ordering of the softening parameter b """
    #lim_mags = np.array([22.9, 23.7, 23.5, 22.2]) # old
    # see ../test/full_run_on_data/limiting_mags_in_data.ipynb
    lim_mags = np.array([22.92, 23.7, 23.49, 22.28]) # g band is copied from commented array above because g band is not in up to date mastercat
    s = (10**((zp - lim_mags) / 2.5)) / 10  # des limiting mag is 10 sigma

    return luptize(flux, var, s, zp)

def luptize_deep(flux, var=0, zp=22.5):
    """The flux must be 8 dimensional and must be given in the order [f_i, f_g, f_r, f_z, f_u, f_Y, f_J, f_H, f_K] to match the ordering of the softening parameter b """
    #lim_mags_des = np.array([22.9, 23.7, 23.5, 22.2, 25]) # old
    #lim_mags_vista = np.array([24.6, 24.5, 24.0, 23.5]) # old
    lim_mags_des = np.array([24.66, 25.57, 25.27, 24.06, 24.64])
    lim_mags_vista = np.array([24.6, 24.02, 23.69, 23.58]) # y band value is copied from array above because Y band is not in the up to date catalog
    s_des = (10**((zp-lim_mags_des)/2.5)) / 10  # des limiting mag is 10 sigma
    s_vista = (10**((zp-lim_mags_vista)/2.5)) / 10  # vista limiting mag is 10 sigma

    s = np.concatenate([s_des, s_vista])

    return luptize(flux, var, s, zp)

def luptize_wide_irz(flux, var=0, zp=22.5):
    """ The flux must be four dimensional and must be given in the order [f_i, f_r, f_z] to match the ordering of the softening parameter b """
    #lim_mags = np.array([22.9, 23.5, 22.2]) # old
    lim_mags = np.array([22.92, 23.49, 22.28])
    s = (10**((zp - lim_mags) / 2.5)) / 10  # des limiting mag is 10 sigma
    return luptize(flux, var, s, zp)

def luptize_wide_igrz(flux, var=0, zp=22.5):
    """ The flux must be four dimensional and must be given in the order [f_i, f_g, f_r, f_z] to match the ordering of the softening parameter b """
    #lim_mags = np.array([22.9,23.7, 23.5, 22.2]) # old
    lim_mags = np.array([22.92, 23.7, 23.49, 22.28])
    s = (10**((zp - lim_mags) / 2.5)) / 10  # des limiting mag is 10 sigma

    return luptize(flux, var, s, zp)

def luptize_deep_withoutY(flux, var=0, zp=22.5):
    """The flux must be 8 dimensional and must be given in the order [f_i, f_g, f_r, f_z, f_u, f_J, f_H, f_K] to match the ordering of the softening parameter b """
    #lim_mags_des = np.array([22.9, 23.7, 23.5, 22.2, 25])
    #lim_mags_vista = np.array([24.5, 24.0, 23.5])
    lim_mags_des = np.array([24.66, 25.57, 25.27, 24.06, 24.64])
    lim_mags_vista = np.array([24.02, 23.69, 23.58])
    s_des = (10**((zp-lim_mags_des)/2.5)) / 10  # des limiting mag is 10 sigma
    s_vista = (10**((zp-lim_mags_vista)/2.5)) / 10  # vista limiting mag is 5 sigma

    s = np.concatenate([s_des, s_vista])

    return luptize(flux, var, s, zp)

def luptize_deep_softb(flux, var=0, b_plus=0, zp=22.5):
    """The flux must be 8 dimensional and must be given in the order [f_i, f_g, f_r, f_z, f_u, f_Y, f_J, f_H, f_K] to match the ordering of the softening parameter b """
    #lim_mags_des = np.array([22.9, 23.7, 23.5, 22.2, 25]) + b_plus
    #lim_mags_vista = np.array([24.6, 24.5, 24.0, 23.5]) + b_plus
    lim_mags_des = np.array([24.66, 25.57, 25.27, 24.06, 24.64]) + b_plus
    lim_mags_vista = np.array([24.6, 24.02, 23.69, 23.58]) + b_plus # y band value is copied from array above because Y band is not in the up to date catalog

    s_des = (10**((zp-lim_mags_des)/2.5)) / 10  # des limiting mag is 10 sigma
    s_vista = (10**((zp-lim_mags_vista)/2.5)) / 10  # vista limiting mag is 5 sigma

    s = np.concatenate([s_des, s_vista])

    return luptize(flux, var, s, zp)

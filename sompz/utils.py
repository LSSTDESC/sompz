import os
import numpy as np
from datetime import datetime
import twopoint
from astropy.io import fits

# def make_2pt_ensemble_file(ensemble_of_nz, template2ptfile, outfile, z_low, z_mid, z_high, incl_covmat=False):
#     nrealizations= len(ensemble_of_nz)
#     twopt = twopoint.TwoPointFile.from_fits(template2ptfile)
    
#     #rewrite source_nz with the mean of realisations
#     tmp = np.mean(ensemble_of_nz,axis=0)
#     twopt.kernels[0]=twopoint.NumberDensity('nz_source',z_low, z_mid, z_high, [tmp[0],tmp[1],tmp[2],tmp[3]])
    
#     #print("n real: ",nrealizations)
#     #print(np.shape(ensemble_of_nz[0]))
    
#     for ir in range(5): #nrealizations):
#         print('nz_source_realisation_%d'%ir)
#         Nz=[]
#         Nz.append(ensemble_of_nz[ir,0,:])
#         Nz.append(ensemble_of_nz[ir,1,:])
#         Nz.append(ensemble_of_nz[ir,2,:])
#         Nz.append(ensemble_of_nz[ir,3,:])
        
#         #add new extensions
#         twopt.kernels[ir+100]=twopoint.NumberDensity('nz_source_realisation_{0}'.format(ir),z_low, z_mid, z_high, Nz)
#     twopt.to_fits(outfile,clobber=True,overwrite=True)
#     print('write', outfile)
#     os.system('chmod a+r {}'.format(outfile))

def make_2pt_ensemble_file(ensemble_of_nz, template2ptfile, outfile, z_low, z_mid, z_high, incl_covmat=False):
    nrealizations= len(ensemble_of_nz)
    NN =[]
    #print(np.shape(ensemble_of_nz))
    for ix in range(nrealizations):
        Nz=[]
        Nz.append(ensemble_of_nz[ix,0,:])
        Nz.append(ensemble_of_nz[ix,1,:])
        Nz.append(ensemble_of_nz[ix,2,:])
        Nz.append(ensemble_of_nz[ix,3,:])
        NN.append(twopoint.NumberDensity('nz_source_realisation_{0}'.format(ix), 
                                         z_low, z_mid, z_high, Nz))

    obj = twopoint.TwoPointFile([], NN, windows=None, covmat_info=None)
    print("here")
    obj.to_fits('sorry.fits', overwrite=True) # We save it to then loading it again, because I (Carles) don't know better :)
    print("here")
    NZ = fits.open('sorry.fits')
    DV = fits.open(template2ptfile)
    primary_hdu = fits.PrimaryHDU()
    #print(type(DV['nz_source']))
    #print(DV['nz_source'].data)
    if incl_covmat:
        hdu_list = [primary_hdu,
            DV['COVMAT'],
            DV['xip'],
            DV['xim'],
            DV['gammat'],
            DV['wtheta'],
            DV['nz_source'],
            DV['nz_lens'],
            ]
    else:
        hdu_list = [primary_hdu,
            DV['xip'],
            DV['xim'],
            DV['gammat'],
            DV['wtheta'],
            DV['nz_source'],
            DV['nz_lens'],
            ]   
        
    #print("nzlens: ", DV['nz_lens'])
    print("n real: ",nrealizations)
    for ir in range(nrealizations):
        if ir%nrealizations==0:
            print(ir)
        #print('nz_source_realisation_%d'%ir)
        ext = NZ['nz_source_realisation_%d'%ir]
        hdu_list.append(ext)
        
    hdu = fits.HDUList(hdu_list)
    hdu.writeto(outfile, overwrite=True)
    print('replacing the nz_source with mean')
    tmp = np.mean(ensemble_of_nz,axis=0)
    twopt = twopoint.TwoPointFile.from_fits(outfile)
    twopt.kernels[0]=twopoint.NumberDensity('nz_source',z_low, z_mid, z_high, [tmp[0],tmp[1],tmp[2],tmp[3]])
    print('write', outfile)
    twopt.to_fits(outfile,clobber=True,overwrite=True)
    os.system('chmod a+r {}'.format(outfile))
    
def mean_of_hist(y, bins):
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
    #var = np.trapz((x - mean) ** 2 * y, x=x, dx=dx) / normalization
    #sigma = np.sqrt(var)
    #return normalization, mean, sigma
    return mean
def mag2flux(mag, zero_pt=30):
    # zeropoint: M = 30 <=> f = 1
    exponent = (mag - zero_pt)/(-2.5)
    val = 1 * 10 ** (exponent)
    return val

def flux2mag(flux, zero_pt=30):
    return zero_pt - 2.5 * np.log10(flux)

def fluxerr2magerr(flux, fluxerr):
    coef = -2.5 / np.log(10)
    return np.abs(coef * (fluxerr / flux))

def magerr2fluxerr(magerr, flux):
    coef = np.log(10) / -2.5
    return np.abs(coef * magerr * flux)

def printlog(msg,logfile="sompz.log"):
    print msg
    with open(logfile, "a") as myfile:
        now = datetime.now()
        myfile.write(now.strftime("%m/%d/%Y, %H:%M:%S ### ")+msg)
        myfile.write("\n")


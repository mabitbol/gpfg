import numpy as np
import healpy as hp
import glob
import os, errno

simdir = '/project/projectdirs/sobs/v4_sims/mbs/201901_gaussian_fg_lensed_cmb_realistic_noise/512/'
fgsavedir = '/global/homes/m/mabitbol/scratch/cmbsims/fgs/'
savedir = '/global/homes/m/mabitbol/scratch/cmbsims/sims/'

freqs = ['027', '039', '093', '145', '225', '280']


def make_fgs():
    for freq in freqs:
        dust = hp.read_map(simdir+'dust/0010/simonsobs_dust_uKCMB_sa'+freq+'_nside512_0010.fits', field=(0,1,2), verbose=False)
        synch = hp.read_map(simdir+'synchrotron/0010/simonsobs_synchrotron_uKCMB_sa'+freq+'_nside512_0010.fits', field=(0,1,2), verbose=False)
        fg = dust + synch
        hp.write_map(fgsavedir+'synch_dust_sa'+freq+'.fits', fg, overwrite=True)
    return
    
def cmb_plus_noise():
    ncut = 4
    cmbdirs = list(np.sort(glob.glob(simdir+'cmb/*/')))
    noisedirs = list(np.sort(glob.glob(simdir+'noise/*/')))

    #preload fgs
    fgmaps = {}
    for freq in freqs:
        fgmaps[freq] = hp.read_map(fgsavedir+'synch_dust_sa'+freq+'.fits', field=(0,1,2), verbose=False)

    for cmbf in cmbdirs:
        cmbfnames = list(np.sort(glob.glob(cmbf+'*.fits')))
        cmbdirname = cmbf.split('/cmb/')[-1]
        dir1 = make_outdir(savedir+cmbdirname)
        for noisef in noisedirs[:ncut]:
            noisefnames = list(np.sort(glob.glob(noisef+'*.fits')))
            noisedirname = noisef.split('/noise/')[-1]
            dir2 = make_outdir(dir1+noisedirname)
            for k, freq in enumerate(freqs):
                assert(freq in cmbfnames[k])
                assert(freq in noisefnames[k])
                cmb = hp.read_map(cmbfnames[k], field=(0,1,2), verbose=False)
                noise = hp.read_map(noisefnames[k], field=(0,1,2), verbose=False)
                total = cmb + fgmaps[freq] + noise * np.sqrt(ncut)
                hp.write_map(dir2+'total_sim_sa'+freq+'.fits', total, overwrite=True)
    return

def make_outdir(fdir):
    try:
        os.makedirs(fdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return fdir

#make_fgs()
cmb_plus_noise()

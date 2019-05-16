import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import V3calc as v3
import os
import sostuff as stf
import bfore.components as bcm
import bfore.maplike as mpl
import bfore.skymodel as sky
import bfore.instrumentmodel as ins
from bfore.sampling import clean_pixels,run_emcee,run_minimize,run_fisher
import time
import os
import sys

if len(sys.argv)!=16 :
    print("Usage: run_sim.py nside nside_spec_sim nside_spec seed w_nhits output_level w_nell w_fwhm fg_code sensitivity ell_knee verbose removal_complexity with_r with_delensing")
    exit(1)


NSIDE_DEFAULT=int(sys.argv[1])
NSIDE_SPEC_SIMULATE=int(sys.argv[2])
NSIDE_SPEC_DEFAULT=int(sys.argv[3])
NSIDE_RATIO=int(NSIDE_DEFAULT/NSIDE_SPEC_DEFAULT)
ZER0=1E-3
SEED=int(sys.argv[4])
if int(sys.argv[5])>0 :
    W_NHITS=True
else :
    W_NHITS=False
OUTPUT_LEVEL=int(sys.argv[6])
if int(sys.argv[7])>0 :
    W_NELL=True
else :
    W_NELL=False
if int(sys.argv[8])>0 :
    W_FWHM=True
else :
    W_FWHM=False
FG_CODE=sys.argv[9]
SENS_OPTION=int(sys.argv[10])
KNEE_OPTION=int(sys.argv[11])
VERBOSE=int(sys.argv[12])
FGRM_TYP=int(sys.argv[13])
W_R=int(sys.argv[14])
W_DLNS=int(sys.argv[15])
if W_R :
    RFID=0.01
else :
    RFID=0.
if W_DLNS :
    ALENS=0.5
else :
    ALENS=1.0

nu_ref_sync_i=0.408
nu_ref_sync_p=23.
beta_sync_fid=-3.
curv_sync_fid=0.
nu_ref_dust_i=545.
nu_ref_dust_p=353.
beta_dust_fid=1.546
temp_dust_fid=22.76

def read_cl_teb(fname,lmax=3000) :
    """
    Read TEB power spectrum in CAMB format into l,C_l
    """
    data=np.loadtxt(fname,unpack=True)
    larr=data[0]
    clteb=np.zeros([len(larr)+2,4])
    clteb[2:,0]=data[1]*2*np.pi/(larr*(larr+1))
    clteb[2:,1]=data[2]*2*np.pi/(larr*(larr+1))
    clteb[2:,2]=data[3]*2*np.pi/(larr*(larr+1))
    clteb[2:,3]=data[4]*2*np.pi/(larr*(larr+1))
    
    return np.arange(lmax),np.transpose(clteb[:lmax])

def get_nhits() :
    fname_out='data_SO/norm_nHits_SA_35FOV_G.fits'

    if not os.path.isfile(fname_out) :
        fname_in='data_SO/norm_nHits_SA_35FOV.fits'
        mp_C=hp.read_map(fname_in,verbose=False)
        nside_l=hp.npix2nside(len(mp_C))

        nside_h=512
        ipixG=np.arange(hp.nside2npix(nside_h))
        thG,phiG=hp.pix2ang(nside_h,ipixG)
        r=hp.Rotator(coord=['G','C'])
        thC,phiC=r(thG,phiG)
        ipixC=hp.ang2pix(nside_l,thC,phiC)

        mp_G=hp.ud_grade(mp_C[ipixC],nside_out=nside_l)
        hp.write_map(fname_out,mp_G,overwrite=True)

    return hp.ud_grade(hp.read_map(fname_out,verbose=False),
                       nside_out=NSIDE_DEFAULT)

def get_mask() :
    nh=get_nhits()
    nh/=np.amax(nh)
    msk=np.zeros(len(nh))
    not0=np.where(nh>ZER0)[0]
    msk[not0]=nh[not0]
    return msk

def get_noise_sim(sensitivity=SENS_OPTION,knee_mode=KNEE_OPTION,ny_lf=2.,
                  use_nhits=W_NHITS,seed=None,use_nell=W_NELL) :
    if seed is not None :
        np.random.seed(seed)
    nh=get_nhits()
    msk=get_mask()
    fsky=np.mean(msk)
    ll,nll,nlev=v3.so_V3_SA_noise(sensitivity,knee_mode,ny_lf,fsky,3*NSIDE_DEFAULT,remove_kluge=True)
    units=bcm.cmb(v3.so_V3_SA_bands())
    n_perpix=nlev*np.sqrt(hp.nside2npix(NSIDE_DEFAULT)/(4*np.pi))*np.pi/180/60
    id_cut=np.where(nh<ZER0)[0]
    nh[id_cut]=np.amax(nh)
    if not use_nhits :
        nh[:]=1
    mps_no=[]; mps_nv=[]
    for i_n in np.arange(len(nll)) :
        n=nll[i_n]
        nl=np.zeros(3*NSIDE_DEFAULT)
        nl[2:]=n; nl[:2]=n[0]
        if use_nell :
            no_t,no_q,no_u=hp.synfast([nl/2.,nl,nl,0*nl,0*nl,0*nl],nside=NSIDE_DEFAULT,
                                      pol=True,new=True,verbose=False)
        else :
            no_t=np.random.randn(hp.nside2npix(NSIDE_DEFAULT))*n_perpix[i_n]/np.sqrt(2.)
            no_q=np.random.randn(hp.nside2npix(NSIDE_DEFAULT))*n_perpix[i_n]
            no_u=np.random.randn(hp.nside2npix(NSIDE_DEFAULT))*n_perpix[i_n]
        nv_t=n_perpix[i_n]*np.ones_like(no_t)/np.sqrt(2.);
        nv_q=n_perpix[i_n]*np.ones_like(no_q); nv_u=n_perpix[i_n]*np.ones_like(no_u)
        no_t/=np.sqrt(nh/np.amax(nh)); no_q/=np.sqrt(nh/np.amax(nh)); no_u/=np.sqrt(nh/np.amax(nh));
        nv_t/=np.sqrt(nh/np.amax(nh)); nv_q/=np.sqrt(nh/np.amax(nh)); nv_u/=np.sqrt(nh/np.amax(nh));
        mps_no.append([no_t,no_q,no_u])
        mps_nv.append([nv_t,nv_q,nv_u])
    mps_no=np.array(mps_no)*units[:,None,None]
    mps_nv=np.array(mps_nv)*units[:,None,None]

    return msk,mps_no,mps_nv

def get_cmb_sim(r,alens,freqs,seed=None) :
    if seed is not None :
        np.random.seed(seed)
    lp,cteb_prim=read_cl_teb("data_SO/planck1_r1p00_tensCls.dat")
    ll,cteb_lens=read_cl_teb("data_SO/planck1_r0p00_lensedtotCls.dat")
    cltt=(r*cteb_prim+      cteb_lens)[0,:3*NSIDE_DEFAULT]
    clee=(r*cteb_prim+      cteb_lens)[1,:3*NSIDE_DEFAULT]
    clbb=(r*cteb_prim+alens*cteb_lens)[2,:3*NSIDE_DEFAULT]
    clte=(r*cteb_prim+      cteb_lens)[3,:3*NSIDE_DEFAULT]
    amc=np.array(hp.synfast([cltt,clee,clbb,clte,0*clte,0*clte],NSIDE_DEFAULT,
                            pol=True,new=True,verbose=False))[1:]
    mpc=amc[:,:,None]*bcm.cmb(freqs)[None,None,:]
    return amc,mpc

def get_fg_maps() :
    fgmps=np.zeros([2,hp.nside2npix(NSIDE_DEFAULT),6])
    for i in np.arange(6) :
        t,q,u=hp.read_map('maps_FG/sim_'+FG_CODE+'_band%d.fits'%(i+1),field=[0,1,2],verbose=False)
        fgmps[0,:,i]=hp.ud_grade(q,nside_out=NSIDE_DEFAULT); fgmps[1,:,i]=hp.ud_grade(u,nside_out=NSIDE_DEFAULT);
    return fgmps

#Generate directories
predir0='output_sens%d_knee%d/'%(SENS_OPTION,KNEE_OPTION)
if FGRM_TYP>0 :
    predir0+='fgrm%d_'%FGRM_TYP
if W_NHITS :
    predir0+='wnh'
else :
    predir0+='wonh'
if W_NELL :
    predir0+='_nell'
if W_FWHM :
    predir0+='_fwhm'
if W_R :
    predir0+='_r0p01'
if W_DLNS==0 :
    predir0+='_Al1p0'
predir0+='_'+FG_CODE
predir0+='_Ns%d_Nsim%d_Nspec%d/'%(NSIDE_DEFAULT,NSIDE_SPEC_SIMULATE,NSIDE_SPEC_DEFAULT)
predir=predir0+'sim%d/'%SEED
os.system('mkdir -p '+predir)

#Instrument and sky model
nus=v3.so_V3_SA_bands()
bms=v3.so_V3_SA_beams()
nnu=len(nus)
bps=np.array([{'nu':np.array([n-0.5,n+0.5]),'bps':np.array([1])} for n in nus])
instrument=ins.InstrumentModel(bps)
sky_true=sky.SkyModel(['sync_curvedpl','dustmbb','cmb'])
beam_max=np.amax(bms)*np.pi/(180*60) #Largest beam in radians

#Read component amplitudes and spectral parameters
if VERBOSE :
    print("Generating simulation")
amc,mpc=get_cmb_sim(RFID,ALENS,nus,seed=SEED)
mpfg=get_fg_maps()
msk,mps_no1,mps_nv=get_noise_sim(seed=SEED+1)
msk,mps_no2,mps_nv=get_noise_sim(seed=SEED+2)
mps_no1=np.transpose(mps_no1[:,1:,:],axes=[1,2,0])
mps_no2=np.transpose(mps_no2[:,1:,:],axes=[1,2,0])
mps_nv=np.transpose(mps_nv[:,1:,:],axes=[1,2,0])
if W_FWHM :
    mps_d=np.zeros([2,2,hp.nside2npix(NSIDE_DEFAULT),nnu])
    for inu in np.arange(nnu) :
        for ip in [0,1] :
            mp_sky=hp.smoothing((mpfg+mpc)[ip,:,inu],fwhm=beam_max,verbose=False)
            mps_d[0,ip,:,inu]=mp_sky+mps_no1[ip,:,inu]
            mps_d[1,ip,:,inu]=mp_sky+mps_no2[ip,:,inu]
    mps_d_rw=np.array([mps_no1,mps_no2])+(mpfg+mpc)[None,:,:,:]
else :
    mps_d=np.array([mps_no1,mps_no2])+(mpfg+mpc)[None,:,:,:]
if OUTPUT_LEVEL>0 :
    hp.write_map(predir+'cmb_true.fits',amc,overwrite=True)

#Domain decomposition
ipnest_sub=hp.ring2nest(NSIDE_SPEC_DEFAULT,np.arange(hp.nside2npix(NSIDE_SPEC_DEFAULT)))
ipring=hp.nest2ring(NSIDE_DEFAULT,np.arange(hp.nside2npix(NSIDE_DEFAULT)))
ip_patches_good=[]
for ip_sub_ring in np.arange(hp.nside2npix(NSIDE_SPEC_DEFAULT)) :
    ips_ring=ipring[ipnest_sub[ip_sub_ring]*NSIDE_RATIO**2+np.arange(NSIDE_RATIO**2)]
    if np.sum(msk[ips_ring]>ZER0)>0 :
        ip_patches_good.append(ips_ring[msk[ips_ring]>ZER0])

plotmap=-1.*np.ones(hp.nside2npix(NSIDE_DEFAULT))
for i,ips in enumerate(ip_patches_good) :
    plotmap[ips]=i

#Zero mask outside edges for safety
ids_bad=np.where(msk<ZER0)[0]
msk[ids_bad]=0.
if OUTPUT_LEVEL>0 :
    hp.write_map(predir0+'mask.fits',msk,overwrite=True)

if VERBOSE :
    print("Cleaning")
spec_i=np.zeros([2,hp.nside2npix(NSIDE_DEFAULT)]);
spec_o=np.zeros([2,2,hp.nside2npix(NSIDE_DEFAULT)]);
amps_o=np.zeros([2,3,2,hp.nside2npix(NSIDE_DEFAULT)]);
cova_o=np.zeros([2,6,2,hp.nside2npix(NSIDE_DEFAULT)]);
sampler_args = {
    "method" : 'Powell',
    "tol" : None,
    "callback" : None,
    "options" : {'xtol':1E-4,'ftol':1E-8,'maxiter':None,'maxfev':None,'direc':None}
    }
for i,ips in enumerate(ip_patches_good) :
    npix_here=len(ips)
    if VERBOSE>1 :
        print("Patch %d, %d pixels: "%(i,npix_here))
    bs=beta_sync_fid; bd=beta_dust_fid; td=temp_dust_fid; cs=curv_sync_fid;
    sbs=1.0; sbd=1.0; std=5.; scs=0.5
    spec_i[0,ips]=bs; spec_i[1,ips]=bd
    for isim in [0,1] :
        if VERBOSE>1 :
            print(" Cleaning region %d"%(isim+1))
        if FGRM_TYP==0 :
            fixed_pars={'nu_ref_d':nu_ref_dust_p,'nu_ref_s':nu_ref_sync_p,'T_d':td,'beta_c':cs}
            var_pars=['beta_s','beta_d']
            var_prior_mean=[bs,bd]
            var_prior_width=[sbs,sbd]
        elif FGRM_TYP==1 :
            fixed_pars={'nu_ref_d':nu_ref_dust_p,'nu_ref_s':nu_ref_sync_p,'beta_c':cs}
            var_pars=['beta_s','beta_d','T_d']
            var_prior_mean=[bs,bd,td]
            var_prior_width=[sbs,sbd,std]
        elif FGRM_TYP==2 :
            fixed_pars={'nu_ref_d':nu_ref_dust_p,'nu_ref_s':nu_ref_sync_p}
            var_pars=['beta_s','beta_d','T_d','beta_c']
            var_prior_mean=[bs,bd,td,cs]
            var_prior_width=[sbs,sbd,std,scs]
        ml=mpl.MapLike({'data':mps_d[isim][:,ips,:],'noisevar':mps_nv[:,ips,:]**2,
                        'fixed_pars':fixed_pars,
                        'var_pars':var_pars,
                        'var_prior_mean':var_prior_mean,
                        'var_prior_width':var_prior_width,
                        'var_prior_type':['tophat' for b in var_pars]},
                       sky_true,instrument)
        if W_FWHM :
            ml_rw=mpl.MapLike({'data':mps_d_rw[isim][:,ips,:],'noisevar':mps_nv[:,ips,:]**2,
                               'fixed_pars':fixed_pars,
                               'var_pars':var_pars,
                               'var_prior_mean':var_prior_mean,
                               'var_prior_width':var_prior_width,
                               'var_prior_type':['tophat' for b in var_pars]},
                              sky_true,instrument)
        rdict=clean_pixels(ml,run_minimize,**sampler_args)
        if VERBOSE>1 :
            print("  - %d function evaluations"%(rdict['ML_nev']))
        if W_FWHM :
            amps=ml_rw.get_amplitude_mean(rdict['params_ML']).reshape([2,npix_here,3])
            cova=np.linalg.inv(ml_rw.get_amplitude_covariance(rdict['params_ML'])).reshape([2,npix_here,3,3])
        else :
            amps=ml.get_amplitude_mean(rdict['params_ML']).reshape([2,npix_here,3])
            cova=np.linalg.inv(ml.get_amplitude_covariance(rdict['params_ML'])).reshape([2,npix_here,3,3])
        spec_o[isim,0][ips]=rdict['params_ML'][0]; spec_o[isim,1][ips]=rdict['params_ML'][1];
        amps_o[isim][:,:,ips]=np.transpose(amps,axes=[2,0,1])
        cova_o[isim,0][:,ips]=cova[:,:,0,0]
        cova_o[isim,1][:,ips]=cova[:,:,1,1]
        cova_o[isim,2][:,ips]=cova[:,:,2,2]
        cova_o[isim,3][:,ips]=cova[:,:,0,1]
        cova_o[isim,4][:,ips]=cova[:,:,1,2]
        cova_o[isim,5][:,ips]=cova[:,:,0,2]

if VERBOSE :
    print("Writing output")
if OUTPUT_LEVEL>1 :
    hp.write_map(predir0+'specs_i.fits',spec_i,overwrite=True)
    for isim in [0,1] :
        hp.write_map(predir+'specs_o_sim%d.fits'%(isim+1),spec_o[isim],overwrite=True)
        hp.write_map(predir+'sync_sim%d.fits'%(isim+1),amps_o[isim,0],overwrite=True)
        hp.write_map(predir+'esync_sim%d.fits'%(isim+1),cova_o[isim,0],overwrite=True)
        hp.write_map(predir+'dust_sim%d.fits'%(isim+1),amps_o[isim,1],overwrite=True)
        hp.write_map(predir+'edust_sim%d.fits'%(isim+1),cova_o[isim,1],overwrite=True)
if OUTPUT_LEVEL>0 :
    for isim in [0,1] :
        hp.write_map(predir+'cmb_sim%d.fits'%(isim+1) ,amps_o[isim,2],overwrite=True)
        hp.write_map(predir+'ecmb_sim%d.fits'%(isim+1) ,cova_o[isim,2],overwrite=True)

pmsk=np.zeros_like(msk); pmsk[ids_bad]=hp.UNSEEN
if VERBOSE>1 :
    for ic in [0,1,2] :
        for isim in [0,1] :
            hp.mollview(amps_o[isim,ic,0]+pmsk);
            hp.mollview(np.sqrt(cova_o[isim,ic,0])+pmsk)
        plt.show()
    for ic in [0,1] :
        hp.mollview(spec_i[ic]+pmsk)
        for isim in [0,1] :
            hp.mollview(spec_o[isim,ic]+pmsk)
        plt.show()

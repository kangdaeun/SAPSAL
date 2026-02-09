#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:40:32 2025

@author: daeun

Hydrogen slab model codes
Assumptions: a plane-parallel slab composed of pure hydrogen assumed in LTE

Hydrogen emission: free-bound radiation (recombination) & free-free radiation

"""

import numpy as np
import astropy.constants as cst
import astropy.units as u

from numba import njit, prange # for faster get_gaunt_fb

h_cgs = cst.h.cgs.value
k_b_cgs = cst.k_B.cgs.value
c_cgs = cst.c.cgs.value
m_e = cst.m_e.cgs.value

nu_0 = 3.28795e15 # [Hz] from Manara. ionization frequence for hydrogen
lam_0 = 1.6419 # [um] photo-detachment threshold wavelength for H-
alpha_coeff = 1.439 * 1e8 # [AA K] ? okay to use AA?
alpha_coeff = alpha_coeff *1e-4 # [um K] ? 
# alpha_coeff = alpha_coeff *1e-8 # [um K] ? 

# Alwasy use array for nu. values for Tslab, ne, Zi, etc
# nu [Hz], Tslab [K], lam [um]


#
def wl_to_freq(wl, unit=u.AA):
    return (cst.c/(wl*unit)).to(u.Hz).value

def freq_to_wl(nu, unit=u.AA):
    return (cst.c/(nu*u.Hz)).to(unit).value

def convert_nu_to_lam(Xnu, nu, unit=u.AA):
    return (Xnu * (nu*nu/c_cgs)*((1/u.cm).to(1/unit)).value)

def convert_lam_to_nu(Xlam, lam, unit=u.AA):
    lam_cgs = (lam*unit).to(u.cm).value
    return Xlam * ((1/unit).to(1/u.cm).value)  * (lam_cgs*lam_cgs/c_cgs)


def planck_nu(nu, T, **kwarg): # Bv(T)
    return ( 2*h_cgs*(nu**3)/(c_cgs*c_cgs) /(np.exp(h_cgs*nu/(k_b_cgs*T))-1) )

def planck_lam(lam, T, **kwarg): # Bv(T): only use cm lambda
    return ( 2*h_cgs*(c_cgs**2)/(lam**5) /(np.exp(h_cgs*c_cgs/(k_b_cgs*T*lam))-1) )


def get_gaunt_ff(nu, Tslab, Zi=1, **kwarg):
    # Eq. 2.17
    comp1 = (nu/(nu_0*Zi*Zi))**(1/3)
    comp2 = (k_b_cgs * Tslab / (h_cgs*nu))
    return ( 1+ 0.1728*comp1*(1+2*comp2) - 0.0496*comp1*comp1*(1+2/3*comp2 + 4/3*comp2*comp2) )

@njit(cache=True)
def get_qtm(nu, Zi=1):
    val = np.floor(np.sqrt(nu_0*Zi*Zi/nu))+1
    return val.astype(np.int64)

# def get_gaunt_fb(nu, Tslab, Zi=1, stop=1e-8, stop_num=1000):
#     # Eq 2.13 + 2.14 + 2.15
#     # stop: sys.float_info.epsilon = 2.220446049250313e-16
#     const1 = h_cgs * nu_0 / (k_b_cgs * Tslab)
    
#     # sigma
#     qtm = get_qtm(nu, Zi=Zi) # for multiple nu this will be also an array with same dimension: dim(nu) = dim(qtm)
#     gaunt_fb = np.zeros(len(nu))
#     for i, (nu_i, m_i) in enumerate(zip(nu, qtm)):
        
#         # summed = 0.
#         # n = float(m_i)
#         # resi = 1
#         # while resi > stop:
#         # # for n in range(m_i, m_i):
#         #     comp = (nu_i/(nu_0*Zi*Zi))**(1/3)
#         #     gn = (1 + 0.1728*(comp - 2/((n*comp)**2)) 
#         #           - 0.0496*( comp*comp - 2/(3*n*n*comp) + 2/3/((n*comp)**4) ) )
#         #     vv = np.exp(h_cgs *nu_0/(n*n*k_b_cgs*Tslab)) *(n**(-3)) * gn
#         #     # print(vv)
#         #     if n==m_i:
#         #         resi = 1
#         #     else:
#         #         resi = (vv)/summed
#         #     summed += vv
#         #     n+=1

#         na = np.arange(m_i, stop_num).astype(np.float64)
#         comp = (nu_i/(nu_0*Zi*Zi))**(1/3)
#         gna = (1 + 0.1728*(comp - 2/((na*comp)**2)) 
#                   - 0.0496*( comp*comp - 2/(3*na*na*comp) + 2/3/((na*comp)**4) ) )
#         vv = np.exp(h_cgs *nu_0/(na*na*k_b_cgs*Tslab)) *(na**(-3)) * gna
#         summed = np.sum(vv)      
#         gaunt_fb[i] = summed
#     return (2*const1*Zi*Zi)*gaunt_fb  


##  Faster version using numba
# @njit(parallel=True, cache=True)
@njit(cache=True)
def get_gaunt_fb(nu, Tslab, Zi=1, stop_num=1000):
    # Eq 2.13 + 2.14 + 2.15
    # stop: sys.float_info.epsilon = 2.220446049250313e-16
    const1 = h_cgs * nu_0 / (k_b_cgs * Tslab)
    
    # sigma
    qtm = get_qtm(nu, Zi=Zi) # for multiple nu this will be also an array with same dimension: dim(nu) = dim(qtm)
    gaunt_fb = np.zeros(len(nu))
    for i in range(len(nu)):
    # for i, (nu_i, m_i) in enumerate(zip(nu, qtm)): # cannot use enumerate or zip in numba prange
        nu_i = nu[i]
        m_i = qtm[i] 
        na = np.arange(m_i, stop_num).astype(np.float64) # specific numby type for numba
        comp = (nu_i/(nu_0*Zi*Zi))**(1/3)
        gna = (1 + 0.1728*(comp - 2/((na*comp)**2)) 
                  - 0.0496*( comp*comp - 2/(3*na*na*comp) + 2/3/((na*comp)**4) ) )
        vv = np.exp(h_cgs *nu_0/(na*na*k_b_cgs*Tslab)) *(na**(-3)) * gna
        summed = np.sum(vv)

        gaunt_fb[i] = summed

    return (2*const1*Zi*Zi)*gaunt_fb  
    

# 1. Hydrogen emissivity
# Eq. 2.18: total emissivity: fb + ff <- by adding two gaunt factor (2.13(+2.15) + 2.17)
def get_H_emissivity(nu, Tslab, ne, Zi=1, **kwarg):
    return (5.44e-39 * np.exp(-h_cgs * nu /(k_b_cgs*Tslab)) * 
            (Zi*Zi)/np.sqrt(Tslab) * (ne*ne) * 
            (get_gaunt_ff(nu, Tslab, **kwarg) + get_gaunt_fb(nu, Tslab, **kwarg)) )

def get_tauH(nu, Tslab, ne, tau_sp, wl_sp=3000, Zi=1, wl_sp_unit=u.AA,  Lslab=None, **kwarg):
    if Lslab is None:
        Lslab = get_Lslab(tau_sp, Tslab, ne, wl_sp=wl_sp,  Zi=Zi, wl_sp_unit=wl_sp_unit, **kwarg)
    
    return get_H_emissivity(nu, Tslab, ne, Zi=Zi, **kwarg)*Lslab/planck_nu(nu, Tslab)


# This is only for one value
# tau_sp is given at specific "wavelength" : it is tau_lambda value at specific lamdba (wl_sp)
def get_Lslab(tau_sp, Tslab, ne, wl_sp=3000,  Zi=1, wl_sp_unit=u.AA, **kwarg):
    # calcualted Lslab is  in [cm] only
    
    # Lslab is only based on optical depth of a pure hydrogen slab -> only use j of H
    # nu_sp = np.array([wl_to_freq(wl_sp, unit=unit)])
    # return (tau_sp  * planck_nu(nu_sp, Tslab)/get_H_emissivity(nu_sp, Tslab, ne, Zi=Zi, **kwarg))[0]

    nu_sp = np.array([wl_to_freq(wl_sp, unit=wl_sp_unit)])
    # Use frequency dependent variables. keep tau same as input (input tau is defined at wavelength)
    return (tau_sp * planck_nu(nu_sp, Tslab) / get_H_emissivity(nu_sp, Tslab, ne, Zi=Zi, **kwarg) )[0]
    # Change everthing in lambda dependent variables
    # return (tau_sp  * planck_lam(np.array([(wl_sp*wl_sp_unit).to(u.cm).value]), Tslab)/
    #         (get_H_emissivity(nu_sp, Tslab, ne, Zi=Zi, **kwarg)*nu_sp*nu_sp/c_cgs ))[0] # converted j_nu to j_lam and used tau_lam and B(T)_lam
    

# -------------------------------------------------------------
# Hn _lam (/um)
    
def get_nH_n(n, Tslab, ne, **kwarg): # nH at n-th level (number density of atoms in bound state, n-th level, with LTE condition)
    # Eq 2.9
    return ( ((h_cgs/np.sqrt(2*np.pi*m_e*k_b_cgs))**3) *(n*n)*(Tslab**(-1.5))
            *np.exp(h_cgs*nu_0/(n*n*k_b_cgs*Tslab))
            *ne*ne)

def get_pd_cross(lam, Tslab, ne, **kwarg): # lam should  be in um
    # Eq. 2.27, 2.28 Table 2.1
    # Wl restriction: only when 0.125 um <= lam <= 1.6419 = lam_0
    Cn = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])
    summed = 0
    for n in range(1,6+1):
        summed += ( Cn[n-1]*( (1/lam - 1/lam_0)**(0.5*n - 0.5))) 
    
    return 1e-18 * (lam**3) * ((1/lam - 1/lam_0)**(1.5))*summed

# Kappa in invalid wl are all zero
def get_Kabs_fb(lam, Tslab, ne, **kwarg): # lam should  be in um
    # Eq 2.26, 2.27, 2.28, Table 2.1 (x n_Hn)
    # Wl restriction: only when 0.125 um <= lam < 1.6419 = lam_0  
    # const1 = alpha_coeff/lam_0/Tslab  
    valid = np.logical_and( (lam>=0.125), (lam<1.6419) ) 
    if np.sum(valid)==0:
        return np.zeros(len(lam))
    output = np.zeros(len(lam))
    output[valid] = (0.750 * (Tslab**(-2.5))*
            np.exp(alpha_coeff/lam_0/Tslab)*(1-np.exp(-alpha_coeff/lam[valid]/Tslab))
            *get_pd_cross(lam[valid], Tslab, ne) )
    return output
    
def get_Kabs_ff(lam, Tslab, ne, **kwarg):
    # Eq. 2.29
    # Wl restriction: only when lam>0.182um &  0.5 <= 5040/T <= 3.6 : 1400 ~ 10800
    # Table 2.2 for 0.182 < lam < 0.3645
    # Table 2.3 for lam >= 0.3645
    
    valid_kff_T = np.logical_and( (5040/Tslab >=0.5), (5040/Tslab <= 3.6))
    if valid_kff_T is False: # temperature outside the range
        return np.zeros(len(lam))
    
    Param1 = np.array(
             [ [518.1012, -734.8667, 1021.1775, -479.0721, 93.1373, -6.4285],
               [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600],
               [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
               [115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             ])
    Param2 =np.array(
               [ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [2483.3460, 285.8270, -2054.2910, 2827.7760, -1341.5370, 208.9520],
               [-3449.8890, -1158.3820, 8746.5230, -11485.6320, 5303.6090, -812.9390],
               [2200.0400, 2427.7190, -13651.1050, 16755.5240, -7510.4940, 1132.7380],
               [-696.2710, -1841.4000, 8624.9700, -10051.5300, 4400.0670, -655.0200],
               [88.2830, 444.5170, -1863.8640, 2095.2880, -901.7880, 132.9850],
             ])
        
    valid1 = np.logical_and( lam>0.182, lam<0.3645)
    valid2 = (lam >= 0.3645)
    # invalid = lam <= 0.182
    
    summed = np.zeros(len(lam))
    for n in range(0,6):
        const = (5040/Tslab)**(0.5*((n+1)+1))
        # const = (5040/Tslab)**(0.5*(n+1))
        
        summed[valid1] += (const*
                           (lam[valid1]*lam[valid1]*Param1[n,0] + Param1[n,1]
                           + Param1[n,2]/lam[valid1] + Param1[n,3]/(lam[valid1]**2)
                           + Param1[n,4]/(lam[valid1]**3)
                           + Param1[n,5]/(lam[valid1]**4) ) )
                           # * get_nH_n(n+1, Tslab, ne) )
        summed[valid2] += (const*
                            (lam[valid2]*lam[valid2]*Param2[n,0] + Param2[n,1]
                           + Param2[n,2]/lam[valid2] + Param2[n,3]/(lam[valid2]**2)
                           + Param2[n,4]/(lam[valid2]**3)
                           + Param2[n,5]/(lam[valid2]**4) ))
                           # * get_nH_n(n+1, Tslab, ne) )
    return 1e-29 * summed
    

    
def get_Hn_abscoeff(lam, Tslab, ne, **kwarg): # lam should  be in um
    # Eq 2.30
#     valid_kfb = np.logical_and( (lam>=0.125), (lam<=1.6419) ) # valid wl range for kappa bf
#     valid_kfb = lam > 0.182
    n_Hsum = 0
    # for n in range(1,6+1):
    for n in range(1,1+1):
        n_Hsum += get_nH_n(n, Tslab, ne)
    # n_Hsum = get_nH_n(1, Tslab, ne)
    return ( get_Kabs_fb(lam, Tslab, ne, **kwarg) + get_Kabs_ff(lam, Tslab, ne, **kwarg)) * ne * n_Hsum * k_b_cgs * Tslab

    # kapp_tot = ne k T (k_fb*n_H + k_ff*n_H) nH already in each kabs
    # return ( get_Kabs_fb(lam, Tslab, ne, **kwarg) + get_Kabs_ff(lam, Tslab, ne, **kwarg)) * ne  * k_b_cgs * Tslab

def get_tauHn(nu, Tslab, ne, tau_sp, wl_sp=3000, Zi=1, wl_sp_unit=u.AA, Lslab=None, **kwarg):  # tau_lam (1/cm)
    if Lslab is None:
        Lslab = get_Lslab(tau_sp, Tslab, ne, wl_sp=wl_sp,  Zi=Zi, wl_sp_unit=wl_sp_unit, **kwarg)
    return get_Hn_abscoeff( freq_to_wl(nu, unit=u.um), Tslab, ne, **kwarg) * Lslab # tau_lam (1/cm)
    # abscoeff_lam = get_Hn_abscoeff( freq_to_wl(nu, unit=u.um), Tslab, ne, **kwarg) # 
    # return convert_lam_to_nu(abscoeff_lam, lam, unit=u.AA) * Lslab #* (c_cgs/nu/nu) # convert to dlam -> dnu
    

#------------------------------------------------------------------------------------------------

def get_total_intensity(nu_lam, Tslab, ne, tau_sp, wl_sp=3000, Zi=1, wl_sp_unit=u.AA, Lslab=None, include_Hn=True, 
                        Int_lam=False, lam_unit=u.nm, **kwarg): 
    # By default, expect freqeuncey and Int_nu (1/Hz)
    # Only if Int_lam=True, regard nu_lam as lam with lam_unit (1/lam_unit)
    # Calculations are based on freqeuncy
    if Int_lam:
        nu = wl_to_freq(nu_lam, unit=lam_unit)
    else:
        nu = nu_lam
    
    if Lslab is None:
        Lslab = get_Lslab(tau_sp, Tslab, ne, wl_sp=wl_sp,  Zi=Zi, wl_sp_unit=wl_sp_unit, **kwarg)

    
    tau_H = get_tauH(nu, Tslab, ne, tau_sp, wl_sp=wl_sp, Zi=Zi, wl_sp_unit=wl_sp_unit,  Lslab=Lslab, **kwarg)

    if include_Hn:
        tau_Hn = get_tauHn(nu, Tslab, ne, tau_sp, wl_sp=wl_sp, Zi=Zi, wl_sp_unit=wl_sp_unit,  Lslab=Lslab, **kwarg)
        tau_tot = tau_H + tau_Hn
    else:
        tau_tot = tau_H

    # 간단버젼
    planck = planck_nu(nu, Tslab)    
    # Int = planck *  (1-np.exp(-tau_H))
    Int = planck * (1-np.exp(-tau_tot))
    # beta = (1-np.exp(-tau_tot))/tau_tot
    # Int = tau_tot* planck * beta
    
    # 풀어쓰는 버젼
    # planck = planck_nu(nu, Tslab)
    # j_H = get_H_emissivity(nu, Tslab, ne, Zi=Zi, **kwarg) 
    # tau_H = j_H*Lslab/planck
    # beta_H = (1-np.exp(-tau_H))/tau_H
    # Int = j_H * Lslab * beta_H
    
    # if include_Hn:
    #     # 간단버젼
    #     tau_Hn = get_tauHn(nu, Tslab, ne, tau_sp, wl_sp=wl_sp, Zi=Zi, wl_sp_unit=wl_sp_unit,  Lslab=Lslab, **kwarg)
    #     Int += ( planck *  (1-np.exp(-tau_Hn)) )
        
        # 풀어쓰는 버젼
        # Kappa_Hn_tot = get_Hn_abscoeff( freq_to_wl(nu, unit=u.um), Tslab, ne, **kwarg) #* Lslab
        # tau_Hn = Kappa_Hn_tot * Lslab
        # j_Hn = Kappa_Hn_tot * planck
        # beta_Hn = (1-np.exp(-tau_Hn))/tau_Hn
        # Int += (j_Hn * Lslab * beta_Hn)
        
        
    if Int_lam:
        return convert_nu_to_lam(Int, nu, unit=lam_unit) # Int_lam
    else:
        return Int    

    
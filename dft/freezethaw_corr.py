#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
freezethaw_corr.py

Correlated freeze-thaw PRBE variants.  The active subsystem is optimized at the
embedded mean-field level, then a cheap DF-MP2 density is formed on the active
fragment and used to relax the bath in the next freeze-thaw step.
"""

import numpy as np
from pyscf import scf, dft, df
from pyscf.mp import dfmp2_native
from pyscf.lib import logger

from pyscf_embedding.dft.freezethaw import (
    _bath_relaxed_prbe_build,
    _filter_restricted_active_mos,
    _run_active_freeze_thaw_step,
    bath_fixed_active_kernel,
)
from pyscf_embedding.dft.vprbe import rVPRBE


def _active_only_restricted_copy(mf_A, ovlp, moC_occ_B, moC_vir_B,
                                 epsilon=1e-5):
    """Return an MF copy whose MO arrays contain only active-space orbitals."""
    mf_active = mf_A.copy()
    bath_orbs = np.hstack([moC_occ_B, moC_vir_B])
    if bath_orbs.shape[1] == 0:
        return mf_active

    mask_act = np.linalg.norm(mf_active.mo_coeff.T @ ovlp @ bath_orbs,
                              axis=1) <= epsilon
    mf_active.mo_energy = mf_active.mo_energy[mask_act]
    mf_active.mo_coeff = mf_active.mo_coeff[:, mask_act]
    mf_active.mo_occ = mf_active.mo_occ[mask_act]

    nao, nmo = mf_active.mo_coeff.shape
    if nmo < nao:
        npad = nao - nmo
        mf_active.mo_coeff = np.hstack([
            mf_active.mo_coeff, np.zeros((nao, npad), dtype=mf_active.mo_coeff.dtype)
        ])
        pad_energy = np.full(npad, 1e12, dtype=mf_active.mo_energy.dtype)
        mf_active.mo_energy = np.hstack([mf_active.mo_energy, pad_energy])
        mf_active.mo_occ = np.hstack([mf_active.mo_occ, np.zeros(npad)])
    return mf_active


def _run_dfmp2_density(mf_A, ovlp, moC_occ_B, moC_vir_B,
                       relaxed=True, mp2_kwargs=None):
    """
    Run MP2 on active-only embedded orbitals and return the AO correlated DM.

    Native density-fitted MP2 is used because this calculation is repeated at
    every freeze-thaw macrocycle.  Relaxed MP2 density matrices are the default.
    """
    if mp2_kwargs is None:
        mp2_kwargs = {}

    mf_mp2 = _active_only_restricted_copy(
        mf_A, ovlp, moC_occ_B, moC_vir_B
    )
    pt = dfmp2_native.DFMP2(mf_mp2, **mp2_kwargs)
    pt.run()

    dm_corr = pt.make_rdm1(relaxed=relaxed, ao_repr=True)
    dm_corr = 0.5 * (dm_corr + dm_corr.T)
    return dm_corr, pt


class rVPRBEFreezeThawCorr(rVPRBE):
    """
    Restricted correlated freeze-thaw one-density PRBE.

    Each macrocycle:
      1. optimizes the active determinant with fixed bath density;
      2. runs active-only DF-MP2 and builds the correlated active density;
      3. relaxes the bath using that correlated active density.

    Macro convergence requires both the total correlated embedding energy and
    the MP2 active density to be unchanged.
    """

    def make_bath_space(self):
        return np.hstack([self.moC_occ_B, self.moC_vir_B])

    def kernel(self, xc_embed=None, ecp_embed=None, pseudo_embed=None,
               max_macro=20, max_cycle_A=None, max_cycle_B=None,
               conv_tol_macro=None, conv_tol_grad=None,
               damping_B=0.0, final_active=True, relaxed_mp2_dm=True,
               mp2_kwargs=None, **kwargs):
        mf_B = self.mf_B
        ovlp = mf_B.get_ovlp()
        mol = mf_B.mol.copy()
        nocc_A = self.moC_occ_A.shape[1]
        nocc_B = self.moC_occ_B.shape[1]
        mol.nelectron = int(2 * nocc_A)

        if conv_tol_macro is None:
            conv_tol_macro = 1e-8
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol_macro)

        dm_A = self.make_rdmA_init()
        dm_A_corr = np.array(dm_A, copy=True)
        dm_B = self.make_rdmB_fixed()

        if xc_embed is None:
            self.mf_A = scf.RHF(mol, **kwargs)
        else:
            self.mf_A = dft.RKS(mol)
            self.mf_A.xc = xc_embed

        if hasattr(mf_B, 'with_df'):
            self.mf_A = df.density_fit(self.mf_A)
            self.mf_A.with_df.auxbasis = mf_B.with_df.auxbasis

        mf_A = self.mf_A
        mf_A.diis_space = mf_B.diis_space
        mf_A.diis_damp = mf_B.diis_damp
        mf_A.max_cycle = mf_B.max_cycle
        mf_A.verbose = mf_B.verbose

        P_occ_B = self.moC_occ_B @ self.moC_occ_B.T
        P_vir_B = self.moC_vir_B @ self.moC_vir_B.T
        space_B = self.make_bath_space()

        _, _, _, _, _, _, e_prbe = _bath_relaxed_prbe_build(
            mf_A, mf_B, dm_A_corr, dm_B
        )
        e_mp2_corr = 0.0
        e_tot = e_prbe
        logger.info(mf_A, 'init correlated freeze-thaw PRBE E= %.15g', e_tot)

        scf_conv = False
        norm_ddm_A = np.inf
        norm_gorb_A = np.inf
        self.mp2_A = None

        for macro in range(max_macro):
            dm_A_corr_last = np.array(dm_A_corr, copy=True)
            last_e = e_tot

            dm_A, norm_gorb_A, _ = _run_active_freeze_thaw_step(
                mf_A, mf_B, dm_A, dm_B, P_occ_B, P_vir_B,
                conv_tol=conv_tol_macro, conv_tol_grad=conv_tol_grad,
                max_cycle=max_cycle_A
            )

            dm_A_corr, self.mp2_A = _run_dfmp2_density(
                mf_A, ovlp, self.moC_occ_B, self.moC_vir_B,
                relaxed=relaxed_mp2_dm, mp2_kwargs=mp2_kwargs
            )
            e_mp2_corr = self.mp2_A.e_corr

            _, _, self.mo_energy_B, self.mo_coeff_B, self.mo_occ_B, dm_B = \
                bath_fixed_active_kernel(
                    mf_low=mf_B, dm_A_fixed=dm_A_corr, space_B=space_B,
                    nocc_B=nocc_B, dm_B0=dm_B, unrestricted=False,
                    conv_tol=conv_tol_macro, conv_tol_grad=conv_tol_grad,
                    max_cycle=max_cycle_B, damping=damping_B
                )

            # Rebuild Huzinaga projectors from the relaxed bath occupations.
            mask_occ = self.mo_occ_B > 0.5
            P_occ_B = self.mo_coeff_B[:, mask_occ] @ self.mo_coeff_B[:, mask_occ].T
            P_vir_B = self.mo_coeff_B[:, ~mask_occ] @ self.mo_coeff_B[:, ~mask_occ].T

            _, _, _, _, _, _, e_prbe = _bath_relaxed_prbe_build(
                mf_A, mf_B, dm_A, dm_B
            )
            e_tot = e_prbe + e_mp2_corr
            norm_ddm_A = np.linalg.norm(dm_A_corr - dm_A_corr_last)

            logger.info(
                mf_A,
                'macro= %d correlated freeze-thaw PRBE E= %.15g  '
                'delta_E= %4.3g  E_MP2= %.15g  |g_A|= %4.3g  |ddm_A_corr|= %4.3g',
                macro + 1, e_tot, e_tot - last_e, e_mp2_corr,
                norm_gorb_A, norm_ddm_A
            )

            if abs(e_tot - last_e) < conv_tol_macro and norm_gorb_A < conv_tol_grad:
                scf_conv = True
                break

        if final_active:
            dm_A_corr_prev = np.array(dm_A_corr, copy=True)
            dm_A, norm_gorb_A, _ = _run_active_freeze_thaw_step(
                mf_A, mf_B, dm_A, dm_B, P_occ_B, P_vir_B,
                conv_tol=conv_tol_macro, conv_tol_grad=conv_tol_grad,
                max_cycle=max_cycle_A
            )
            dm_A_corr, self.mp2_A = _run_dfmp2_density(
                mf_A, ovlp, self.moC_occ_B, self.moC_vir_B,
                relaxed=relaxed_mp2_dm, mp2_kwargs=mp2_kwargs
            )
            e_mp2_corr = self.mp2_A.e_corr
            _, _, _, _, _, _, e_prbe = _bath_relaxed_prbe_build(
                mf_A, mf_B, dm_A, dm_B
            )
            e_tot = e_prbe + e_mp2_corr
            norm_ddm_A = np.linalg.norm(dm_A_corr - dm_A_corr_prev)

        self.e_prbe = e_prbe
        self.e_corr = e_mp2_corr
        self.e_tot = e_tot
        self.dm_A = dm_A
        self.dm_A_corr = dm_A_corr
        self.dm_B = dm_B

        _filter_restricted_active_mos(
            mf_A, ovlp, self.moC_occ_B, self.moC_vir_B
        )
        mf_A.e_tot = e_tot
        mf_A.converged = scf_conv
        mf_A.prbe_dm_A = dm_A
        mf_A.prbe_dm_A_corr = dm_A_corr
        mf_A.prbe_dm_B = dm_B
        mf_A.prbe_e_prbe = e_prbe
        mf_A.prbe_e_corr = e_mp2_corr
        mf_A.prbe_mo_energy_B = getattr(self, 'mo_energy_B', None)
        mf_A.prbe_mo_coeff_B = getattr(self, 'mo_coeff_B', None)
        mf_A.prbe_mo_occ_B = getattr(self, 'mo_occ_B', None)

        self.e_tot -= self.calc_disp_embed()
        mf_A.e_tot = self.e_tot
        logger.info(
            mf_A,
            'final correlated freeze-thaw PRBE E= %.15g  E_PRBE= %.15g  '
            'E_MP2= %.15g  |g_A|= %4.3g  |ddm_A_corr|= %4.3g',
            self.e_tot, self.e_prbe, self.e_corr, norm_gorb_A, norm_ddm_A
        )
        return self.e_tot, mf_A.mo_energy, mf_A.mo_coeff, mf_A.mo_occ
#%%
if __name__ == '__main__':
    from pyscf_embedding.local import regional as re
    from pyscf_embedding.utils import chemcore
    from pyscf import gto, scf, cc, grad
    from pyscf_embedding.dft.prbe import rPRBE, uPRBE

    coords1 = \
    """
      H           1.83033035325714     -1.55914459515730      0.85018703157270
      C           0.90263631613210     -0.08578242736390      0.14894116637203
      O           1.92201300538061      0.51665324420946     -0.08503290170897
      O           0.90171075857076     -1.32072671224472      0.70739805133646
      C          -0.44742060897905      0.43887835522198     -0.13930696034172
      C          -1.57631378519914     -0.21501349238018      0.11262246365465
      H          -0.44354482529961      1.42537195115242     -0.58435608510154
      H          -1.57016561414080     -1.20100172981207      0.55763852778426
      H          -2.53835159972200      0.22238440637431     -0.12155029356787
    """
    
    coords2 = \
    """
      C           0.93501939826289     -0.16731316159968      0.18715060054646
      O           1.88694577197618      0.59099117395221     -0.12029735496945
      O           0.96364675571253     -1.30644525101509      0.70374527602260
      C          -0.45331566554332      0.41330164264060     -0.12783548248701
      C          -1.60070946799917     -0.21426796566953      0.11128233993953
      H          -0.44856062322574      1.40525848237891     -0.57525090759213
      H          -1.57811875566373     -1.20278092786669      0.55806286649746
      H          -2.56689441351964      0.22480300717926     -0.12390333795747
    """
    
    # coords1 = \
    # """
    #   H          -2.33853423528445     -1.80445188593265      0.11651648020221
    #   O          -2.34422459255885     -0.85278916752672     -0.03576411026300
    #   C          -1.06258456679029     -0.37939268607129     -0.01728490895882
    #   C           0.03672552600454     -1.20674427294869      0.19587817315120
    #   C          -0.87933383150076      0.98550108666543     -0.22229854535058
    #   C           1.31693787084500     -0.66574822089220      0.20334230293598
    #   C           0.40339408165426      1.51340780309772     -0.21253603009361
    #   C           1.50876316862276      0.69480639109537     -0.00029246918990
    #   H          -1.74545802184674      1.61252309509986     -0.38639091454627
    #   H          -0.10850689384867     -2.26985861428912      0.35517200444118
    #   H           2.16662091581719     -1.31570774357644      0.36979039101203
    #   H           0.53987391557788      2.57552108882858     -0.37240095762938
    #   H           2.50632566330812      1.11293312645015      0.00626858428895
    # """
    
    # coords2 = \
    # """
    #   Cl         -2.66552880321122     -1.04648920352946     -0.02858438211425
    #   C          -1.05418475844935     -0.37402118589571     -0.01754899463371
    #   C          -0.88351585053082      0.98806033343056     -0.22302996160454
    #   C           0.03181906088976     -1.21156891873802      0.19626721449624
    #   C           0.40161946228383      1.51651330760525     -0.21314994006458
    #   C           1.31127908283482     -0.66945895604463      0.20350987507472
    #   C           1.50050564548318      0.69236009337727     -0.00049441681239
    #   H          -0.12528181587270     -2.26933992568938      0.35382176314591
    #   H          -1.74412056922073      1.62100921590816     -0.38766608487821
    #   H           0.54020920533309      2.57808841901490     -0.37274437771367
    #   H           2.16154435858189     -1.31831745634710      0.36981403020102
    #   H           2.49900198187823      1.10907027690817      0.00636727490347
    # """
    
    #%%
    ##### Calculation One, Test rPRBE
    basis='ccpvdz'
    xc = 'pbe'
    #%%
    # run CC
    mol1 = gto.Mole(atom=coords1,basis=basis,spin=0,verbose=4)
    mf1 = mol1.RHF(max_cycle=100).density_fit().run()
    mycc1 = cc.CCSD(mf1,frozen=chemcore(mol1)).density_fit()
    mycc1 = mycc1.run()
    ecc1 = mycc1.e_tot
    
    mol2 = gto.Mole(atom=coords2,basis=basis,spin=0,verbose=4,charge=-1)
    mf2 = mol2.RHF(max_cycle=100).density_fit().run()
    mycc2 = cc.CCSD(mf2,frozen=chemcore(mol2)).density_fit()
    mycc2 = mycc2.run()
    ecc2 = mycc2.e_tot
    
    del mycc1, mycc2
    #%%
    # run DFT
    mol1 = gto.Mole(atom=coords1,basis=basis,spin=0,verbose=4)
    mf1 = mol1.RKS(xc=xc,max_cycle=100).density_fit().run()
    edft1 = mf1.e_tot
    
    mol2 = gto.Mole(atom=coords2,basis=basis,spin=0,verbose=4,charge=-1)
    mf2 = mol2.RKS(xc=xc,max_cycle=100).density_fit().run()
    edft2 = mf2.e_tot
    
    #%%
    ##### standard PRBE
    
    # run embedding 1
    occ_calc = re.RegionalActiveSpace(mf1, np.arange(0,4), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf1, np.arange(0,4), 'vir', basis=mol1.basis, cutoff=1e-6)
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed1 = rPRBE(mf1, moC_occ, moC_vir, mask_occ_act, mask_vir_act)
    e_mf,mo_energy,mo_coeff,mo_occ = embed1.kernel(xc_embed=None)
    mycc1 = cc.CCSD(embed1.mf_A).density_fit()
    mycc1.kernel()
    e_embed1 = e_mf + mycc1.e_corr
    
    # run embedding 2
    occ_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'vir', basis=mol1.basis, cutoff=1e-6)
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed2 = rPRBE(mf2, moC_occ, moC_vir, mask_occ_act, mask_vir_act)
    e_mf,mo_energy,mo_coeff,mo_occ = embed2.kernel(xc_embed=None)
    mycc2 = cc.CCSD(embed2.mf_A).density_fit()
    mycc2.kernel()
    e_embed2 = e_mf + mycc2.e_corr
    
    print((ecc2-ecc1)*627.5095, (edft2-edft1)*627.5095, 
          (e_embed2-e_embed1)*627.5095)
    #%%
    ##### standard vPRBE
    
    # run embedding 1
    occ_calc = re.RegionalActiveSpace(mf1, np.arange(0,4), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf1, np.arange(0,4), 'vir', basis=mol1.basis, cutoff=1e-6)
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed1 = rVPRBEFreezeThawCorr(mf1, moC_occ, moC_vir, mask_occ_act, mask_vir_act)
    e_mf,mo_energy,mo_coeff,mo_occ = embed1.kernel(xc_embed=None, relaxed_mp2_dm=False)
    mycc1 = cc.CCSD(embed1.mf_A).density_fit()
    mycc1.max_cycle = 100 
    mycc1.kernel()
    e_vembed1 = e_mf + mycc1.e_corr
    
    # run embedding 2
    occ_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'vir', basis=mol1.basis, cutoff=1e-6)
    mf2.diis_damp=0.7
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed2 = rVPRBEFreezeThawCorr(mf2, moC_occ, moC_vir, mask_occ_act, mask_vir_act, )
    e_mf,mo_energy,mo_coeff,mo_occ = embed2.kernel(xc_embed=None, relaxed_mp2_dm=False)
    mycc2 = cc.CCSD(embed2.mf_A).density_fit()
    mycc2.max_cycle = 1000 
    mycc2.kernel()
    e_vembed2 = e_mf + mycc2.e_corr
    
    print((ecc2-ecc1)*627.5095, (edft2-edft1)*627.5095, 
          (e_embed2-e_embed1)*627.5095, (e_vembed2-e_vembed1)*627.5095)

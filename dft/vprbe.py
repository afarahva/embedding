#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vprbe.py

Implements Huzinaga projection based embedding, including a one-density
variational PRBE variant with fixed D^B and optimized D^A. 

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from pyscf import scf, dft, df
from pyscf.scf import chkfile
from pyscf.lib import logger, diis
from functools import partial

from pyscf import __config__
TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)

from pyscf_embedding.dft.prbe import rPRBE, uPRBE

# =============================================================================
# One-density / variational PRBE
# =============================================================================
# This is the modified embedding discussed here: there is no separate
# unoptimized D^A and optimized \tilde{D}^A.  A single active density D^A is
# optimized with fixed D^B according to
#
#   E_12[D^A;D^B] = E_2[D^A + D^B] + E_1[D^A] - E_2[D^A]
#
# The corresponding active Fock-like matrix is
#
#   F_eff[D^A] = F_1[D^A] + F_2[D^A + D^B] - F_2[D^A]
#
# and Huzinaga projection against the fixed B orbital space is applied to this
# full F_eff at every SCF step.


def _transpose_last_two(a):
    """Transpose the last two dimensions for either 2D or spin-batched 3D arrays."""
    a = np.asarray(a)
    if a.ndim == 3:
        return a.transpose(0, 2, 1)
    return a.T


def _huzinaga_projected_hcore(hcore_eff, vhf_active, s1e,
                              P_occ_B=None, P_vir_B=None):
    """
    Return an h1e such that h1e + vhf_active is the Huzinaga-projected
    effective Fock matrix.

    Parameters
    ----------
    hcore_eff
        h + F_2[D_A + D_B] - F_2[D_A], possibly spin-batched.
    vhf_active
        F_1[D_A] - h, i.e. the active high-level Coulomb/XC/HF contribution.
    s1e
        AO overlap matrix.
    P_occ_B, P_vir_B
        Orbital projectors C_B C_B^T, not spin-summed density matrices.  The
        unrestricted case uses arrays of shape (2, nao, nao).
    """
    h1e = np.array(hcore_eff, copy=True)
    f_unproj = hcore_eff + vhf_active

    if P_occ_B is not None:
        P_occ_B_s1e = P_occ_B @ s1e
        m = f_unproj @ P_occ_B_s1e
        h1e = h1e - m - _transpose_last_two(m)

    if P_vir_B is not None:
        P_vir_B_s1e = P_vir_B @ s1e
        S_P_vir_B = s1e @ P_vir_B
        m = f_unproj @ P_vir_B_s1e
        h1e = h1e - m - _transpose_last_two(m)
        m2 = 2 * S_P_vir_B @ f_unproj @ _transpose_last_two(S_P_vir_B)
        h1e = h1e + m2

    return h1e


def _energy_elec0(mf, dm, h0, vhf):
    """Electronic energy helper for RHF/RKS/UHF/UKS objects."""
    e = mf.energy_elec(dm=dm, h1e=h0, vhf=vhf)
    return e[0] if isinstance(e, tuple) else e


def _one_density_prbe_build(mf_A, mf_B, dm_A, dm_B,
                            P_occ_B=None, P_vir_B=None):
    """
    Build the one-density PRBE effective core, projected core, high-level veff,
    low-level veffs, and total embedded energy for the current active density.
    """
    mol_A = mf_A.mol
    mol_B = mf_B.mol
    s1e = mf_A.get_ovlp(mol_A)
    h0 = mf_B.get_hcore(mol_B)

    dm_AB = dm_A + dm_B

    # Low-level method-2 pieces.
    v2_AB = mf_B.get_veff(mol_B, dm_AB)
    v2_A = mf_B.get_veff(mol_B, dm_A)

    # High-level method-1 active piece.
    v1_A = mf_A.get_veff(mol_A, dm_A)

    # h + V_emb[D_A, D_B].  This is now density-dependent because D_A is the
    # same active density being optimized.
    hcore_eff = h0 + v2_AB - v2_A

    # Huzinaga projection is applied to the complete effective Fock
    # F_eff = hcore_eff + v1_A.
    h1e_proj = _huzinaga_projected_hcore(
        hcore_eff, v1_A, s1e, P_occ_B=P_occ_B, P_vir_B=P_vir_B
    )

    # Variational one-density PRBE total energy:
    # E_2[D_A + D_B] + E_1[D_A] - E_2[D_A]
    e2_AB_tot = mf_B.energy_tot(dm=dm_AB, h1e=h0, vhf=v2_AB)
    e1_A_elec = _energy_elec0(mf_A, dm_A, h0, v1_A)
    e2_A_elec = _energy_elec0(mf_B, dm_A, h0, v2_A)
    e_prbe = e2_AB_tot + e1_A_elec - e2_A_elec

    return hcore_eff, h1e_proj, v1_A, v2_AB, v2_A, e_prbe


def one_density_huzinaga_kernel(mf, mf_B, dm_B, conv_tol=1e-10,
           conv_tol_grad=None, dump_chk=True, dm0=None, callback=None,
           conv_check=True, P_occ_B=None, P_vir_B=None, **kwargs):
    """
    Restricted or unrestricted SCF kernel for the one-density variational PRBE.

    This replaces the fixed hcore = h + V_emb[D_A0, D_B] used in conventional
    PRBE with a density-dependent hcore = h + V_emb[D_A, D_B].  The optimized
    active density D_A is used in both the high-level active term and the
    low-level subtractive terms.
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e, **kwargs)
    else:
        dm = np.array(dm0, copy=True)

    hcore_eff, h1e, vhf, v2_AB, v2_A, e_tot = _one_density_prbe_build(
        mf, mf_B, dm, dm_B, P_occ_B=P_occ_B, P_vir_B=P_vir_B
    )
    mf.get_hcore = lambda *args, h1e=h1e: h1e
    logger.info(mf, 'init one-density PRBE E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, _corth = mf.eig(fock, s1e)
        mf_diis.Corth = _corth[0] if _corth.ndim == 3 else _corth
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)

    mf.pre_kernel(locals())
    fock_last = None
    cput1 = logger.timer(mf, 'initialize one-density prbe scf', *cput0)
    mf.cycles = 0

    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis,
                           fock_last=fock_last)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)

        hcore_eff, h1e, vhf, v2_AB, v2_A, e_tot = _one_density_prbe_build(
            mf, mf_B, dm, dm_B, P_occ_B=P_occ_B, P_vir_B=P_vir_B
        )
        mf.get_hcore = lambda *args, h1e=h1e: h1e

        fock_last = fock
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        grad = mf.get_grad(mo_coeff, mo_occ, fock)
        norm_gorb = np.linalg.norm(grad)
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / np.sqrt(grad.size)
        norm_ddm = np.linalg.norm(dm - dm_last)

        logger.info(mf,
            'cycle= %d one-density PRBE E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
            cycle + 1, e_tot, e_tot - last_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d' % (cycle + 1), *cput1)
        if scf_conv:
            break

    mf.cycles = cycle + 1

    if scf_conv and conv_check:
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        last_e = e_tot

        hcore_eff, h1e, vhf, v2_AB, v2_A, e_tot = _one_density_prbe_build(
            mf, mf_B, dm, dm_B, P_occ_B=P_occ_B, P_vir_B=P_vir_B
        )
        mf.get_hcore = lambda *args, h1e=h1e: h1e

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        grad = mf.get_grad(mo_coeff, mo_occ, fock)
        norm_gorb = np.linalg.norm(grad)
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / np.sqrt(grad.size)
        norm_ddm = np.linalg.norm(dm - dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot - last_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True

        logger.info(mf,
            'Extra cycle one-density PRBE E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
            e_tot, e_tot - last_e, norm_gorb, norm_ddm)
        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())

    logger.timer(mf, 'one_density_prbe_scf_cycle', *cput0)
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


class rVPRBE(rPRBE):
    """
    Restricted one-density variational PRBE.

    D^B is fixed from the initial LMO partition.  A single active density D^A is
    optimized self-consistently in

        E_12[D^A;D^B] = E_2[D^A + D^B] + E_1[D^A] - E_2[D^A]

    using the Huzinaga projector built from the fixed bath orbital space.
    """

    def make_rdmB_fixed(self):
        """Construct the fixed closed-shell environment density matrix D^B."""
        return 2 * self.moC_occ_B @ self.moC_occ_B.T

    def energy_tot_one_density(self, dm_A=None):
        """Evaluate E_2[D_A + D_B] + E_1[D_A] - E_2[D_A]."""
        if self.mf_A is None:
            return None
        if dm_A is None:
            dm_A = self.mf_A.make_rdm1()
        dm_B = self.make_rdmB_fixed()
        _, _, v1_A, v2_AB, v2_A, e = _one_density_prbe_build(
            self.mf_A, self.mf_B, dm_A, dm_B,
            P_occ_B=self.moC_occ_B @ self.moC_occ_B.T,
            P_vir_B=self.moC_vir_B @ self.moC_vir_B.T,
        )
        return e

    def kernel(self, xc_embed=None, ecp_embed=None, pseudo_embed=None,
               filtermo=True, **kwargs):
        """Run one-density variational Huzinaga PRBE."""
        mf_B = self.mf_B
        ovlp = mf_B.get_ovlp()
        mol = mf_B.mol.copy()
        mol.nelectron = int(2 * self.moC_occ_A.shape[1])

        dm_A0 = self.make_rdmA_init()
        dm_B = self.make_rdmB_fixed()
        self.dm_B = dm_B

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

        mf_A.converged, mf_A.e_tot, mf_A.mo_energy, \
            mf_A.mo_coeff, mf_A.mo_occ = one_density_huzinaga_kernel(
                mf=mf_A, mf_B=mf_B, dm_B=dm_B, dm0=dm_A0,
                P_occ_B=P_occ_B, P_vir_B=P_vir_B
            )
        self.e_tot = mf_A.e_tot

        # find indices of bath/frozen MOs
        if filtermo:
            bath_ovlp = np.linalg.norm(mf_A.mo_coeff.T @ ovlp @ np.hstack([self.moC_occ_B, self.moC_vir_B]), axis=1) 
            N = len(self.moE_occ_A)+len(self.moE_vir_A)
            mask_act = np.isin(np.arange(bath_ovlp.size), np.argpartition(bath_ovlp, min(N, bath_ovlp.size - 1))[:N])
                
            mf_A.mo_energy = mf_A.mo_energy[mask_act]
            mf_A.mo_coeff = mf_A.mo_coeff[:,mask_act]
            mf_A.mo_occ = mf_A.mo_occ[mask_act]            

        self.e_tot -= self.calc_disp_embed()
        return self.e_tot, mf_A.mo_energy, mf_A.mo_coeff, mf_A.mo_occ


class uVPRBE(uPRBE):
    """Unrestricted one-density variational PRBE."""

    def make_rdmB_fixed(self):
        """Construct the fixed spin-resolved environment density matrix D^B."""
        dm_B_alpha = self.moC_occ_B[0] @ self.moC_occ_B[0].T
        dm_B_beta = self.moC_occ_B[1] @ self.moC_occ_B[1].T
        return np.array([dm_B_alpha, dm_B_beta])

    def energy_tot_one_density(self, dm_A=None):
        if self.mf_A is None:
            return None
        if dm_A is None:
            dm_A = self.mf_A.make_rdm1()
        dm_B = self.make_rdmB_fixed()
        P_occ_B = np.array([
            self.moC_occ_B[0] @ self.moC_occ_B[0].T,
            self.moC_occ_B[1] @ self.moC_occ_B[1].T,
        ])
        P_vir_B = np.array([
            self.moC_vir_B[0] @ self.moC_vir_B[0].T,
            self.moC_vir_B[1] @ self.moC_vir_B[1].T,
        ])
        _, _, _, _, _, e = _one_density_prbe_build(
            self.mf_A, self.mf_B, dm_A, dm_B,
            P_occ_B=P_occ_B, P_vir_B=P_vir_B,
        )
        return e

    def kernel(self, xc_embed=None, ecp_embed=None, pseudo_embed=None,
               filtermo=True, **kwargs):
        """Run open-shell one-density variational Huzinaga PRBE."""
        mf_B = self.mf_B
        ovlp = mf_B.get_ovlp()
        mol = mf_B.mol.copy()

        n_alpha = self.moC_occ_A[0].shape[1]
        n_beta = self.moC_occ_A[1].shape[1]
        mol.nelectron = int(n_alpha + n_beta)
        mol.spin = int(n_alpha - n_beta)

        dm_A0 = self.make_rdmA_init()
        dm_B = self.make_rdmB_fixed()
        self.dm_B = dm_B

        if xc_embed is None:
            self.mf_A = scf.UHF(mol, **kwargs)
        else:
            self.mf_A = dft.UKS(mol)
            self.mf_A.xc = xc_embed

        if hasattr(mf_B, 'with_df'):
            self.mf_A = df.density_fit(self.mf_A)
            self.mf_A.with_df.auxbasis = mf_B.with_df.auxbasis

        mf_A = self.mf_A
        mf_A.diis_space = mf_B.diis_space
        mf_A.diis_damp = mf_B.diis_damp
        mf_A.max_cycle = mf_B.max_cycle
        mf_A.verbose = mf_B.verbose

        P_occ_B = np.array([
            self.moC_occ_B[0] @ self.moC_occ_B[0].T,
            self.moC_occ_B[1] @ self.moC_occ_B[1].T,
        ])
        P_vir_B = np.array([
            self.moC_vir_B[0] @ self.moC_vir_B[0].T,
            self.moC_vir_B[1] @ self.moC_vir_B[1].T,
        ])

        mf_A.converged, mf_A.e_tot, mf_A.mo_energy, \
            mf_A.mo_coeff, mf_A.mo_occ = one_density_huzinaga_kernel(
                mf=mf_A, mf_B=mf_B, dm_B=dm_B, dm0=dm_A0,
                P_occ_B=P_occ_B, P_vir_B=P_vir_B
            )
        self.e_tot = mf_A.e_tot

        # find indices of bath/frozen MOs
        if filtermo:
            bath_ovlp = np.linalg.norm(mf_A.mo_coeff[0].T @ ovlp @ np.hstack([self.moC_occ_B[0], self.moC_vir_B[0]]), axis=1) 
            N = len(self.moE_occ_A[0])+len(self.moE_vir_A[0])
            mask_act_a = np.isin(np.arange(bath_ovlp.size), np.argpartition(bath_ovlp, min(N, bath_ovlp.size - 1))[:N])
            
            bath_ovlp = np.linalg.norm(mf_A.mo_coeff[1].T @ ovlp @ np.hstack([self.moC_occ_B[1], self.moC_vir_B[1]]), axis=1)
            N = len(self.moE_occ_A[1])+len(self.moE_vir_A[1])
            mask_act_b = np.isin(np.arange(bath_ovlp.size), np.argpartition(bath_ovlp, min(N, bath_ovlp.size - 1))[:N])
            
            # pad arrays in the case that the number of active alpha/beta orbitals is different
            N_act_a, N_act_b = np.sum(mask_act_a),np.sum(mask_act_b)
            N_max = max(N_act_a,N_act_b)
            mo_energy = np.zeros((2,N_max),dtype=np.float64)+1e30
            mo_coeff = np.zeros((2,mf_A.mo_coeff.shape[1],N_max),dtype=np.float64)
            mo_occ = np.zeros((2,N_max),dtype=np.float64)
            #
            mo_energy[0,0:N_act_a] = mf_A.mo_energy[0][mask_act_a]
            mo_energy[1,0:N_act_b] = mf_A.mo_energy[1][mask_act_b]
            mo_coeff[0,:,0:N_act_a] = mf_A.mo_coeff[0][:, mask_act_a]
            mo_coeff[1,:,0:N_act_b] = mf_A.mo_coeff[1][:, mask_act_b]
            mo_occ[0,0:N_act_a] = mf_A.mo_occ[0][mask_act_a]
            mo_occ[1,0:N_act_b] = mf_A.mo_occ[1][mask_act_b]
            #
            mf_A.mo_energy=mo_energy
            mf_A.mo_coeff=mo_coeff
            mf_A.mo_occ=mo_occ
            del mo_energy,mo_coeff,mo_occ
        
        self.e_tot -= self.calc_disp_embed()
        return self.e_tot, mf_A.mo_energy, mf_A.mo_coeff, mf_A.mo_occ


#%%
if __name__ == '__main__':
    from pyscf_embedding.local import regional as re
    from pyscf_embedding.utils import chemcore
    from pyscf import gto, scf, cc, grad

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

    
    #%%
    ##### Calculation One, Test rPRBE
    basis='ccpvdz'
    xc = 'pbe'
    #%%
    # run CC
    mol1 = gto.Mole(atom=coords1,basis=basis,spin=0,verbose=4)
    mf1 = mol1.RHF(max_cycles=100).density_fit().run()
    mycc1 = cc.CCSD(mf1,frozen=chemcore(mol1)).density_fit()
    mycc1 = mycc1.run()
    ecc1 = mycc1.e_tot
    
    mol2 = gto.Mole(atom=coords2,basis=basis,spin=0,verbose=4,charge=-1)
    mf2 = mol2.RHF(max_cycles=100).density_fit().run()
    mycc2 = cc.CCSD(mf2,frozen=chemcore(mol2)).density_fit()
    mycc2 = mycc2.run()
    ecc2 = mycc2.e_tot
    
    del mycc1, mycc2
    #%%
    # run DFT
    mol1 = gto.Mole(atom=coords1,basis=basis,spin=0,verbose=4)
    mf1 = mol1.RKS(xc=xc,max_cycles=100).density_fit().run()
    edft1 = mf1.e_tot
    
    mol2 = gto.Mole(atom=coords2,basis=basis,spin=0,verbose=4,charge=-1)
    mf2 = mol2.RKS(xc=xc,max_cycles=100).density_fit().run()
    edft2 = mf2.e_tot
    
    #%%
    ##### standard PRBE
    
    # run embedding 1
    occ_calc = re.RegionalActiveSpace(mf1, np.arange(0,4), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf1, np.arange(0,4), 'vir', basis=mol1.basis, cutoff=1e-6)
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed1 = rPRBE(mf1, occ_calc, vir_calc)
    e_mf,mo_energy,mo_coeff,mo_occ = embed1.kernel(xc_embed=None)
    mycc1 = cc.CCSD(embed1.mf_A)
    mycc1.kernel()
    e_embed1 = e_mf + mycc1.e_corr
    
    # run embedding 2
    occ_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'vir', basis=mol1.basis, cutoff=1e-6)
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed2 = rPRBE(mf2, occ_calc, vir_calc)
    e_mf,mo_energy,mo_coeff,mo_occ = embed2.kernel(xc_embed=None)
    mycc2 = cc.CCSD(embed2.mf_A)
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

    embed1 = rVPRBE(mf1, occ_calc, vir_calc)
    e_mf,mo_energy,mo_coeff,mo_occ = embed1.kernel(xc_embed=None)
    mycc1 = cc.CCSD(embed1.mf_A)
    mycc1.kernel()
    e_vembed1 = e_mf + mycc1.e_corr
    
    # run embedding 2
    occ_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'vir', basis=mol1.basis, cutoff=1e-6)
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed2 = rVPRBE(mf2, occ_calc, vir_calc)
    e_mf,mo_energy,mo_coeff,mo_occ = embed2.kernel(xc_embed=None)
    mycc2 = cc.CCSD(embed2.mf_A)
    mycc2.kernel()
    e_vembed2 = e_mf + mycc2.e_corr
    
    print((ecc2-ecc1)*627.5095, (edft2-edft1)*627.5095, 
          (e_embed2-e_embed1)*627.5095, (e_vembed2-e_vembed1)*627.5095)

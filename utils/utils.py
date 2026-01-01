#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib.py

Library of functions and classes for PySCF embedding code
Generalized for Restricted and Unrestricted references.

author: Ardavan Farahvash, github.com/afarahva
"""
import sys

import numpy as np
from scipy.linalg import eigh

from pyscf.lib import logger
from pyscf import lo
from pyscf.gto import intor_cross as mol_intor_cross
from pyscf.pbc.gto import intor_cross as pbc_intor_cross

OCCUPATION_CUTOFF=1e-10

def chemcore(mol, atm_indx=None, spinorb=False):
    """
    Get number of core electrons 
    """
    from pyscf.data.elements import charge, chemcore_atm
    core = 0
    
    if atm_indx is None:
        atm_indx = range(mol.natm)
    
    for a in atm_indx:
        atm_nelec = mol.atom_charge(a)
        atm_z = charge(mol.atom_symbol(a))
        ne_ecp = atm_z - atm_nelec
        ncore_ecp = ne_ecp // 2
        atm_ncore = chemcore_atm[atm_z]
        if ncore_ecp > atm_ncore:
            core += 0
        else:
            core += atm_ncore - ncore_ecp

    if spinorb:
        core *= 2
    
    return core

def matrix_projection(M, P, S=None):
    """
    Generalized Matrix Projection Function
    Let P be a projection matrix of (not necessarily orthnormal) columns
    w.r.t to the modified inner product <P_i,P_j> = P_i S P_j. 
    
    Computes matrix M'=P^+ M P, a projection of M into the columns of P.
    P+ is the Moore-Pensore pseudoinverse.
    """
    if S is None:
        S = np.eye(P.shape[0])
        
    test = P.T.conj() @ S @ P
    if np.all(np.isclose(test,np.eye(P.shape[1]))):
        P_T = P.T.conj()
    else:
        P_T = np.linalg.inv(P.T.conj() @ S @ P ) @ P.T.conj()
    
    # project fock matrix using P
    M_P = P_T @ M @ P
        
    return M_P

def matrix_projection_eigh(M, P, S=None):
    """
    Projects M into a basis defined by the columns of P: M' = P^+ M P
    and solves generalized eigenvalue problem for M'
    """
    
    M_P = matrix_projection(M, P, S=None)
    E_P, C_P = eigh(M_P,b=S)
    
    return E_P, C_P

class FC_AO_Ints:
    """
    Class for generating orbital overlap integrals between the AOs of a 
    composite system and localized orbitals of a fragment. 
    """
    
    def __init__(self, mol_or_cell, frag_inds, frag_inds_type='atom', 
                 basis_frag=None, orth=False):
        """
        Parameters
        ----------
        mol_or_cell : PySCF mol/cell object for composite system
        
        frag_inds : String
            Fragment indices
            
        frag_inds_type : Array
            Can be either 'atom' or orbital', Default : 'atom'
        
        basis_frag : String
            Basis set for fragment, Default is to use same as in mol
            
        orth : Bool
            Whether to orthogonalization fragment orbitals. Default is False.
        """
        
        self.mol = mol_or_cell
                    
        # define fragment basis set
        if basis_frag is None:
            self.basis = self.mol.basis
        elif basis_frag == 'iao':
            self.basis='minao'
        else:
            self.basis = basis_frag
        
        # determine indices of orbitals belonging to fragment
        self.mol2 = self.mol.copy()
        self.mol2.basis = self.basis
        self.mol2.build()

        if frag_inds_type.lower() == "atom":
            self.frag_atm_inds = frag_inds
            self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                                self.mol2.aoslice_by_atom()[frag_inds]]).astype(int)
        
        elif frag_inds_type.lower() == 'orbital':
            self.frag_atm_inds = None
            self.frag_ao_inds = frag_inds
        
        else:
            raise ValueError("frag_inds_type must be either 'atom' or 'orbital'")
            
        # orthogonalization method for fragment AOs
        self.orth = orth
        
        # < frag AO | frag AO > overlap
        self.s_ff = None
        
        # < frag AO | composite AO > overlap
        self.s_fc = None
                
    def calc_ao_ovlp(self, moC_occ=None):
        """
        Calculate overlap matrices between fragment AOs and composite AOs
                
        Returns:
        ----------
        s_ff : fragment-fragment overlap integral
        s_fc : fragment-composite overlap integral
        """
            
        # special case, intrinsic atomic orbitals
        if self.basis=='iao':
            
            # check mol or cell
            if getattr(self.mol, 'pbc_intor', None) is None: # mol
                s_all = self.mol.intor_symmetric('int1e_ovlp')
            else:
                s_all = self.mol.pbc_intor('int1e_ovlp', hermi=1)
                
            c_iao = lo.iao.iao(self.mol, moC_occ)
            
            # Handle UHF/UKS IAO which returns (alpha, beta) usually, 
            # though standard PySCF lo.iao expects restricted or takes first.
            # If c_iao is tuple, we assume same IAO space for both or handle complex logic.
            # NOTE: Standard PySCF IAO usually generates one set of IAOs. 
            # If moC_occ is tuple, lo.iao might return tuple or array. 
            if isinstance(c_iao, (list, tuple)):
                # If IAO returns a list, usually implies distinct IAOs for spins
                # Simplification: Use alpha IAOs or error out for simplicity 
                # unless explicitly handling unrestricted IAO projection.
                c_iao = c_iao[0]

            s_ff = (c_iao.T @ s_all @ c_iao)[np.ix_(self.frag_ao_inds,self.frag_ao_inds)]
            s_fc = (c_iao.T @ s_all)[self.frag_ao_inds]
        
        # all other cases
        else:
            # check mol or cell
            if getattr(self.mol2, 'pbc_intor', None) is None: # mol
                s_all = self.mol2.intor_symmetric('int1e_ovlp')
                intor_cross = mol_intor_cross
            else:
                s_all = self.mol2.pbc_intor('int1e_ovlp', hermi=1)
                intor_cross = pbc_intor_cross
                
            # < frag AO | frag AO > overlap
            s_ff = s_all[np.ix_(self.frag_ao_inds,self.frag_ao_inds)]
            
            # < frag AO | composite AO > overlap
            s_fc = intor_cross('int1e_ovlp', self.mol2, self.mol)[self.frag_ao_inds]
                
            # orthogonalize frag AOs and return orthogonalized overlap integrals
            if self.orth:
                s_ff, s_fc = self.lowdin(s_ff, s_fc)

        return s_ff, s_fc
    
    def calc_mo_ovlp(self, c_mo):
        """
        Calculate overlap matrix between MOs and fragment orbitals. 
        Supports both Restricted (Array) and Unrestricted (Tuple/List) inputs.
        """
        
        # Generalize for Unrestricted: Recursive call if input is list/tuple
        if isinstance(c_mo, (list, tuple)):
            res = [self.calc_mo_ovlp(c) for c in c_mo]
            # unzip into [(s_a, s_b), (C_a, C_b)]
            return list(zip(*res))

        if self.s_ff is None or self.s_fc is None:
            self.calc_ao_ovlp()
            
        # < FO | composite mo > overlap
        s_f_mo = self.s_fc @ c_mo
        
        # composite MO coefficients in projected into fragment orbitals
        C_f_mo = np.linalg.inv(self.s_ff) @ s_f_mo
        
        return s_f_mo , C_f_mo        
    
    def frag_prob(self, c_mo):
        """
        Calculate total probability of each MO on fragment orbitals. 
        Supports both Restricted (Array) and Unrestricted (Tuple/List) inputs.
        """
        # Generalize for Unrestricted: Recursive call if input is list/tuple
        if isinstance(c_mo, (list, tuple)):
            res = [self.frag_prob(c) for c in c_mo]
            return tuple(res)

        _, C_f_mo = self.calc_mo_ovlp(c_mo)
        pop = np.sum( np.abs(C_f_mo)**2, axis=1)
        return pop
    
    def lowdin(self, s_ff, s_fc):
        """
        Orthogonalize fragment orbitals.
        """
        C_fo = lo.lowdin(s_ff)
        s_fc = C_fo.T @ s_fc
        s_ff = C_fo.T @ s_ff @ C_fo

        return s_ff, s_fc     
    
class ActiveSpace:
    """
    Base class for calculating MO energies and coefficients of an active space
    of orbitals which are a rotation of the canonical MOs. 
    
    Supports both Restricted (RHF/RKS) and Unrestricted (UHF/UKS) references.
    """
    
    # Unitary Projection Operators for active/frozen MOs (in canonoical MO basis)
    # If UHF, these should be tuples (P_a, P_b)
    P_act=None
    P_frz=None
    Norb_act=None
    
    def calc_projection(self):
        pass
    
    def __init__(self, mf, mo_coeff="occupied", frozen_core=False):
        """
        Parameters
        ----------
        mf : PySCF Mean-Field Object (Restricted or Unrestricted)
        mo_coeff : string or Numpy Array or Tuple of Arrays
        frozen_core : Bool
        """
        
        # copy mean field object
        self.mf = mf
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.frozen_core = frozen_core
        
        # create logger
        self.log = logger.new_logger(self)
        
        # Check if Unrestricted (mo_coeff is list/tuple or 3D array)
        mf_mo_coeff = self.mf.mo_coeff
        self.is_uhf = (isinstance(mf_mo_coeff, np.ndarray) and mf_mo_coeff.ndim == 3)

        if type(mo_coeff) == list:
             # If user passes list but MF is restricted, assume k-point (not supported)
             self.log.error(" Multiple k-point embedding is not supported. ")
             sys.exit()

        # Handle user-provided coefficients
        if not isinstance(mo_coeff, str):
            self.moC = mo_coeff
            if self.is_uhf:
                # Calculate energies for alpha and beta separately
                self.moE = (
                    np.diag(self.moC[0].T @ self.mf.get_fock()[0] @ self.moC[0]),
                    np.diag(self.moC[1].T @ self.mf.get_fock()[1] @ self.moC[1])
                )
            else:
                self.moE = np.diag(self.moC.T @ self.mf.get_fock() @ self.moC)
                
        # Handle string selector ('occ', 'vir')
        else:
            if mo_coeff.lower() in ["occupied","occ","o"]:
                if self.is_uhf:
                    mask = (self.mf.mo_occ[0] >= OCCUPATION_CUTOFF, self.mf.mo_occ[1] >= OCCUPATION_CUTOFF)
                else:
                    mask = self.mf.mo_occ >= OCCUPATION_CUTOFF
                
            elif mo_coeff.lower() in ["virtual","vir","v"]:
                if self.is_uhf:
                    mask = (self.mf.mo_occ[0] < OCCUPATION_CUTOFF, self.mf.mo_occ[1] < OCCUPATION_CUTOFF)
                else:
                    mask = self.mf.mo_occ < OCCUPATION_CUTOFF
                
            else:
                self.log.error("Invalid mo_coeff selection string.")
                sys.exit()
                
            if self.is_uhf:
                self.moE = (self.mf.mo_energy[0][mask[0]], self.mf.mo_energy[1][mask[1]])
                self.moC = (self.mf.mo_coeff[0][:,mask[0]], self.mf.mo_coeff[1][:,mask[1]])
            else:
                self.moE = self.mf.mo_energy[mask]
                self.moC = self.mf.mo_coeff[:,mask]
        
        # frozen core approximation
        if frozen_core:
            if isinstance(mo_coeff, str) and mo_coeff.lower() not in ['o','occ','occupied']:
                self.log.error('''frozen core only supported for occ orbital transformations''')
                sys.exit()
            
            ncore = chemcore(self.mf.mol)
            indx_core = np.arange(0, ncore, dtype=np.int64)

            if self.is_uhf:
                # Freeze lowest N orbitals for both alpha and beta
                self.moE_core = (self.moE[0][indx_core], self.moE[1][indx_core])
                self.moC_core = (self.moC[0][:,indx_core], self.moC[1][:,indx_core])
                self.moE = (np.delete(self.moE[0], indx_core), np.delete(self.moE[1], indx_core))
                self.moC = (np.delete(self.moC[0], indx_core, axis=1), np.delete(self.moC[1], indx_core, axis=1))
            else:
                self.moE_core = self.moE[indx_core]
                self.moC_core = self.moC[:,indx_core]
                self.moE = np.delete(self.moE, indx_core)
                self.moC = np.delete(self.moC, indx_core, axis=1)
        else:
            # Empty placeholders
            if self.is_uhf:
                self.moE_core = (np.array([]), np.array([]))
                self.moC_core = (np.zeros((self.moC[0].shape[0],0)), np.zeros((self.moC[1].shape[0],0)))
            else:
                self.moE_core = np.array([])
                self.moC_core = np.zeros((self.moC.shape[0],0))
        
    def pseudocanonical(self, moE, moC, P):
        """
        Project and Diagonalize the Fock Matrix (Single Spin Channel)
        """
        # fock matrix in canonical MO basis
        fock = np.diag(moE)

        # diagonalize Fock operator in this basis
        E_P, C_P = matrix_projection_eigh(fock, P)
        
        # express eigenvectors of F in terms of AOs
        C_P = moC @ P @ C_P
            
        return E_P, C_P    
        
    def calc_mo(self, debug=False):
        """
        Calculate active and frozen pseudocanonical MOs.
        Handles both Restricted (returns arrays) and Unrestricted (returns tuples).
        """
        
        # project fock matrix (Should set self.P_act)
        # If UHF, child class must set P_act to (P_act_a, P_act_b)
        self.calc_projection()
        
        if self.is_uhf:
            # Unrestricted Logic: Process Alpha and Beta separately
            # Assumes P_act/P_frz are tuples (P_a, P_b)
            
            res_moE, res_moC, res_mask = [], [], []
            
            for s in [0, 1]: # Alpha=0, Beta=1
                # Access per-spin attributes
                P_act_s = self.P_act[s] if self.P_act is not None else None
                P_frz_s = self.P_frz[s] if self.P_frz is not None else None
                moE_s = self.moE[s]
                moC_s = self.moC[s]
                moE_core_s = self.moE_core[s]
                moC_core_s = self.moC_core[s]

                if P_act_s is not None:
                    moE_act, moC_act = self.pseudocanonical(moE_s, moC_s, P_act_s)
                    moE_frz, moC_frz = self.pseudocanonical(moE_s, moC_s, P_frz_s)
                    
                    moC_final = np.hstack([moC_act, moC_core_s, moC_frz])
                    moE_final = np.hstack([moE_act, moE_core_s, moE_frz])
                    
                    # Norb_act might be tuple (N_a, N_b) or scalar
                    n_act = self.Norb_act[s] if isinstance(self.Norb_act, (tuple,list)) else self.Norb_act
                    mask_act = np.arange(len(moE_final)) < n_act
                else:
                    moC_final = moC_s
                    moE_final = moE_s
                    mask_act = np.array([True]*len(moE_s))

                # reorder
                order = np.argsort(moE_final)
                res_moE.append(moE_final[order])
                res_moC.append(moC_final[:, order])
                res_mask.append(mask_act[order])
            
            return tuple(res_moE), tuple(res_moC), tuple(res_mask)

        else:
            # Restricted Logic (Original)
            if self.P_act is not None:
                moE_act, moC_act = self.pseudocanonical(self.moE, self.moC, self.P_act)
                moE_frz, moC_frz = self.pseudocanonical(self.moE, self.moC, self.P_frz)
                moC = np.hstack([moC_act, self.moC_core, moC_frz])
                moE = np.hstack([moE_act, self.moE_core, moE_frz])
                mask_act = np.arange(len(moE)) < self.Norb_act
            else:
                moC = self.moC
                moE = self.moE
                mask_act = np.array([True]*len(moE))
            
            order = np.argsort(moE)
            return moE[order], moC[:,order], mask_act[order]


class HFEmbedding:
    """
    Base class for Wavefunction-in-HF Embedding.
    """
    
    def __init__(self, OActiveSpace, VActiveSpace):
        self.occ_calc = OActiveSpace
        self.vir_calc = VActiveSpace
        
        assert self.occ_calc is not None or self.vir_calc is not None
        
        # Determine if UHF based on one of the calculators
        calc = self.occ_calc if self.occ_calc else self.vir_calc
        self.is_uhf = calc.is_uhf
        
    def _get_ref_data(self, mf, select_str):
        """Helper to get reference data for missing calculator"""
        if self.is_uhf:
            if select_str == "occupied":
                mask = (mf.mo_occ[0] >= OCCUPATION_CUTOFF, mf.mo_occ[1] >= OCCUPATION_CUTOFF)
            else:
                mask = (mf.mo_occ[0] < OCCUPATION_CUTOFF, mf.mo_occ[1] < OCCUPATION_CUTOFF)
            
            moE = (mf.mo_energy[0][mask[0]], mf.mo_energy[1][mask[1]])
            moC = (mf.mo_coeff[0][:,mask[0]], mf.mo_coeff[1][:,mask[1]])
            mask_act = (np.array([True]*len(moE[0])), np.array([True]*len(moE[1])))
            return moE, moC, mask_act
        else:
            if select_str == "occupied":
                mask = mf.mo_occ >= OCCUPATION_CUTOFF
            else:
                mask = mf.mo_occ < OCCUPATION_CUTOFF
            moE = mf.mo_energy[mask]
            moC = mf.mo_coeff[:,mask]
            mask_act = np.array([True]*np.sum(mask))
            return moE, moC, mask_act

    def calc_mo(self):
        
        # Get Occupied
        if self.occ_calc is not None:
            moE_occ, moC_occ, mask_occ_act = self.occ_calc.calc_mo()
        else:
            moE_occ, moC_occ, mask_occ_act = self._get_ref_data(self.vir_calc.mf, "occupied")
            
        # Get Virtual
        if self.vir_calc is not None:
            moE_vir, moC_vir, mask_vir_act = self.vir_calc.calc_mo()
        else:
            moE_vir, moC_vir, mask_vir_act = self._get_ref_data(self.occ_calc.mf, "virtual")
            
        if self.is_uhf:
            # Stack Alpha (0) and Beta (1) independently
            moE_embed = (np.hstack([moE_occ[0], moE_vir[0]]), np.hstack([moE_occ[1], moE_vir[1]]))
            moC_embed = (np.hstack([moC_occ[0], moC_vir[0]]), np.hstack([moC_occ[1], moC_vir[1]]))
            
            # Combine masks (tuple of masks)
            self.mask_act = (np.hstack([mask_occ_act[0], mask_vir_act[0]]), 
                             np.hstack([mask_occ_act[1], mask_vir_act[1]]))
            self.mask_frz = (~self.mask_act[0], ~self.mask_act[1])
            
            indx_frz_a = np.argwhere(~self.mask_act[0])[:,0]
            indx_frz_b = np.argwhere(~self.mask_act[1])[:,0]
            indx_frz = (indx_frz_a, indx_frz_b)
            
        else:
            moE_embed = np.hstack([moE_occ, moE_vir])
            moC_embed = np.hstack([moC_occ, moC_vir])
            
            self.mask_act = np.hstack([mask_occ_act, mask_vir_act])
            self.mask_frz = ~self.mask_act
            
            indx_frz = np.argwhere(~self.mask_act)[:,0]
            
        return moE_embed, moC_embed, indx_frz
    
if __name__ == '__main__':
    pass
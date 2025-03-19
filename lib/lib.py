#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib.py

Library of functions and classes for PySCF embedding code

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from scipy.linalg import eigh
from pyscf import lo

from pyscf.gto import intor_cross as mol_intor_cross
from pyscf.pbc.gto import intor_cross as pbc_intor_cross
        
def matrix_projection(M, P, S=None):
    """
    Generalized Matrix Projection Function
    
    Let P be a projection matrix of (not necessarily orthnormal) columns
    w.r.t to the modified inner product <P_i,P_j> = P_i S P_j. 
    
    Computes matrix M'=P^+ M P, a projection of M into the columns of P.
    P+ is the Moore-Pensore pseudoinverse.
    
    Parameters
    ----------
    M : Numpy Array.
        Matrix to be projected.
    P : Numpy Array.
        Projection Operator (columns define basis vectors).
    S : Numpy Array.
        Overlap/Inner product matrix. Gectors vectors defined as orthnormal
        w.r.t v^T S v.
    Note: for this projection to be well-defined S must not be singular and
    the columns of P must be linearly independent.
    
    Returns
    ----------
    M_p : Numpy Array.
        Projected Matrix
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
    
    Parameters
    ----------
    M : Numpy Array.
        Matrix to be projected.
    P : Numpy Array.
        Projection Operator (columns define basis vectors).
    S : Numpy Array.
        Overlap/Inner product matrix. Gectors vectors defined as orthnormal
        w.r.t v^T S v.
    Note: for this projection to be well-defined S must not be singular and
    the columns of P must be linearly independent.
    
    Returns
    ----------
    E_P, C_P : Projected eigenvalues/eigenvector coefficients
    """
    
    M_P = matrix_projection(M, P, S=None)
    E_P, C_P = eigh(M_P,b=S)
    
    return E_P, C_P

class FC_AO_Ints:
    """
    Class for generating orbital overlap integrals between the AOs of a 
    composite system and localized orbitals of a fragment. 
    
    Automatically detects whether the given object is mol or cell and adjusts 
    accordingly.
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
            s_ff = (c_iao.T @ s_all @ c_iao)[np.ix_(self.frag_ao_inds,self.frag_ao_inds)]
            s_fc = (c_iao.T @ s_all)[self.frag_ao_inds]
        
        # all other cases
        else:
            # check mol or cell
            if getattr(self.mol2, 'pbc_intor', None) is None: # mol
                s_all = self.mol.intor_symmetric('int1e_ovlp')
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
        
        Note: MO coefficients should be given in the same basis set as the mol 
        attribute and should be expressed in therms of the appropriate 
        orthgonalized orbitals if the 'orth' attribute was provided. 
        
        Parameters
        ----------
        c_mo : Numpy Array.
            Matrix of coefficients of molecular orbitals (Nao,Nmo). 
            Coefficients must be in the same basis in mol

        Returns
        -------
        s_f_mo : Numpy Array
            < FO | composite mo > overlap integrals
            
        C_f_mo : Numpy Array
            Molecular orbital coefficients projected into fragment orbitals
        """
        
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
                
        Note: MO coefficients should be given in the same basis set as the mol 
        attribute and should be expressed in therms of the appropriate 
        orthgonalized orbitals if the 'orth' attribute was provided. 
        
        Parameters
        ----------
        c_mo : Numpy Array.
            Matrix of coefficients of molecular orbitals (Nao,Nmo). 
            Coefficients must be in the same basis in mol

        Returns
        -------
        pop : Numpy Array. (Nmo)
            Total population of MOs in fragment
        """
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
    
class rUnitaryActiveSpace:
    """
    Base class for calculating MO energies and coefficients of an active space
    of orbitals which are a rotation of the canonical MOs. 
    
    Assumes reference orbitals are from a restricted MF calculation.
    
    Attributes
    ----------
    P_act : Numpy Array
        Projection Matrix on Active Orbitals
    P_frz : Numpy Array
        Projection Matrix on Frozen Orbitals
    Norb_act : Int
        Number of active orbitals

    Methods
    -------
    calc_projection()
        empty method, replace with a function that calculates the projection
        matrix
    pseudocanonical(moE, moC, P)
        Block diagonalizes the Fock matrix by projecting using matrix P
    calc_mo()
        Calculates active and frozen MO energies and coefficiens.
    """
    
    # Unitary Projection Operators for active/frozen MOs (in canonoical MO basis)
    P_act=None
    P_frz=None
    Norb_act=None
    
    # empty method, to be replaced by child class
    def calc_projection(self):
        pass
    
    def __init__(self, mf, mo_occ_type, frozen_core=False):
        """

        Parameters
        ----------
        mf : PySCF Resctricted Mean-Field Object
        mo_occ_type : String,
            Which MO space to rotate (must be either 'occupied' or 'virtual').

        """
        self.mf=mf
        self.mo_space = mo_occ_type
        
        mo_coeff, mo_energy, mo_occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
        
        # check mol or pbc
        if type(mo_coeff) == list:
            if len(mo_coeff) > 1 or len(mo_energy) > 1 or len(mo_occ) > 1:
                raise NotImplementedError(" Multiple k-point embedding is not supported. ")
            else:
                mo_coeff, mo_energy, mo_occ = mo_coeff[0], mo_energy[0], mo_occ[0]
                
        if type(mo_coeff) == np.ndarray and len(mo_coeff.shape) > 2:
                raise NotImplementedError(" Unrestricted references are not supported. ")
            
        
        mask_occ = mo_occ > 1e-10
        mask_vir = (mask_occ==False)
         
        moE_occ = mo_energy[mask_occ]
        moE_vir = mo_energy[mask_vir]
        moC_occ = mo_coeff[:,mask_occ]
        moC_vir = mo_coeff[:,mask_vir]
        
        if self.mo_space.lower() in ['o','occ','occupied']:
            self.moE = moE_occ 
            self.moC = moC_occ
        
        elif self.mo_space.lower() in ['v','vir','virtual']:
            self.moE = moE_vir
            self.moC = moC_vir
        
        else:
            raise ValueError(" 'mo_occ_type' must be either 'occupied' or 'virtual' ")
            
        self.Nmo, self.Nocc, self.Nvir = len(mo_energy), len(moE_occ), len(moE_vir)
        pass
        
    def pseudocanonical(self, moE, moC, P):
        """
        Project and Diagonalize the Fock Matrix

        Parameters
        ----------
        moE : Canonical MO energies
        moC : Canonical MO Coefficiens
        P : Projection Matrix

        Returns
        -------
        E_P : Pseudocanonical MO energies
        C_P : Pseudocanonical MO coefficients
        """
        
        # fock matrix in canonical MO basis
        fock = np.diag(moE)

        # diagonalize Fock operator in this basis
        E_P, C_P = matrix_projection_eigh(fock, P)
        
        # express eigenvectors of F in terms of AOs
        C_P = moC @ P @ C_P
            
        return E_P, C_P    
        
    def calc_mo(self):
        """
        Calculate active and frozen pseudocanonical MOs
        
        Returns
        -------
        moE : Numpy Array
            Active + Frozen MO Energies.
        moC : Numpy Array
            Active + Frozen MO Coefficients.
        mask_act : Numpy Array
            mask for which orbitals are active.
        """
        
        # project fock matrix
        self.calc_projection()
        
        # calculate projected MOs
        if self.P_act is not None:
            moE_act,moC_act = self.pseudocanonical(self.moE,self.moC,self.P_act)
            moE_frz,moC_frz = self.pseudocanonical(self.moE,self.moC,self.P_frz)            
            moC = np.hstack([moC_act,moC_frz])
            moE = np.hstack([moE_act,moE_frz])
            mask_act = np.arange(len(moE)) < self.Norb_act
        else:
            mask_act = np.array([True]*len(moE))
        
        # reorder by energy (makes analysis easier)
        order = np.argsort(moE)
        moE = moE[order]
        moC = moC[:,order]
        mask_act = mask_act[order]
        
        return moE, moC, mask_act
    
class rWVFEmbedding:
    """
    Base class for Wavefunction-in-Wavefunction Embedding methods with a 
    restricted reference. 
    
    Given an active-space of occupied and virtual orbitals constructs embedded 
    mo energies and coefficients.
    """
    
    def __init__(self, OActiveSpace, VActiveSpace):
        """
        
        Parameters
        ----------
        OActiveSpace : rUnitaryActiveSpace
            Occupied Active Space Calculator.
        VActiveSpace : rUnitaryActiveSpace
            Virtual Active Space Calculator.
        """
        self.occ_calc = OActiveSpace
        self.vir_calc = VActiveSpace
        
        # at least one active space calculator should be provided
        assert self.occ_calc is not None or self.vir_calc is not None
        
        
    def calc_mo(self):
        
        if self.occ_calc is not None:
            moE_occ, moC_occ, mask_occ_act = self.occ_calc.calc_mo()
            
        else:
            moE_occ = self.vir_calc.mf.mo_energy[self.vir_calc.mf.mo_occ >= 1]
            moC_occ = self.vir_calc.mf.mo_coeff[:,self.vir_calc.mf.mo_occ >= 1]
            mask_occ_act = np.array([True]*self.vir_calc.Nocc)
            
        if self.vir_calc is not None:
            moE_vir, moC_vir, mask_vir_act = self.vir_calc.calc_mo()
            
        else:
            moE_vir = self.occ_calc.mf.mo_energy[self.occ_calc.mf.mo_occ < 1]
            moC_vir = self.occ_calc.mf.mo_coeff[:,self.occ_calc.mf.mo_occ < 1]
            mask_vir_act = np.array([True]*self.occ_calc.Nvir)
            
        moE_embed = np.hstack([moE_occ,moE_vir])
        moC_embed = np.hstack([moC_occ,moC_vir])
        
        # set useful masks
        self.mask_act = np.hstack([mask_occ_act,mask_vir_act])       
        self.mask_frz = ~self.mask_act
        
        self.mask_occ_act = np.array([False]*moE_embed.shape[0])
        self.mask_occ_act[0:mask_occ_act.shape[0]] = mask_occ_act
        self.mask_occ_frz = ~mask_occ_act
        
        self.mask_vir_act = np.array([False]*moE_embed.shape[0])
        self.mask_vir_act[mask_occ_act.shape[0]:] = mask_vir_act
        self.mask_vir_frz = ~mask_occ_act

        indx_frz = np.argwhere(~self.mask_act)[:,0]
            
        return moE_embed, moC_embed, indx_frz
    
if __name__ == '__main__':
    pass
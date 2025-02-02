#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib.py

Library of functions and classes for PySCF embedding code

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from scipy.linalg import eigh
from pyscf import gto
from pyscf import lo

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

def orbital_center(mol, orb_coeff):
    """
    Compute the expectation values of x, y, z for a given orbital.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        A PySCF Mole object that has been built and initialized.
    orb_coeff : np.ndarray
        orbital coefficients of shape (n_bas, n_orb). 

    Returns
    -------
    center : np.ndarray
        A length-3 numpy array [ <x>, <y>, <z> ] for the chosen orbital.
    """

    x, y, z = mol.intor('int1e_r')

    cx = orb_coeff.conj().T @ x @ orb_coeff
    cy = orb_coeff.conj().T @ y @ orb_coeff
    cz = orb_coeff.conj().T @ z @ orb_coeff

    return np.array([ np.diag(cx), np.diag(cy), np.diag(cz)]).T

class FC_AO_Ints:
    """
    Class for generating orbital overlap integrals between the AOs of a 
    composite system and localized orbitals of a fragment. 
    """
    
    def __init__(self, mol_comp, frag_inds, frag_inds_type='atom', basis_frag=None, orth=None):
        """
        Parameters
        ----------
        mol : PySCF mol object for composite system
        
        frag_inds : String
            Fragment indices
            
        frag_inds_type : Array
            Can be either 'atom' or orbital', Default : 'atom'
        
        basis_frag : String
            Basis set for fragment, Default is to use same as in mol
            
        orth : String
            Method for orthogonalization orbitals. Default is None.
            Options are 'lowdin' or 'meta-lowdin'
        """
        
        self.mol = mol_comp
                    
        # define fragment basis set
        if basis_frag is None:
            self.basis = self.mol.basis
        else:
            self.basis = basis_frag
        
        # determine indices of orbitals belonging to fragment
        self.mol2 = gto.M(atom=self.mol._atom, basis=self.basis, spin=0, unit='Bohr')
        
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
        self.orth_method = orth
        
        # < frag AO | frag AO > overlap
        self.s_ff = None
        
        # < frag AO | composite AO > overlap
        self.s_fc = None
                
    def calc_ao_ovlp(self):
        """
        Calculate overlap matrices between fragment AOs and composite AOs
                
        Returns:
        ----------
        s_ff : fragment-fragment overlap integral
        s_fc : fragment-composite overlap integral
        """
        
        # < frag AO | frag AO > overlap
        s_ff = self.mol2.intor_symmetric('int1e_ovlp')[np.ix_(self.frag_ao_inds,self.frag_ao_inds)]
        
        # < frag AO | composite AO > overlap
        s_fc = gto.mole.intor_cross('int1e_ovlp', self.mol2, self.mol)[self.frag_ao_inds]
            
        # orthogonalize frag AOs and return orthogonalized overlap integrals
        if self.orth_method is not None:
            s_ff, s_fc = self.orth(self.orth_method, s_ff, s_fc)
            
        self.s_ff, self.s_fc = s_ff, s_fc
        
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
        pop = np,sum( np.abs(C_f_mo)**2, axis=1)
        return pop
    
    def orth(self, method, s_ff, s_fc):
        """
        Orthogonalize fragment orbitals.
        """
          
        if method.lower == 'lowdin':            
            s_fo = lo.lowdin(s_ff)
            s_fc = s_fo.T @ s_ff @ s_fc
            s_ff = s_fo.T @ s_ff @ s_fo
            
        elif method.lower== 'meta-lowdin':
            pre_orth_ao = np.eye(self.mol.nao)
            weight = np.ones(pre_orth_ao.shape[0])
            s_fo = lo.nao._nao_sub(self.mol, weight, pre_orth_ao, s_ff)
            s_fc = s_fo.T @ s_ff @ s_fc
            s_ff = s_fo.T @ s_ff @ s_fo

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
    
    def __init__(self, mf, mo_occ_type):
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
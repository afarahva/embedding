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
    """
    
    # Unitary Projection Operators for active/frozen MOs (in canonoical MO basis)
    P_act=None
    P_frz=None
    Norb_act=None
    
    # empty method, to be replaced by child class
    def calc_projection(self):
        pass
    
    def __init__(self, mf, mo_occ_type):
        self.mf=mf
        self.mo_space = mo_occ_type
        
        mo_coeff, mo_energy, mo_occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
        
        self.mask_occ = mo_occ > 1e-10
        self.mask_vir = (self.mask_occ==False)
         
        self.moE_occ = mo_energy[self.mask_occ]
        self.moE_vir = mo_energy[self.mask_vir]
        self.moC_occ = mo_coeff[:,self.mask_occ]
        self.moC_vir = mo_coeff[:,self.mask_vir]
        
        if self.mo_space.lower() in ['o','occ','occupied']:
            self.moE = self.moE_occ 
            self.moC = self.moC_occ
        elif self.mo_space.lower() in ['v','vir','virtual']:
            self.moE = self.moE_vir
            self.moC = self.moC_vir
        
        self.Nmo, self.Nocc, self.Nvir = len(mo_energy), len(self.moE_occ), len(self.moE_vir)
        pass
        
    def pseudocanonical(self, moE, moC, P):
        
        # fock matrix in canonical MO basis
        fock = np.diag(moE)

        # diagonalize Fock operator in this basis
        E_P, C_P = matrix_projection_eigh(fock, P)
        
        # express eigenvectors of F in terms of AOs
        C_P = moC @ P @ C_P
            
        return E_P, C_P    
        
    def calc_mo(self):
        
        # project fock matrix
        self.calc_projection()
        
        # calculate projected MOs
        if self.P_act is not None:
            moE_act,moC_act = self.fock_rotate(self.moE,self.moC,self.P_act)
            moE_frz,moC_frz = self.fock_rotate(self.moE,self.moC,self.P_frz)
            
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
        
        return moE,moC,mask_act
    
class rActiveSpaceEmbedding:
    """
    Base class for embedding methods. Given an active-space of occupied 
    and virtual orbitals constructs relevant mo energies and 
    """
    
    def __init__(self,OActiveSpace, VActiveSpace):
        self.occ_calc = OActiveSpace
        self.vir_calc = VActiveSpace
        
    def kernel(self):
        
        self.moE_occ,self.moC_occ,self.mask_occ_act = self.occ_calc.calc_mo()
        self.moE_vir,self.moC_vir,self.mask_vir_act = self.vir_calc.calc_mo()
        
        self.moE_proj = np.hstack([self.moE_occ,self.moE_vir])
        self.moC_proj = np.hstack([self.moC_occ,self.moC_vir])
        self.mask_act = np.hstack([self.mask_occ_act,self.mask_vir_act])
        self.mask_frz = ~self.mask_act
        
        
        self.indx_act = np.argwhere(self.mask_act)[:,0]
        self.indx_frz = np.argwhere(~self.mask_act)[:,0]
            
        return self.moE_proj, self.moC_proj, self.indx_frz
    
if __name__ == '__main__':
    pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib.py

Library of functions and classes for PySCF embedding code
"""

import numpy as np
from pyscf import lo
import pyscf

class FockProjection:
    """
    Class for freezing non-canonical orbitals by seperate projections of 
    occupied/virtual orbitals. 
    
    Canonical orbitals are eigevectors of the Fock matrix. 
    F psi = E S psi
    Given a set of frozen non-canonical orbitals {phi}, we can re-diagonalize 
    the Fock operator in the subspace of non-frozen orbitals. 
    
    This defines a new set of active MOs spanning the non-frozen subspace, 
    which can subsequently be correlated with post-HF methods. 
    """
    
    def __init__(self,mf):
        """
        Initial Active Space Selector 
        
        Parameters
        ----------
        mf : PySCF Mean Field Object

        Returns
        -------
        None.

        """
        self.mf=mf
        mo_coeff, mo_energy, mo_occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
        
        self.mask_occ = mo_occ > 1e-10
        self.mask_vir = (self.mask_occ==False)
         
        self.moE_occ = mo_energy[self.mask_occ]
        self.moE_vir = mo_energy[self.mask_vir]
        self.moC_occ = mo_coeff[:,self.mask_occ]
        self.moC_vir = mo_coeff[:,self.mask_vir]
        
        self.Nmo, self.Nocc, self.Nvir = len(mo_energy), len(self.moE_occ), len(self.moE_vir)
        
    def project(self, P, fock):
        """
        Project Fock Matrix into a basis given by the columns of U.
        
        Returns MO coefficients in projected/rotated basis

        Parameters
        ----------
        P : Numpy Array
            Projection matrix in MO basis, columns should be orthonormal. 
        indx_act : 
            Indices of active orbitals
            
        Returns
        -------
        moE_U : projected mo energy
        evec_F_mo : eigenvectors of F in terms of canonical MOs
        """
        
        # rotate fock matrix into basis of U
        F_P = P.conj().T @ fock @ P
        
        # diagonalize Fock operator in this basis
        moE, evec_F = np.linalg.eigh(F_P)
        
        # express eigenvectors of F in canonical orbitals
        evec_F_mo = P @ evec_F
        
        return moE, evec_F_mo
        
        
    def calc_mo(self, P_occ_act, P_occ_frz, P_vir_act, P_vir_frz):
        """
        Calculated new MO energies and coefficients from projecting 
        occupied/virtual orbitals P_occ_act and P_vir_act
        
        If a transformation only on occupied or virtual is desired specify none
        as argument. 
        
        Parameters
        ----------
        P_occ_act : Numpy Array.
            Projection matrix for ACTIVE occupied MOs.
        P_occ_frz : Numpy Array.
            Projection matrix for FROZEN occupied MOs.
        P_vir_act : Numpy Array.
            Projection matrix for ACTIVE virtual MOs.
        P_vir_frz : Numpy Array.
            Projection matrix for FROZEN virtual MOs.

        Returns
        -------
        moE_proj : projected MO energies, 
            order (nocc_frz + nocc_act + nvir_act, + nvir_frozen)
        moC_proj : projected MO coefficients,
            column order (nocc_frz + nocc_act + nvir_act, + nvir_frozen)
        indx_act : indices of active orbitals
        indx_frz : indices of frozen orbitals
        """

        # project occupied orbitals
        if P_occ_act is not None:
            # set active orbitals
            self.moE_occ_act, evec_F_mo = self.project(P_occ_act, np.diag(self.moE_occ))
            self.moC_occ_act = self.moC_occ @ evec_F_mo
            
            # set frozen orbitals
            self.moE_occ_frz, evec_F_mo = self.project(P_occ_frz, np.diag(self.moE_occ))
            self.moC_occ_frz = self.moC_occ @ evec_F_mo
        else:
            self.moE_occ_act, self.moC_occ_act = self.moE_occ, self.moC_occ
            self.moE_occ_frz, self.moE_occ_frz = [],[]

        # project virtual orbitals
        if P_vir_act is not None:
            # set active orbitals
            self.moE_vir_act, evec_F_mo = self.project(P_vir_act, np.diag(self.moE_vir))
            self.moC_vir_act = self.moC_vir @ evec_F_mo
            
            # set frozen orbitals, note frozen VIRTUAL orbital energies/coefficients do not affect observables
            self.moE_vir_frz, evec_F_mo = self.project(P_vir_frz, np.diag(self.moE_vir))
            self.moC_vir_frz = self.moC_vir @ evec_F_mo
            # self.moE_vir_frz,self.moC_vir_frz = np.zeros(N_vir_frz),np.zeros(self.Nmo, N_vir_frz)
            
        else:
            self.moE_vir_act, self.mo_vir_act = self.moE_vir, self.moC_vir
            self.moE_vir_frz, self.moC_vir_frz = [], []

        # concatenate orbitals
        self.moE_proj = np.hstack([self.moE_occ_frz,self.moE_occ_act,self.moE_vir_act,self.moE_vir_frz])
        self.moC_proj = np.hstack([self.moC_occ_frz,self.moC_occ_act,self.moC_vir_act,self.moC_vir_frz])
        
        # indices of frozen orbitals
        N_occ_act, N_occ_frz, N_vir_act, N_vir_frz = (len(self.moE_occ_act), 
        len(self.moE_occ_frz), len(self.moE_vir_act), len(self.moE_vir_frz))
        
        self.indx_act = np.hstack([np.arange(N_occ_frz,self.Nocc),np.arange(self.Nocc,self.Nocc+N_vir_act)])
        self.indx_frz = np.hstack([np.arange(0,N_occ_frz),np.arange(self.Nocc+N_vir_act,self.Nmo)])

        return self.moE_proj, self.moC_proj, self.indx_act, self.indx_frz
    
class FC_AO_Ints:
    """
    Class for generating atomic orbital integrals between embedded fragments
    and composite system
    """
    
    def __init__(self, mol_comp, mf_comp=None):
        self.mol = mol_comp
        self.mf = mf_comp

    
    def calc_ovlp(self, frag_inds, frag_inds_type, basis_frag=None, orth=None):
        """
        Calculate overlap matrices between fragment/composite AOs
        
        Parameters
        ----------
        basis_frag : AO basis for fragment orbitals
        frag_mol : PySCF mol object for fragment
        frag_atm_inds : Indices of fragment atoms in composite mol object
        frag_ao_inds : Indices of fragment AOs in composite mol object
        orth : whether to orthogonalize fragment AOs
        
        Either one of frag_mol, frag_atoms, frag_ao_inds must be specified
        
        Returns:
        ----------
        s_ff : fragment-fragment overlap integral
        s_fc : fragment-composite overlap integral
        """
        if basis_frag is None:
            basis_frag = self.mol.basis
        
        # for an IAO basis
        if basis_frag=='iao':
            self.basis = 'minao'
            
            if self.mf is None:
                raise ValueError("iao basis requires mf to be provided")
            
            mol2 = pyscf.M(atom=self.mol._atom, basis=self.basis, spin=0, unit='Bohr')
            
            # select fragment AO indices if atoms are given
            if frag_inds_type.lower() != "orbital":
                self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                          mol2.aoslice_by_atom()[frag_inds]]).astype(int)
            else:
                self.frag_ao_inds = frag_inds
                
            # fragment intrinsic atomic orbitals in mol basis
            c_iao = lo.iao.iao(self.mol, self.mf.mo_coeff[:,self.mf.mo_occ>0])[:,self.frag_ao_inds]
            
            # fragment iao overlap matrix
            s_ff = c_iao.T @ self.mol.intor_symmetric('int1e_ovlp') @ c_iao
            
            # fragment iao | composite ao ovelap matrix
            s_fc = pyscf.gto.mole.intor_cross('int1e_ovlp', self.mol, self.mol)
            s_fc = c_iao.T @ s_fc
            
        else:
            self.basis=basis_frag
            
            mol2 = pyscf.M(atom=self.mol._atom, basis=self.basis, spin=0, unit='Bohr')
            
            # if fragment indices are atoms
            if frag_inds_type.lower() != "orbital":
                self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                          mol2.aoslice_by_atom()[frag_inds]]).astype(int)
            else:
                self.frag_ao_inds = frag_inds
            
            # fragment fragment AO overlap
            s_ff = mol2.intor_symmetric('int1e_ovlp')[np.ix_(self.frag_ao_inds,self.frag_ao_inds)]
            
            # fragment composite AO overlap
            s_fc = pyscf.gto.mole.intor_cross('int1e_ovlp', mol2, self.mol)[self.frag_ao_inds] 
            
        # orthogonalize
        if orth is not None:
            s_ff, s_fc = self.orthogonalize(orth, s_ff, s_fc)
            
        return s_ff, s_fc
    
    def orthogonalize(self, method, s_ff, s_fc):
        """
        Orthogonalize fragment AOs.
        
        Returns fragment-fragment overlap matrices (s_ff) 
            and fragment-composite overlap matrices (s_fc) 
        """
            
        # s_fo is overlap between orthogonalized fragment orbitals and fragment AOs
        # rows are AOs columns are LAOs. 
        
        if method.lower() == 'lowdin':
            s_fo = lo.lowdin(s_ff)
            
        elif method.lower() == 'schmidt':
            s_fo = lo.schmidt(s_ff)
        
        else:
            raise ValueError("Orthogonalization method not implemented")
            
        # re-express fragment/composite ovelap matrix in terms of orthogonal orbitals
        s_ff = s_fo.T @ s_ff @ s_fo
        s_fc = s_fo.T @ s_ff @ s_fc
         
        return s_ff, s_fc
    
    def population(self,s_ff,s_fc,mo):
        """
        Returns total population of occupied orbitals on fragment
        """
        C_f_mo = np.linalg.inv(s_ff) @ s_fc @ mo
        P_f_mo = np.abs(C_f_mo)**2
        pop = np.sum(P_f_mo,axis=0)
        return pop

        
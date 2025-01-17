#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pao.py

Projected Atomic Orbitals

author: Ardavan Farahvash, github.com/afarahva
"""

from pyscf.cc import ccsd

import numpy as np
from lib import rUnitaryActiveSpace, rActiveSpaceEmbedding
#from scipy.linalg import eigh, eigvalsh

# PAO generator
class rPAO(rUnitaryActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, 
                 frag_inds_type='atom', cutoff=0.1, scutoff=1e-3):
        """
        
        Parameters
        ----------
        mol : PySCF Molecule object
        mf : PySCF Mean Field object
        frag_inds : Iterable
            Indices of fragment atoms.
        mo_occ_type : String.
            One of either 'occupied' or 'virtual'. 
            
        OPTIONAL: 
        ----------
        """
        super().__init__(mf,mo_occ_type)
        self.cutoff=cutoff
        self.scutoff = scutoff
        
        if frag_inds_type.lower() == "atom":
            self.frag_atm_inds = frag_inds
            self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                      self.mf.mol.aoslice_by_atom()[frag_inds]]).astype(int)
        elif frag_inds_type.lower() == 'orbital':
            self.frag_atm_inds = None
            self.frag_ao_inds = frag_inds
        else:
            raise ValueError("frag_inds_type must be either 'atom' or 'orbital'")

        
    def calc_projection(self):
        
        S = self.mf.get_ovlp()
        
        # Determine PAOs in AO basis
        if self.mo_space.lower() in ['o','occ','occupied']:
            P = self.moC_vir @ self.moC_vir.T
        elif self.mo_space.lower() in ['v','vir','virtual']:
            P = self.moC_occ @ self.moC_occ.T
            
        C_pao = (np.eye(P.shape[0]) - P @ S)
        
        # Calculate population of PAOs on fragment atoms and keep only those 
        # with significant population
        fpop = np.einsum("ij,ij->j",C_pao[self.frag_ao_inds,:], (S@C_pao)[self.frag_ao_inds,:])
        C_pao_frag = C_pao[:,fpop>self.cutoff]
        
        # Orthonormalize fragment PAOs amongst each other
        S_pao_frag = C_pao_frag.T @ S @ C_pao_frag
        s,v = np.linalg.eigh(S_pao_frag)
        mask = s > self.scutoff
        C_pao_active = np.einsum("ab,ia->ib",v[:,mask]/ np.sqrt(s[None,mask]),C_pao_frag)
        
        # Generate Bath/Frozen PAOs
        C_pao_bath =  (np.eye(P.shape[0]) - P @ S - C_pao_active@C_pao_active.T@S)
        S_pao_bath = C_pao_bath.T @ S @ C_pao_bath
        s,v = np.linalg.eigh(S_pao_bath)
        
        mask = np.array( [False] * (self.Nocc + self.Nvir))
        if self.mo_space.lower() in ['o','occ','occupied']:
            mask[self.Nvir+C_pao_active.shape[1]:] = True
        elif self.mo_space.lower() in ['v','vir','virtual']:
            P = self.moC_vir @ self.moC_vir.T
            mask[self.Nocc+C_pao_active.shape[1]:] = True
        
        C_pao_frozen = np.einsum("ab,ia->ib",v[:,mask]/ np.sqrt(s[None,mask]),C_pao_bath)
        
        # Concatenate active and frozen PAOs
        C_final = np.hstack([C_pao_active,C_pao_frozen])
        
        # unitary transformation from MOs to PAOs
        u = self.moC.T @ self.mf.get_ovlp() @ C_final
        
        self.Norb_act = C_pao_active.shape[1]
        self.P_act = u[:,0:self.Norb_act]
        self.P_frz = u[:,self.Norb_act:]
        
        return self.P_act, self.P_frz
    
#%%

if __name__ == '__main__':
    import pyscf
    
    coords = \
    """
    O         -3.65830        0.00520       -0.94634
    H         -4.10550        1.27483       -1.14033
    C         -2.05632        0.04993       -0.35355
    C         -1.42969        1.27592       -0.14855
    C         -0.12337        1.31114        0.33487
    C          0.54981        0.12269        0.61082
    C         -0.08157       -1.10218        0.40403
    C         -1.38785       -1.13991       -0.07931
    H         -1.93037        2.15471       -0.35367
    H          0.34566        2.21746        0.48856
    H          1.51734        0.14971        0.96884
    H          0.41837       -1.98145        0.60889
    H         -1.85763       -2.04579       -0.23330
    """
    
    mol = pyscf.M(atom=coords,basis='ccpvdz',verbose=3)
    mf = mol.RHF().run()
    #%%
    frag_inds=[0,1]
    occ_calc = rPAO(mf, frag_inds, 'occ', cutoff=0.0)
    vir_calc = rPAO(mf, frag_inds, 'vir', cutoff=0.1)
    
    embed = rActiveSpaceEmbedding(occ_calc,vir_calc)
    moE_new, moC_new, indx_frz = embed.kernel()
    print(len(indx_frz))
    
    # embedded
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
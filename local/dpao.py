#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pao.py

Projected Atomic Orbitals
Generalized for Restricted and Unrestricted references.

author: Ardavan Farahvash, github.com/afarahva
"""
import numpy as np
from pyscf import lo
from pyscf_embedding.utils import ActiveSpace, HFEmbedding

# PAO generator
class PAO(ActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, frag_inds_type='atom', 
        cutoff_type="overlap", cutoff=0.1, scutoff=1e-3):
        """
        Parameters
        ----------
        mf : PySCF Mean Field object
        frag_inds : Iterable
            Indices of fragment atoms.
        mo_occ_type : String.
            One of either 'occupied' or 'virtual'. 
            
        OPTIONAL: 
        ----------
        frag_inds_type : String.
            Specify 'orbital' if supplying a list of orbital indices in 
            frag_inds instead of atom indices
            
        cutoff : Float or Int
            Cutoff for active orbitals. Default: 0.1
            
        cutoff_type : String
            Type of cutoff value. One of 'overlap', 'pct_occ', or 'norb'.
            
            'overlap' (default) assigns active MOs as those with a higher
            overlap value than the cutoff specified. 
            
            'norb' assigns active MOs as those with the higest overlap with 
            the fragment until the cutoff. 
        """
        super().__init__(mf, mo_coeff=mo_occ_type)
        self.cutoff = cutoff
        self.scutoff = scutoff
        self.cutoff_type = cutoff_type
        
        if frag_inds_type.lower() == "atom":
            self.frag_atm_inds = frag_inds
            # aoslice_by_atom is geometric, spin-independent
            self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                    self.mf.mol.aoslice_by_atom()[frag_inds]]).astype(int)
        
        elif frag_inds_type.lower() == 'orbital':
            self.frag_atm_inds = None
            self.frag_ao_inds = frag_inds
        
        else:
            raise ValueError("frag_inds_type must be either 'atom' or 'orbital'")
            
    def population(self, mo_coeff, method='meta-lowdin'):
        
        # Handle Unrestricted input recursively
        if isinstance(mo_coeff, (list, tuple)):
            return (self.population(mo_coeff[0], method), self.population(mo_coeff[1], method))

        # set population method
        if method=='mulliken':
            mo = mo_coeff
            s = self.mf.get_ovlp()
            # Handle if S is tuple (rare for pure geometric overlap, but possible in some contexts)
            if isinstance(s, (list, tuple)): s = s[0] 
        else:
            C = lo.orth_ao(self.mf.mol, method)
            s_mat = self.mf.get_ovlp()
            if isinstance(s_mat, (list, tuple)): s_mat = s_mat[0]
            
            mo = C.T @ s_mat @ mo_coeff
            s = C.T @ s_mat @ C
            
        # calculate the total population on fragment AOs
        fpops = []
        for i in range(mo.shape[1]):
            pops_i = np.diag(np.outer(mo[:,i],mo[:,i]) @ s)
            fpops_i = np.sum(pops_i[self.frag_ao_inds])
            fpops.append(fpops_i)
            
        fpops = np.array(fpops)
        return fpops
        
    def _project_one_spin(self, moC, S):
        """
        Helper function to generate PAOs for a single spin channel.
        """
        # Construct PAOs in AO basis
        # P is the density matrix of the space (occ or vir) being projected
        P = moC @ moC.T
        C_pao = P @ S # unnormalized PAOs
        
        # Calculate population of PAOs on fragment atoms and keep only those 
        # with significant population
        # fpop[j] = <pao_j | P_frag | pao_j> roughly
        fpop = np.einsum("ij,ij->j", C_pao[self.frag_ao_inds,:], (S @ C_pao)[self.frag_ao_inds,:])
        
        if self.cutoff_type.lower() in ['overlap','pop','population']:
            mask = fpop > self.cutoff
        elif self.cutoff_type.lower() in ['norb','norb_act']:
            indx_sort = np.flip(np.argsort(fpop))
            mask = np.zeros(len(fpop), dtype=bool)
            mask[indx_sort[0:self.cutoff]] = True 
        else:
            raise ValueError("Incorrect cutoff type. Must be one of 'overlap', or 'norb'" )
            
        # C_pao_frag = C_pao[:, mask]
        C_pao_frag = P@S[:,self.frag_ao_inds]
        S_pao_frag = C_pao_frag.T @ S @ C_pao_frag

        # Orthonormalize fragment PAOs amongst each other
        s, v = np.linalg.eigh(S_pao_frag)
        # Filter out linearly dependent PAOs
        mask_s = s > self.scutoff
        C_pao_active = np.einsum("ab,ia->ib", v[:, mask_s] / np.sqrt(s[None, mask_s]), C_pao_frag)
        
        # Generate Bath/Frozen PAOs (Project out active PAOs from original space)
        # C_bath = C_orig - C_act C_act^T S C_orig ?? 
        # Original code: C_pao_bath =  C_pao - C_pao_active@C_pao_active.T@S
        # Note: This logic assumes C_pao spans the whole space initially. 
        # C_pao was P@S. 
        C_pao_bath = C_pao - C_pao_active @ C_pao_active.T @ S
        S_pao_bath = C_pao_bath.T @ S @ C_pao_bath
        
        s_bath, v_bath = np.linalg.eigh(S_pao_bath)
        
        # Identify non-null bath vectors
        # We expect exactly (N_total - N_active) non-zero eigenvalues
        mask_bath = np.array([False] * len(s_bath))
        delmo = moC.shape[1] - C_pao_active.shape[1]
        
        if delmo > 0:
            mask_bath[-delmo:] = True
            
        C_pao_frozen = np.einsum("ab,ia->ib", v_bath[:, mask_bath] / np.sqrt(s_bath[None, mask_bath]), C_pao_bath)
        
        # Concatenate active and frozen PAOs
        C_final = np.hstack([C_pao_active, C_pao_frozen])
            
        # Unitary transformation from MOs to PAOs
        # u describes how to rotate Canonical MOs (moC) to get PAOs (C_final)
        u = moC.T @ S @ C_final
        
        norb_act = C_pao_active.shape[1]
        P_act = u[:, 0:norb_act]
        P_frz = u[:, norb_act:]
        
        return P_act, P_frz, norb_act

    def calc_projection(self, debug=False):
        
        S = self.mf.get_ovlp()
        if isinstance(S, (list, tuple)) or (isinstance(S, np.ndarray) and S.ndim==3):
            # Handle case where overlap might be returned as array of arrays (e.g. k-points or weird UKS)
            # Usually PySCF get_ovlp() returns one matrix for molecules.
            if len(S.shape)==3: S = S[0]

        if self.is_uhf:
            # Unrestricted: Project Alpha and Beta separately
            P_act_a, P_frz_a, norb_a = self._project_one_spin(self.moC[0], S)
            P_act_b, P_frz_b, norb_b = self._project_one_spin(self.moC[1], S)
            
            self.P_act = (P_act_a, P_act_b)
            self.P_frz = (P_frz_a, P_frz_b)
            self.Norb_act = (norb_a, norb_b)
            
        else:
            # Restricted
            P_act, P_frz, norb = self._project_one_spin(self.moC, S)
            
            self.P_act = P_act
            self.P_frz = P_frz
            self.Norb_act = norb
        
        return self.P_act, self.P_frz
    
#%%

if __name__ == '__main__':
    import pyscf
    from pyscf import cc
    
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
    
    print("\n--- RHF PAO Embedding ---")
    frag_inds=[0,1]
    occ_calc = None
    vir_calc = PAO(mf, frag_inds, 'vir', cutoff=0.1)
    
    embed = HFEmbedding(occ_calc, vir_calc)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(f"Number of frozen orbitals: {len(indx_frz)}")
    
    # embedded
    mycc = cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
    
    print("\n--- UHF PAO Embedding ---")
    mol_u = pyscf.M(atom=coords,basis='ccpvdz', spin=0, verbose=3)
    mf_u = mol_u.UHF().run()
    
    # Use same logic for UHF
    vir_calc_u = PAO(mf_u, frag_inds, 'vir', cutoff=0.1)
    embed_u = HFEmbedding(None, vir_calc_u)
    
    moE_u, moC_u, indx_frz_u = embed_u.calc_mo()
    print(f"Frozen Alpha: {len(indx_frz_u[0])}, Frozen Beta: {len(indx_frz_u[1])}")
    
    mycc_u = cc.UCCSD(mf_u)
    mycc_u.mo_coeff = moC_u
    mycc_u.frozen = indx_frz_u
    mycc_u.run()
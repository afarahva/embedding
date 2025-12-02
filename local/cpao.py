#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpao.py

Projected Atomic Orbitals

author: Ardavan Farahvash, github.com/afarahva
"""

from pyscf.cc import ccsd

import numpy as np
from pyscf import lo
from pyscf_embedding.lib  import rUnitaryActiveSpace, rWVFEmbedding
from pyscf.lib.scipy_helper import pivoted_cholesky


def pc_probe(matrix, pivot_subset=None , max_rank=None):
    """Runs full Cholesky decomp to record pivot spectrum."""
    matrix = matrix.copy()
    n = matrix.shape[0]
    pivots = np.arange(n)
    diagonals_history = []
    
    # machine tolerance
    machine_epsilon = np.finfo(np.double).eps
    tol = n * machine_epsilon * np.amax(np.diag(matrix))
    print(tol)
    
    # Handle pivot subset masking
    if pivot_subset is None:
        valid_indices = np.ones(n, dtype=bool)
    else:
        valid_indices = np.zeros(n, dtype=bool)
        valid_indices[pivot_subset] = True
    
    L = np.zeros((n, n))
    
    for k in range(n):
        # Mask invalid pivots
        current_diag = np.diag(matrix)[k:]
        current_global_indices = pivots[k:]
        
        search_values = current_diag.copy()
        for i, val in enumerate(search_values):
            if not valid_indices[current_global_indices[i]]:
                search_values[i] = -1.0

        local_best = np.argmax(search_values)
        local_pivot_idx = local_best + k
        max_val = matrix[local_pivot_idx, local_pivot_idx]

        # Stop at numerical noise floor
        if max_val <= tol:
            break
            
        diagonals_history.append(max_val)
        
        # Swap
        matrix[[k, local_pivot_idx]] = matrix[[local_pivot_idx, k]]
        matrix[:, [k, local_pivot_idx]] = matrix[:, [local_pivot_idx, k]]
        L[[k, local_pivot_idx]] = L[[local_pivot_idx, k]]
        pivots[[k, local_pivot_idx]] = pivots[[local_pivot_idx, k]]
        
        # Update
        L[k, k] = np.sqrt(max_val)
        L[k+1:, k] = matrix[k+1:, k] / L[k, k]
        outer_prod = np.outer(L[k+1:, k], L[k+1:, k])
        matrix[k+1:, k+1:] -= outer_prod
        
    return np.array(diagonals_history), L, pivots

def pivoted_cholesky(matrix, pivot_subset=None, cutoff=1e-5, cutoff_type="pop", max_rank=None):
    """
    Wrapper to perform Cholesky with various truncation criteria.
    cutoff_type: 'norb', 'pop', 'cond', 'largest_gap'
    """
    diagonals, L_full, pivots = pc_probe(matrix, pivot_subset, max_rank)
    
    # Determine Final Rank
    final_rank = len(diagonals)
    type_clean = cutoff_type.lower()
    
    if type_clean in ["norbital", "norb"]:
        final_rank = int(cutoff)
        
    elif type_clean in ["pop", "population", "overlap"]:
        # Cutoff is absolute threshold for diagonal value
        below_thresh = np.where(diagonals < cutoff)[0]
        if len(below_thresh) > 0:
            final_rank = below_thresh[0]
            
    elif type_clean in ["cond", "cond_number", "condition_number"]:
        # Cutoff is condition number cap (ratio of eigenvalues)
        # Threshold = max_pivot / (cond_cap^2)
        threshold = diagonals[0] * (cutoff**2)
        below_thresh = np.where(diagonals < threshold)[0]
        if len(below_thresh) > 0:
            final_rank = below_thresh[0]
            
    elif type_clean in ["gap","largest_gap"]:
        # Heuristic: largest log drop after initial orbitals
        if len(diagonals) > 3:
            log_diag = np.log10(diagonals)
            start = min(5, len(diagonals)//2)
            diffs = log_diag[:-1] - log_diag[1:]
            if len(diffs) > start:
                final_rank = np.argmax(diffs[start:]) + start + 1

    # Bounds check
    final_rank = min(final_rank, len(diagonals))
    
    # 1. The Localized Set (0 to final_rank)
    L_frag_perm = L_full[:, :final_rank]
    L_local = np.zeros_like(L_frag_perm)
    L_local[pivots] = L_frag_perm
    
    return L_local

# Cholesky PAO generator
class rCPAO(rUnitaryActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, frag_inds_type='atom', 
        cutoff_type="overlap", cutoff=1e-1):
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
        basis : String.
            Fragment basis set for occupied orbitals. Default: 'minao'
            
        cutoff : Float or Int
            Cutoff for active orbitals. Default: 0.1
            
        cutoff_type : String
            Type of cutoff value. One of 'overlap', or 'norb'.
            
            'overlap' (default) assigns active MOs as those with a higher
            overlap value than the cutoff specified. 
            
            'norb' assigns active MOs as those with the higest overlap with 
            the fragment until the cutoff.  d
        """
        super().__init__(mf, mo_coeff=mo_occ_type)
        self.cutoff=cutoff
        self.cutoff_type = cutoff_type
        
        if frag_inds_type.lower() == "atom":
            self.frag_atm_inds = frag_inds
            self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                      self.mf.mol.aoslice_by_atom()[frag_inds]]).astype(int)
        
        elif frag_inds_type.lower() == 'orbital':
            self.frag_atm_inds = None
            self.frag_ao_inds = frag_inds
        
        else:
            raise ValueError("frag_inds_type must be either 'atom' or 'orbital'")
            
    def population(self, mo_coeff, method='meta-lowdin'):
        
        # set population method
        if method=='mulliken':
            mo = mo_coeff
            s = self.mf.get_ovlp()
        else:
            C = lo.orth_ao(self.mf.mol, method)
            mo = C.T @ self.mf.get_ovlp() @ mo_coeff
            s = C.T @ self.mf.get_ovlp() @ C
            
        # calculate the total population on fragment AOs
        fpops = []
        for i in range(mo.shape[1]):
            pops_i = np.diag(np.outer(mo[:,i],mo[:,i]) @ s)
            fpops_i = np.sum(pops_i[self.frag_ao_inds])
            fpops.append(fpops_i)
            
        fpops = np.array(fpops)
        return fpops
        
    def calc_projection(self,debug=False):
        
        S = self.mf.get_ovlp()
        
        ### Generate active PAOs
        
        # Construct PAOs in AO basis
        P = self.moC @ self.moC.T
        
        C_pao_frag = pivoted_cholesky(P, 
                              pivot_subset=self.frag_ao_inds, 
                              cutoff=self.cutoff, cutoff_type=self.cutoff_type,
                              max_rank=self.moC.shape[1])
        
        norms = np.sqrt(np.diag(C_pao_frag.T @ S @ C_pao_frag))
        C_pao_active = C_pao_frag / norms       
        
        # create bath/environment orbitals
        C_pao_bath =  P@S - C_pao_active@C_pao_active.T@S
        S_pao_bath = C_pao_bath.T @ S @ C_pao_bath
        s,v = np.linalg.eigh(S_pao_bath)
        
        mask = np.array( [False] * (len(s)))
        delmo = self.moC.shape[1] - C_pao_active.shape[1]
        if delmo > 0:
            mask[-delmo:] = True
        C_pao_frozen = np.einsum("ab,ia->ib",v[:,mask]/ np.sqrt(s[None,mask])
                                 ,C_pao_bath)
        
        # Concatenate active and frozen PAOs
        C_final = np.hstack([C_pao_active,C_pao_frozen])
        self.C_pao_active = C_pao_active
        
        print(self.moC.shape, C_pao_active.shape, C_pao_frozen.shape)
        
        # unitary transformation from MOs to PAOs
        u = self.moC.T @ self.mf.get_ovlp() @ C_final
        
        self.Norb_act = C_pao_active.shape[1]
        self.P_act = u[:,0:self.Norb_act]
        self.P_frz = u[:,self.Norb_act:]
        
        if debug:
            self.C_pao_frag=C_pao_frag
            self.C_pao_active=C_pao_active
            self.C_pao_bath=C_pao_bath
            self.C_pao_frozen=C_pao_frozen
        
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
    frag_inds=[0,1,2]
    occ_calc = None
    vir_calc = rCPAO(mf, frag_inds, 'vir', cutoff=1e-1, cutoff_type="gap")
    
    embed = rWVFEmbedding(occ_calc, vir_calc)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(len(indx_frz))
    
    # embedded
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
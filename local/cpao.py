#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cpao.py

Projected Atomic Orbitals via Cholesky Decomposition
Generalized for Restricted and Unrestricted references.

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from pyscf import lo
from pyscf_embedding.utils import ActiveSpace, HFEmbedding

def pc_probe(matrix, pivot_subset=None , max_rank=None):
    """Runs full Cholesky decomp to record pivot spectrum."""
    matrix = matrix.copy()
    n = matrix.shape[0]
    pivots = np.arange(n)
    diagonals_history = []
    
    # machine tolerance
    machine_epsilon = np.finfo(np.double).eps
    tol = n * machine_epsilon * np.amax(np.diag(matrix))
    # print(tol)
    
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
            
    elif type_clean in ["cond", "cond_number", "condition_number"]:
        # Cutoff is condition number cap (ratio of eigenvalues)
        # Threshold = max_pivot / (cond_cap^2)
        threshold = diagonals[0] * (cutoff**2)
        below_thresh = np.where(diagonals < threshold)[0]
        if len(below_thresh) > 0:
            final_rank = below_thresh[0]

    # Bounds check
    final_rank = min(final_rank, len(diagonals))
    
    # 1. The Localized Set (0 to final_rank)
    L_frag_perm = L_full[:, :final_rank]
    L_local = np.zeros_like(L_frag_perm)
    L_local[pivots] = L_frag_perm
    
    return L_local

# Cholesky PAO generator
class CPAO(ActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, frag_inds_type='atom', 
        cutoff_type="cond", cutoff=1e-1):
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
            Type of cutoff value. One of 'overlap', or 'norb'.
            
            'cond' (default) assigns active MOs as those with a higher
            condition number value than the cutoff specified. 
            
            'norb' assigns active MOs as those with the higest overlap with 
            the fragment until the cutoff. 
        """
        super().__init__(mf, mo_coeff=mo_occ_type)
        self.cutoff = cutoff
        self.cutoff_type = cutoff_type
        
        if frag_inds_type.lower() == "atom":
            self.frag_atm_inds = frag_inds
            # Geometric indices are spin-independent
            self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                    self.mf.mol.aoslice_by_atom()[frag_inds]]).astype(int)
        
        elif frag_inds_type.lower() == 'orbital':
            self.frag_atm_inds = None
            self.frag_ao_inds = frag_inds
        
        else:
            raise ValueError("frag_inds_type must be either 'atom' or 'orbital'")
            
    def population(self, mo_coeff, method='meta-lowdin'):
        
        # Handle Unrestricted input
        if isinstance(mo_coeff, (list, tuple)):
            return (self.population(mo_coeff[0], method), self.population(mo_coeff[1], method))
        
        # set population method
        if method=='mulliken':
            mo = mo_coeff
            s = self.mf.get_ovlp()
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
        Helper function to generate CPAOs for a single spin channel.
        """
        # Construct PAOs in AO basis
        # P = density matrix of the subspace
        P = moC @ moC.T
        
        # Perform Pivoted Cholesky on Density Matrix restricting pivots to fragment
        C_pao_frag = pivoted_cholesky(P, 
                              pivot_subset=self.frag_ao_inds, 
                              cutoff=self.cutoff, cutoff_type=self.cutoff_type,
                              max_rank=moC.shape[1])
        
        # Normalize
        # Note: pivoted_cholesky returns L such that P ~ L L^T
        # We project into metric S to normalize properly
        norms = np.sqrt(np.diag(C_pao_frag.T @ S @ C_pao_frag))
        
        # Avoid division by zero if norm is extremely small (though PC shouldn't return those usually)
        norms[norms < 1e-12] = 1.0 
        C_pao_active = C_pao_frag / norms        
        
        # Create bath/environment orbitals
        # Project out the active PAOs from the original subspace
        # C_bath = P S - C_act C_act^T S = (P - C_act C_act^T) S
        C_pao_bath =  P @ S - C_pao_active @ C_pao_active.T @ S
        S_pao_bath = C_pao_bath.T @ S @ C_pao_bath
        
        s, v = np.linalg.eigh(S_pao_bath)
        
        # Identify non-null bath vectors
        mask_bath = np.array([False] * (len(s)))
        delmo = moC.shape[1] - C_pao_active.shape[1]
        
        if delmo > 0:
            mask_bath[-delmo:] = True
            
        C_pao_frozen = np.einsum("ab,ia->ib", v[:, mask_bath] / np.sqrt(s[None, mask_bath]), C_pao_bath)
        
        # Concatenate active and frozen PAOs
        C_final = np.hstack([C_pao_active, C_pao_frozen])
        
        # Unitary transformation from MOs to PAOs
        # u describes rotation of canonical MOs
        u = moC.T @ S @ C_final
        
        norb_act = C_pao_active.shape[1]
        P_act = u[:, 0:norb_act]
        P_frz = u[:, norb_act:]
        
        return P_act, P_frz, norb_act

    def calc_projection(self, debug=False):
        
        S = self.mf.get_ovlp()
        if isinstance(S, (list, tuple)) or (isinstance(S, np.ndarray) and S.ndim==3):
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
        
        if debug:
            pass
        
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
    
    print("\n--- RHF CPAO Embedding ---")
    frag_inds=[0,1,2]
    occ_calc = None
    vir_calc = CPAO(mf, frag_inds, 'vir', cutoff=1e-3, cutoff_type="cond")
    
    embed = HFEmbedding(occ_calc, vir_calc)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(f"Frozen Orbitals: {len(indx_frz)}")
    
    mycc = cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
    
    print("\n--- UHF CPAO Embedding ---")
    mol_u = pyscf.M(atom=coords,basis='ccpvdz', spin=0, verbose=3)
    mf_u = mol_u.UHF().run()
    
    vir_calc_u = CPAO(mf_u, frag_inds, 'vir', cutoff=1e-3, cutoff_type="cond")
    embed_u = HFEmbedding(None, vir_calc_u)
    
    moE_u, moC_u, indx_frz_u = embed_u.calc_mo()
    print(f"Frozen Alpha: {len(indx_frz_u[0])}, Frozen Beta: {len(indx_frz_u[1])}")
    
    mycc_u = cc.UCCSD(mf_u)
    mycc_u.mo_coeff = moC_u
    mycc_u.frozen = indx_frz_u
    mycc_u.run()
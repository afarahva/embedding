#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pao.py

Projected Atomic Orbitals
Generalized for Restricted and Unrestricted references.

author: Ardavan Farahvash, github.com/afarahva
"""
import numpy as np
from scipy.linalg import eigh
from pyscf import lo
from pyscf_embedding.utils import ActiveSpace, HFEmbedding, FC_AO_Ints

# PAO generator
class PAO(ActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, frag_inds_type='atom', 
        basis=None, cutoff_type="overlap", cutoff=0, ov_adjust=True):
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
        self.cutoff_type = cutoff_type
        if self.basis is None:
            self.basis=mf.mol.basis
        else:
            self.basis = basis
        self.ov_adjust=ov_adjust
        self.fc_ints = FC_AO_Ints(mf.mol, 
                                  frag_inds, frag_inds_type=frag_inds_type, 
                                  basis_frag=basis)
        
        if frag_inds_type.lower() == "atom":
            self.frag_atm_inds = frag_inds
            self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                    self.mf.mol.aoslice_by_atom()[frag_inds]]).astype(int)
        
        elif frag_inds_type.lower() == 'orbital':
            self.frag_atm_inds = None
            self.frag_ao_inds = frag_inds
        
        else:
            raise ValueError("frag_inds_type must be either 'atom' or 'orbital'")
        
    def _project_one_spin(self, moC, S, ovlp_ff, ovlp_fc):
        """
        Helper function to generate PAOs for a single spin channel.
        """
        # Construct PAOs in AO basis
        C_pao = moC @ moC.T @ ovlp_fc # unnormalized PAOs
            
        S_pao_frag = C_pao.T @ S @ C_pao

        if self.ov_adjust:
            s, u = np.linalg.eigh(S_pao_frag)
        else:
            s, u = eigh(S_pao_frag, ovlp_ff)
        s = np.abs(s)
        
        # Filter out linearly dependent PAOs
        if self.cutoff_type.lower() in ['overlap','pop','population']:
            mask_act = s >= self.cutoff

        if self.cutoff_type.lower() in ['norb','norb_act']:
            assert isinstance(self.cutoff, int)
            mask_act = np.zeros(len(s), dtype=bool)
            top_inds = np.argsort(s)[-self.cutoff:]
            mask_act[top_inds] = True
            
        C_pao_active = np.einsum("ia,ab->ib", C_pao, u[:, mask_act] / np.sqrt(s[None, mask_act]))
        
        # Generate Bath/Frozen PAOs (Project out active PAOs from original space)
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
        v = moC.T @ S @ C_final
        
        norb_act = C_pao_active.shape[1]
        P_act = v[:, 0:norb_act]
        P_frz = v[:, norb_act:]
        
        return P_act, P_frz, norb_act

    def calc_projection(self, debug=False):
        
        S = self.mf.get_ovlp()
        ovlp_ff, ovlp_fc = self.fc_ints.calc_ao_ovlp()

        if isinstance(S, (list, tuple)) or (isinstance(S, np.ndarray) and S.ndim==3):
            # Handle case where overlap might be returned as array of arrays (e.g. k-points or weird UKS)
            # Usually PySCF get_ovlp() returns one matrix for molecules.
            if len(S.shape)==3: S = S[0]

        if self.is_uhf:
            # Unrestricted: Project Alpha and Beta separately
            P_act_a, P_frz_a, norb_a = self._project_one_spin(self.moC[0], S, ovlp_ff, ovlp_fc)
            P_act_b, P_frz_b, norb_b = self._project_one_spin(self.moC[1], S, ovlp_ff, ovlp_fc)
            
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
    vir_calc = PAO(mf, frag_inds, 'vir', cutoff=1e-4)
    
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
    vir_calc_u = PAO(mf_u, frag_inds, 'vir', cutoff=1e-4)
    embed_u = HFEmbedding(None, vir_calc_u)
    
    moE_u, moC_u, indx_frz_u = embed_u.calc_mo()
    print(f"Frozen Alpha: {len(indx_frz_u[0])}, Frozen Beta: {len(indx_frz_u[1])}")
    
    mycc_u = cc.UCCSD(mf_u)
    mycc_u.mo_coeff = moC_u
    mycc_u.frozen = indx_frz_u
    mycc_u.run()
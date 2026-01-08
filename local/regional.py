#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regional.py

Regional Embedding and Regional Active Space / SPADE code
Generalized for Restricted and Unrestricted references.

author: Ardavan Farahvash, github.com/afarahva
"""     

import numpy as np
from pyscf_embedding.utils import ActiveSpace, HFEmbedding, FC_AO_Ints

class RegionalActiveSpace(ActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, 
        frag_inds_type="atom", basis='minao',
        cutoff_type="overlap", cutoff=0.1, orth=False, frozen_core=False):
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
            Type of cutoff value. One of 'overlap', 'pct_occ', or 'norb'.
            
            'overlap' (default) assigns active MOs as those with a higher
            overlap value than the cutoff specified. 
            
            'auto' assigns active MOs based on the spade approach, using the
            inflection point of the occupancy curve.
            
            'pct_occ' assigns active MOs as those with the higest overlap with 
            the fragment until a percentage of the total overlap is reached.
            
            'norb' assigns active MOs as those with the higest overlap with 
            the fragment until the cutoff. 
            
        orth : Bool
            Whether to orthogonalization fragment orbitals. Default is False.
            
        frozen_core : Bool
            Whether to freeze all core orbital when generating active space.
        """
        super().__init__(mf, mo_coeff=mo_occ_type, frozen_core=frozen_core)
        self.cutoff = cutoff
        self.cutoff_type = cutoff_type
        self.basis = basis
        self.fc_ints = FC_AO_Ints(mf.mol, 
                                  frag_inds, frag_inds_type=frag_inds_type, 
                                  basis_frag=basis, orth=orth)
        
    def _get_active_mask(self, s):
        """
        Helper method to determine active mask based on singular values s
        """
        if self.cutoff_type.lower() in ['overlap','pop','population']:
            mask_act = s >= self.cutoff
            
        elif self.cutoff_type.lower() in ['spade', 'auto']:
            if not isinstance(self.cutoff, int):
                self.cutoff=0
                from warnings import warn
                warn("For spade/auto, cutoff value must be an int representing the number of additional orbitals to include form the inflection point of the population curve")
            
            # Simple derivative check for inflection
            ds = s[1:] - s[0:-1]
            if len(ds) == 0:
                indx_max = 0
            else:
                indx_max = np.argmax(ds)
            
            # Apply offset
            indx_cut = indx_max - self.cutoff
            mask_act = np.zeros(len(s), dtype=bool)
            mask_act[indx_cut+1:] = True
            
        elif self.cutoff_type.lower() in ['pct_occ','occ']:
            # Sort descending for accumulation
            s_sorted = np.sort(s)[::-1]
            cumsum = np.cumsum(s_sorted / np.sum(s_sorted))
            
            # Check how many orbitals needed to hit cutoff
            n_needed = np.searchsorted(cumsum, self.cutoff) + 1
            
            # Map back to original indices (s is usually ascending from eigh)
            # We select the N largest values
            mask_act = np.zeros(len(s), dtype=bool)
            # Indices of largest N elements
            top_inds = np.argsort(s)[-n_needed:]
            mask_act[top_inds] = True
            
        elif self.cutoff_type.lower() in ['norb','norb_act']:
            assert isinstance(self.cutoff, int)
            mask_act = np.zeros(len(s), dtype=bool)
            # Select largest N elements
            top_inds = np.argsort(s)[-self.cutoff:]
            mask_act[top_inds] = True

        else:
            raise ValueError("Incorrect cutoff type. Must be one of 'overlap', 'pct_occ' or 'Norb'" )
            
        return mask_act

    def _project_one_spin(self, moC, s_ff, s_fc):
        """
        Performs projection and diagonalization for a single spin channel.
        """
        ovlp_f_mo = s_fc @ moC
        
        # P = (S_fi C)^T S_ff^-1 (S_fi C)
        P_proj = ovlp_f_mo.conj().T @ np.linalg.inv(s_ff) @ ovlp_f_mo
        
        # diagonalize 
        s, u = np.linalg.eigh(P_proj)
        s[s < 0] = 0
        
        # select indices of active orbitals
        mask_act = self._get_active_mask(s)
        
        # construct unitary projector 
        P_act = u[:, mask_act]
        P_frz = u[:, ~mask_act]
        norb_act = np.sum(mask_act)
        
        return P_act, P_frz, norb_act, s

    def calc_projection(self, debug=False):
        
        # compute projection operator 
        # Note: self.fc_ints should handle tuple return for IAO if moC is tuple
        if self.basis == 'iao':
            ovlp_ff, ovlp_fc = self.fc_ints.calc_ao_ovlp(moC_occ=self.moC)
        else:
            ovlp_ff, ovlp_fc = self.fc_ints.calc_ao_ovlp()

        if self.is_uhf:
            # Handle Unrestricted References
            # Note: For standard bases (minao), ovlp_ff/fc are single matrices (geometry only).
            # For IAO, they might be tuples if spin-dependent.
            
            # Helper to handle polymorphism of overlap matrices
            get_ovlp = lambda obj, idx: obj[idx] if isinstance(obj, (tuple, list)) else obj
            
            P_act_a, P_frz_a, norb_a, s_a = self._project_one_spin(
                self.moC[0], get_ovlp(ovlp_ff, 0), get_ovlp(ovlp_fc, 0)
            )
            P_act_b, P_frz_b, norb_b, s_b = self._project_one_spin(
                self.moC[1], get_ovlp(ovlp_ff, 1), get_ovlp(ovlp_fc, 1)
            )
            
            self.P_act = (P_act_a, P_act_b)
            self.P_frz = (P_frz_a, P_frz_b)
            self.Norb_act = (norb_a, norb_b)
            
            if debug:
                self.s_proj = (s_a, s_b)
                self.ds_proj = (s_a[1:] - s_a[:-1], s_b[1:] - s_b[:-1])
                # Masks are re-calculated in _project_one_spin, not stored here directly
                # but can be inferred from Norb_act if needed.
                
        else:
            # Handle Restricted References
            P_act, P_frz, norb, s = self._project_one_spin(self.moC, ovlp_ff, ovlp_fc)
            
            self.P_act = P_act
            self.P_frz = P_frz
            self.Norb_act = norb
            
            if debug:
                self.P_proj = None # P_proj is transient in the helper
                self.s_proj = s
                self.ds_proj = s[1:] - s[0:-1]
        
        return self.P_act, self.P_frz
    
# Subsystem Projected Atomic DEcomposition
class SPADEActiveSpace(ActiveSpace):
    def __init__(self, mf, frag_inds, mo_occ_type, 
                 cutoff_type="spade", cutoff=0, frozen_core=False):
        
        super().__init__(mf, mo_coeff=mo_occ_type, frozen_core=frozen_core)
        self.frag_inds = frag_inds
        self.cutoff = cutoff
        self.cutoff_type = cutoff_type
        
    def _get_nact(self, s):
        """
        Helper method to determine active mask based on singular values s
        """
        if self.cutoff_type.lower() in ['overlap','pop','population']:
            n_act_mos = np.sum(s >= self.cutoff)
            
        elif self.cutoff_type.lower() in ['spade', 'auto']:
            if not isinstance(self.cutoff, int):
                self.cutoff=0
                from warnings import warn
                warn("For spade/auto, cutoff value must be an int representing the number of additional orbitals to include form the inflection point of the population curve")
            if len(s) > 1:
                delta_s = [-(s[i+1] - s[i]) for i in range(len(s) - 1)]
                n_act_mos = np.argpartition(delta_s, -1)[-1] + 1
            else:
                n_act_mos = len(s)
            
        elif self.cutoff_type.lower() in ['norb','norb_act']:
            n_act_mos = self.cutoff

        else:
            raise ValueError("Incorrect cutoff type. Must be one of 'overlap', 'spade' or 'norb'" )
            
        return n_act_mos
    
    def _spade_one_spin(self, moC, S_half, frag_ao_inds):
        # Project MOs onto orthogonalized AOs
        # orthogonal_orbitals (N_frag_ao, N_mo)
        orthogonal_orbitals = (S_half @ moC)[frag_ao_inds, :]
        
        # SVD, u:naoxnao, s:nao, vh:nmoxnmo
        u, s, vh = np.linalg.svd(orthogonal_orbitals, full_matrices=True)
        
        n_act_mos = self._get_nact(s)

        P_act = vh.T[:, :n_act_mos]
        P_frz = vh.T[:, n_act_mos:]
        
        return P_act, P_frz, n_act_mos, s

    def calc_projection(self, debug=False):
        from scipy.linalg import fractional_matrix_power
        
        # 1. Identify Fragment Indices
        frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                    self.mf.mol.aoslice_by_atom()[self.frag_inds]]).astype(int)
        
        # 2. Prepare Overlap Matrix
        S = self.mf.get_ovlp()
        if isinstance(S, (list, tuple)) or (isinstance(S, np.ndarray) and S.ndim == 3):
            # Handle cases where S might be [S_a, S_b] or k-points (take Gamma)
            S = S[0]
            
        S_half = fractional_matrix_power(S, 0.5)
        
        # 3. Calculate Projection
        if self.is_uhf:
            P_act_a, P_frz_a, norb_a, s_a = self._spade_one_spin(self.moC[0], S_half, frag_ao_inds)
            P_act_b, P_frz_b, norb_b, s_b = self._spade_one_spin(self.moC[1], S_half, frag_ao_inds)
            
            self.P_act = (P_act_a, P_act_b)
            self.P_frz = (P_frz_a, P_frz_b)
            self.Norb_act = (norb_a, norb_b)
            
            if debug:
                self.s_proj = (s_a, s_b)
                
        else:
            P_act, P_frz, norb, s = self._spade_one_spin(self.moC, S_half, frag_ao_inds)
            
            self.P_act = P_act
            self.P_frz = P_frz
            self.Norb_act = norb
            
            if debug:
                self.s_proj = s
        
        return self.P_act, self.P_frz

# Standard Regional Embedding
class RegionalEmbedding(HFEmbedding):
    
    def __init__(self, mf, frag_inds, frag_inds_type='atom', basis_occ='minao', 
                 basis_vir=None, cutoff_occ=0.1, cutoff_vir=0.1,
                 orth=None, frozen_core=False):
        
        occ_calc = RegionalActiveSpace(mf, frag_inds, 'occupied', 
            frag_inds_type=frag_inds_type, basis=basis_occ, cutoff=cutoff_occ, 
            cutoff_type="overlap", orth=orth, frozen_core=frozen_core)
    
        occ_calc.calc_mo()
        
        vir_calc = RegionalActiveSpace(mf, frag_inds, 'virtual', 
            frag_inds_type=frag_inds_type, basis=basis_vir, 
            cutoff=occ_calc.Norb_act, cutoff_type="norb", 
            orth=orth, frozen_core=False)
        
        super().__init__(occ_calc, vir_calc)
         
    def kernel(self):
        moE_embed, moC_embed, indx_frz = super().calc_mo()
        return moE_embed, moC_embed, indx_frz
    
# Automated Valence Active Space Method
class AVAS(HFEmbedding):
    
    def __init__(self, mf, frag_inds, frag_inds_type='atom', min_basis='minao', 
                 cutoff_occ=0.1, cutoff_vir=0.1, orth=None, frozen_core=False):
        
        occ_calc = RegionalActiveSpace(mf, frag_inds, 'occupied', 
            frag_inds_type=frag_inds_type, basis=min_basis, cutoff=cutoff_occ, 
            cutoff_type="overlap", orth=orth, frozen_core=frozen_core)
        
        vir_calc = RegionalActiveSpace(mf, frag_inds, 'virtual', 
            frag_inds_type=frag_inds_type, basis=min_basis, cutoff=cutoff_vir, 
            cutoff_type="overlap", orth=orth, frozen_core=False)
        
        super().__init__(occ_calc, vir_calc)
         
    def calc_mo(self):
        moE_embed, moC_embed, indx_frz = super().calc_mo()
        return moE_embed, moC_embed, indx_frz     
    
# # Subsystem Projected Atomic DEcomposition
class SPADE(HFEmbedding):
    
    def __init__(self, mf, frag_inds, frozen_core=False):
                
        occ_calc = SPADEActiveSpace(mf, frag_inds, 'occupied',
                                    frozen_core=frozen_core)
        
        vir_calc = SPADEActiveSpace(mf, frag_inds, 'virtual', 
                                    frozen_core=False)
        
        super().__init__(occ_calc, vir_calc)
         
    def kernel(self):
        moE_embed, moC_embed, indx_frz = super().calc_mo()
        return moE_embed, moC_embed, indx_frz      

#%%
if __name__ == '__main__':
    import pyscf
    from pyscf.tools import cubegen
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
    
    mol = pyscf.M(atom=coords,basis='ccpvdz',spin=0,verbose=4)
    mol.build()
    mf = mol.RHF().run()
    
    # calculate localized orbital energies/coefficients
    frag_inds=[0,1]
    basis_occ='minao'
    basis_vir=mol.basis
    cutoff_occ=0.1
    cutoff_vir=0.1
    
    # traditional regional embedding
    re = RegionalEmbedding(mf, frag_inds, 'atom', basis_occ, basis_vir, cutoff_occ, cutoff_vir, orth=False, frozen_core=False)
    moE_re, moC_new, indx_frz_re = re.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz_re
    mycc.run()
    print(mycc.e_corr)
    
    #%%
    # spade embedding
    occ_calc = SPADEActiveSpace(mf, frag_inds, 'occ')
    vir_calc = SPADEActiveSpace(mf, frag_inds, 'vir')
    embed = HFEmbedding(occ_calc, vir_calc)
    moE_spade, moC_new, indx_frz_spade = embed.calc_mo()
    
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz_spade
    mycc.run()
    print(mycc.e_corr)
    #%%
    
    

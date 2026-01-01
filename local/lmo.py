#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lmo.py

Pipek-Mezey / Foster-Boys / Edmiston-Ruedenberg localization
Generalized for Restricted and Unrestricted references.

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from pyscf import lo
from pyscf_embedding.utils import ActiveSpace, HFEmbedding

# active-space projector based on 
# Pipek-Mezey/Foster-Boys/Edmiston-Ruedenberg LMOs
class LMOActiveSpace(ActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, localizer, 
                 frag_inds_type='atom', cutoff_type="overlap", cutoff=0.1, 
                 pop_method='meta-lowdin', frozen_core=False):
        """
        Parameters
        ----------
        mf : PySCF Mean Field object
        frag_inds : Iterable
            Indices of fragment atoms.
        mo_occ_type : String.
            One of either 'occupied' or 'virtual'. 
        localizer : Object or Tuple of Objects
            PySCF localizer object (e.g. lo.Boys, lo.PM).
            For UHF, must be a tuple (localizer_alpha, localizer_beta).
            
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
            
        pop_method : String
            Pop method for selecting MOs. Default is meta-lowdin.
        """
        super().__init__(mf, mo_coeff=mo_occ_type, frozen_core=frozen_core)

        self.localizer = localizer
        self.cutoff = cutoff
        self.cutoff_type = cutoff_type
        self.pop_method = pop_method
        
        if frag_inds_type.lower() == "atom":
            self.frag_atm_inds = frag_inds
            # Geometric indices are spin independent
            self.frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                    self.mf.mol.aoslice_by_atom()[frag_inds]]).astype(int)
        
        elif frag_inds_type.lower() == 'orbital':
            self.frag_atm_inds = None
            self.frag_ao_inds = frag_inds
        else:
            raise ValueError("frag_inds_type must be either 'atom' or 'orbital'")

    
    def population(self, mo_coeff, method='meta-lowdin'):
        
        # Handle Unrestricted (recursive)
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
    
    def unitary_mo_to_lmo(self, mo_coeff, lmo_coeff):
        
        # Handle Unrestricted
        if isinstance(mo_coeff, (list, tuple)):
            u_a = self.unitary_mo_to_lmo(mo_coeff[0], lmo_coeff[0])
            u_b = self.unitary_mo_to_lmo(mo_coeff[1], lmo_coeff[1])
            return (u_a, u_b)

        s = self.mf.get_ovlp()
        if isinstance(s, (list, tuple)): s = s[0]
        
        U = mo_coeff.T @ s @ lmo_coeff
        return U
    
    def _project_one_spin(self, moC, localizer):
        """
        Helper to run localization and projection for a single spin channel.
        """
        # Run localization kernel
        lmo_coeff = localizer.kernel()
        
        # Calculate Population
        # Note: self.population handles simple arrays correctly
        frag_pop = self.population(lmo_coeff, method=self.pop_method)
        
        if self.cutoff_type.lower() in ['overlap','pop','population']:
            mask_act = frag_pop > self.cutoff
            mask_frz = ~mask_act
            
        elif self.cutoff_type.lower() in ['spade', 'auto']:
            if not isinstance(self.cutoff, int):
                raise ValueError("For SPADE, cutoff value must be an int.")
            
            s = np.sort(frag_pop)
            ds = s[1:] - s[0:-1]
            if len(ds) == 0:
                indx_max = 0
            else:
                indx_max = np.argmax(ds) - self.cutoff
            
            mask_act = np.zeros(len(s), dtype=bool)
            indx_sort = np.argsort(frag_pop)
            # Safe indexing
            if indx_max + 1 < len(indx_sort):
                mask_act[indx_sort[indx_max+1:]] = True
            mask_frz = ~mask_act

        elif self.cutoff_type.lower() in ['norb','norb_act']:
            indx_sort = np.flip(np.argsort(frag_pop))
            mask_act = np.zeros(len(frag_pop), dtype=bool)
            mask_act[indx_sort[0:self.cutoff]] = True
            mask_frz = ~mask_act
            
        else:
            raise ValueError("Incorrect cutoff type. Must be one of 'overlap' or 'norb'" )
        
        # Unitary matrix U = C_can.T S C_lmo
        s_mat = self.mf.get_ovlp()
        if isinstance(s_mat, (list, tuple)): s_mat = s_mat[0]
        u = moC.T @ s_mat @ lmo_coeff
        
        # Unitary projectors
        P_act = u[:, mask_act]
        P_frz = u[:, mask_frz]
        norb_act = np.sum(mask_act)
        
        return P_act, P_frz, norb_act, lmo_coeff, frag_pop

    def calc_projection(self, **kwargs):
        
        if self.is_uhf:
            # Check if localizer is a tuple
            if not isinstance(self.localizer, (list, tuple)) or len(self.localizer) != 2:
                raise ValueError("For UHF references, 'localizer' must be a tuple of (alpha_loc, beta_loc).")
            
            P_act_a, P_frz_a, norb_a, lmo_a, pop_a = self._project_one_spin(self.moC[0], self.localizer[0])
            P_act_b, P_frz_b, norb_b, lmo_b, pop_b = self._project_one_spin(self.moC[1], self.localizer[1])
            
            self.P_act = (P_act_a, P_act_b)
            self.P_frz = (P_frz_a, P_frz_b)
            self.Norb_act = (norb_a, norb_b)
            
            # Save state
            self.lmo_coeff = (lmo_a, lmo_b)
            self.frag_pop = (pop_a, pop_b)
            # self.u is not strictly stored as attribute in helper but computed there. 
            # If needed for debugging:
            s_mat = self.mf.get_ovlp()
            if isinstance(s_mat, (list, tuple)): s_mat = s_mat[0]
            self.u = (self.moC[0].T @ s_mat @ lmo_a, self.moC[1].T @ s_mat @ lmo_b)

        else:
            # Restricted
            P_act, P_frz, norb, lmo, pop = self._project_one_spin(self.moC, self.localizer)
            
            self.P_act = P_act
            self.P_frz = P_frz
            self.Norb_act = norb
            
            self.lmo_coeff = lmo
            self.frag_pop = pop
            s_mat = self.mf.get_ovlp()
            if isinstance(s_mat, (list, tuple)): s_mat = s_mat[0]
            self.u = self.moC.T @ s_mat @ lmo
            
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
    
    # Check RHF Boys
    print("\n--- RHF Boys Embedding ---")
    frag_inds=[0,1]
    
    # Setup localizers manually
    loc_occ = lo.Boys(mol, mf.mo_coeff[:, mf.mo_occ > 0])
    loc_vir = lo.Boys(mol, mf.mo_coeff[:, mf.mo_occ == 0])
    
    occ_calc = LMOActiveSpace(mf, frag_inds, 'occ', loc_occ, cutoff=0.01)
    vir_calc = LMOActiveSpace(mf, frag_inds, 'vir', loc_vir, cutoff=0.01)
    
    embed = HFEmbedding(occ_calc, vir_calc)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(f"Frozen Orbitals: {len(indx_frz)}")
    
    mycc = cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
    
    # Check UHF Boys
    print("\n--- UHF Boys Embedding ---")
    mol_u = pyscf.M(atom=coords, basis='ccpvdz', spin=0, verbose=3)
    mf_u = mol_u.UHF().run()
    
    # For UHF, we must create separate localizers for Alpha and Beta
    
    # Occupied Localizers
    occ_mask_a = mf_u.mo_occ[0] > 0
    occ_mask_b = mf_u.mo_occ[1] > 0
    
    loc_occ_a = lo.Boys(mol_u, mf_u.mo_coeff[0][:, occ_mask_a])
    loc_occ_b = lo.Boys(mol_u, mf_u.mo_coeff[1][:, occ_mask_b])
    
    # Virtual Localizers
    vir_mask_a = mf_u.mo_occ[0] == 0
    vir_mask_b = mf_u.mo_occ[1] == 0
    
    loc_vir_a = lo.Boys(mol_u, mf_u.mo_coeff[0][:, vir_mask_a])
    loc_vir_b = lo.Boys(mol_u, mf_u.mo_coeff[1][:, vir_mask_b])
    
    # Pass tuple of localizers
    occ_calc_u = LMOActiveSpace(mf_u, frag_inds, 'occ', (loc_occ_a, loc_occ_b), cutoff=0.01)
    vir_calc_u = LMOActiveSpace(mf_u, frag_inds, 'vir', (loc_vir_a, loc_vir_b), cutoff=0.01)
    
    embed_u = HFEmbedding(occ_calc_u, vir_calc_u)
    moE_u, moC_u, indx_frz_u = embed_u.calc_mo()
    
    print(f"Frozen Alpha: {len(indx_frz_u[0])}, Frozen Beta: {len(indx_frz_u[1])}")
    
    mycc_u = cc.UCCSD(mf_u)
    mycc_u.mo_coeff = moC_u
    mycc_u.frozen = indx_frz_u
    mycc_u.run()
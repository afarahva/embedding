#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dmet.py

Density Matrix Embedding Theory / Automated Valence Active Space code

Same as regional-embedding, but the size of the active virtal space is 
constrained to be equal to the size of the active occupied space.

author: Ardavan Farahvash, github.com/afarahva
"""    

import numpy as np
from pyscf_embedding.lib import rWVFEmbedding
from pyscf_embedding.local.regional import rRegionalActiveSpace

# AVAS method
class rAVAS(rWVFEmbedding):
    
    def __init__(self, mf, frag_inds, frag_inds_type='atom', basis='minao', cutoff_type="overlap", cutoff=0.1, orth=False, frozen_core=False):
        
        self.occ_calc = rRegionalActiveSpace(mf, frag_inds, 'occupied', 
        frag_inds_type=frag_inds_type, basis=basis, cutoff_type=cutoff_type,
        cutoff=cutoff, orth=orth, frozen_core=frozen_core)
        
        self.vir_calc = rRegionalActiveSpace(mf, frag_inds, 'virtual', 
        frag_inds_type=frag_inds_type, basis=basis, cutoff=cutoff, 
        orth=orth, frozen_core=frozen_core)
        
    def calc_mo(self):
        
        # occupied space follows same rules as regional embedding
        self.moC_occ,self.moE_occ,self.mask_occ_act = self.occ_calc.calc_mo()
        
        # virtual active space must have the same number of orbitals as occupied
        self.vir_calc.cutoff_type = 'norb'
        self.vir_calc.cutoff = int(self.occ_calc.Norb_act)
        
        moE_embed, moC_embed, indx_frz = super().calc_mo()
        return moE_embed, moC_embed, indx_frz
        
rDMET = rAVAS

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
    
    # calculate localized orbital energies/coefficients
    frag_inds=[0,1]
    basis_occ='minao'
    basis_vir=mol.basis
    cutoff_occ=0.0
    cutoff_vir=0.1
    embed = rDMET(mf, frag_inds, 'atom', basis='minao', cutoff=0.1)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(len(indx_frz))

    # embedded
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
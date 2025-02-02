#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmboys.py

Pipek-Mezey / Boys MO localization

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from pyscf_embedding.lib import rUnitaryActiveSpace, rWVFEmbedding
from pyscf import lo

# regional embedding active space projector (either occupied or virtual)
class PMBoysActiveSpace(rUnitaryActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, localizer, 
                 frag_inds_type='atom', cutoff=0.1, pop_method='meta-lowdin'):
        
        super().__init__(mf,mo_occ_type)

        self.localizer = localizer
        self.cutoff = cutoff
        self.pop_method = pop_method
        
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
    
    def unitary_mo_to_lmo(self,mo_coeff,lmo_coeff):
        U = mo_coeff.T @ self.mf.get_ovlp() @ lmo_coeff
        return U
    
    def calc_projection(self):
        
        # calculate localized MOs
        self.lmo_coeff = self.localizer.kernel()
        self.frag_pop = self.population(self.lmo_coeff, method=self.pop_method)
        mask_act = self.frag_pop > self.cutoff
        mask_frz = ~mask_act
        
        # Unitary transformation matrix
        self.u = self.unitary_mo_to_lmo(self.moC, self.lmo_coeff)
        
        # Unitary projection operators
        self.P_act = self.u[:,mask_act]
        self.P_frz = self.u[:,mask_frz]
        self.Norb_act = np.sum(mask_act)


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
    
    mol = pyscf.M(atom=coords,basis='ccpvdz',verbose=3)
    mf = mol.RHF().run()
    
    ##### boys
    frag_inds=[0,1]
    loc1 = lo.Boys(mol, mf.mo_coeff[:,mf.mo_occ > 0])
    loc1.max_stepsize=0.005
    loc1.init_guess='cholesky'
    occ_calc = PMBoysActiveSpace(mf, frag_inds, 'occ', loc1, cutoff=0.01)
    
    loc2 = lo.Boys(mol, mf.mo_coeff[:,mf.mo_occ == 0])
    loc2.max_stepsize=0.005
    loc2.init_guess='cholesky'
    vir_calc = PMBoysActiveSpace(mf, frag_inds, 'vir', loc2, cutoff=0.01)
    
    embed = rWVFEmbedding(occ_calc,vir_calc)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(len(indx_frz))
    
    # embedded
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
    
    cubegen.orbital(mol,"./boys_homo.cube", moC_new[:,embed.mask_occ_act][:,-1])
    cubegen.orbital(mol,"./boys_lumo.cube", moC_new[:,embed.mask_vir_act][:,0])
    
    #%%
    ##### PM
    frag_inds=[0,1]
    loc1 = lo.PipekMezey(mol, mf.mo_coeff[:,mf.mo_occ > 0])
    loc1.max_stepsize=0.005
    loc1.init_guess='cholesky'
    occ_calc = PMBoysActiveSpace(mf, frag_inds, 'occ', loc1, cutoff=0.01)
    
    loc2 = lo.PipekMezey(mol, mf.mo_coeff[:,mf.mo_occ == 0])
    loc2.max_stepsize=0.005
    loc2.init_guess='cholesky'
    vir_calc = PMBoysActiveSpace(mf, frag_inds, 'vir', loc2, cutoff=0.01)
    
    embed = rWVFEmbedding(occ_calc,vir_calc)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(len(indx_frz))
    
    # embedded
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
    
    cubegen.orbital(mol,"./pm_homo.cube", moC_new[:,embed.mask_occ_act][:,-1])
    cubegen.orbital(mol,"./pm_lumo.cube", moC_new[:,embed.mask_vir_act][:,0])

    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lmo.py

Pipek-Mezey / Boys MO localization

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from pyscf_embedding.utils import ActiveSpace, HFEmbedding
from pyscf import lo

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
            
            'norb' assigns active MOs as those with the higest overlap with 
            the fragment until the cutoff. 
            
        pop_method : String
            Pop method for selecting MOs. Default is meta-lowdin.
        """
        super().__init__(mf, mo_coeff=mo_occ_type, frozen_core=frozen_core)

        self.localizer = localizer
        self.cutoff = cutoff
        self.cutoff_type=cutoff_type
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
    
    def calc_projection(self,**kwargs):
        
        # calculate localized MOs
        self.lmo_coeff = self.localizer.kernel()
        self.frag_pop = self.population(self.lmo_coeff, method=self.pop_method)
        
        if self.cutoff_type.lower() in ['overlap','pop','population']:
            mask_act = self.frag_pop > self.cutoff
            mask_frz = ~mask_act
            
        elif self.cutoff_type.lower() in ['spade', 'auto']:
            if type(self.cutoff)!=int:
                raise ValueError("For SPADE, cutoff value must be an int representing the number of additinoal orbitals to include form the inflection point of the population curve")
            
            s = np.sort(self.frag_pop)
            ds = s[1:] - s[0:-1]
            indx_max = np.argmax(ds)-self.cutoff
            
            mask_act = np.zeros(len(s), dtype=bool)
            indx_sort = np.argsort( self.frag_pop )
            mask_act[indx_sort[indx_max+1:]] = True
            mask_frz = ~mask_act

        elif self.cutoff_type.lower() in ['norb','norb_act']:
            indx_sort = np.flip(np.argsort( self.frag_pop ))
            mask_act = np.zeros(len( self.frag_pop ), dtype=bool)
            mask_act[indx_sort[0:self.cutoff]] = True
            mask_frz = ~mask_act
            
        else:
            raise ValueError("Incorrect cutoff type. Must be one of 'overlap' or 'norb'" )
        
        # Unitary matrix
        self.u = self.unitary_mo_to_lmo(self.moC, self.lmo_coeff)
        
        # Unitary projectors
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
    
    #%%
    ##### Boys
    frag_inds=[0,1]
    loc1 = lo.Boys(mol, mf.mo_coeff[:,mf.mo_occ > 0])
    loc1.max_stepsize=0.005
    loc1.init_guess='cholesky'
    occ_calc = LMOActiveSpace(mf, frag_inds, 'occ', loc1, cutoff=0.01)
    
    loc2 = lo.Boys(mol, mf.mo_coeff[:,mf.mo_occ == 0])
    loc2.max_stepsize=0.005
    loc2.init_guess='cholesky'
    vir_calc = LMOActiveSpace(mf, frag_inds, 'vir', loc2, cutoff=0.01)
    
    embed = HFEmbedding(occ_calc,vir_calc)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(len(indx_frz))
    
    # embedded
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
    
    #%%
    ##### PM
    frag_inds=[0,1]
    loc1 = lo.PipekMezey(mol, mf.mo_coeff[:,mf.mo_occ > 0])
    loc1.max_stepsize=0.005
    loc1.init_guess='cholesky'
    occ_calc = LMOActiveSpace(mf, frag_inds, 'occ', loc1, cutoff_type='norb', cutoff=10)
    
    loc2 = lo.PipekMezey(mol, mf.mo_coeff[:,mf.mo_occ == 0])
    loc2.max_stepsize=0.005
    loc2.init_guess='cholesky'
    vir_calc = LMOActiveSpace(mf, frag_inds, 'vir', loc2, cutoff_type='norb', cutoff=20)
    
    embed = HFEmbedding(occ_calc,vir_calc)
    moE_new, moC_new, indx_frz = embed.calc_mo()
    print(len(indx_frz))
    
    # embedded
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()

    
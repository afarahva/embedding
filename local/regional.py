#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regional.py

Regional Embedding and Regional Active Space / SPADE code

author: Ardavan Farahvash, github.com/afarahva
"""    

import numpy as np
from pyscf_embedding.lib import rUnitaryActiveSpace, rWVFEmbedding, FC_AO_Ints


def FragmentPops(mol, mo_coeff, frag_inds, basis_frag, frag_inds_type="atom",  orth=None):
    """
    Calculates populations of canonical orbitals of a composite system on a 
    fragment.
    """
    AO_ints = FC_AO_Ints(mol)
    ovlp_ff, ovlp_fc = FC_AO_Ints.calc_ovlp(frag_inds, frag_inds_type, basis_frag=basis_frag, orth=orth)
    pop = AO_ints.population(ovlp_ff, ovlp_fc, mo_coeff)
    
    return pop

# regional embedding active space projector (either occupied or virtual)
# this version works for any spin-restricted system, but only single k-point
# periodic calculations are supported. 
class rRegionalActiveSpace(rUnitaryActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, 
                 frag_inds_type="atom", basis='minao', orth=False,
                 cutoff_type="overlap", cutoff=0.1, **kwargs):
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
            
            'spade' assigns active MOs based on the spade approach, using the
            inflection point of the occupancy curve.
            
            'pct_occ' assigns active MOs as those with the higest overlap with 
            the fragment until a percentage of the total overlap is reached.
            
            'norb' assigns active MOs as those with the higest overlap with 
            the fragment until the cutoff. 
            
        orth : Bool
            Whether to orthogonalization fragment orbitals. Default is False.
        """
        super().__init__(mf,mo_occ_type,**kwargs)
        self.cutoff=cutoff
        self.cutoff_type = cutoff_type
        self.basis=basis
        self.fc_ints = FC_AO_Ints(mf.mol, frag_inds, frag_inds_type=frag_inds_type, basis_frag=basis, orth=orth)
        
    def calc_projection(self,debug=False):
        
        # compute projection operator 
        if self.basis=='iao':
            ovlp_ff, ovlp_fc = self.fc_ints.calc_ao_ovlp(moC_occ=self.moC)
        else:
            ovlp_ff, ovlp_fc = self.fc_ints.calc_ao_ovlp()

        ovlp_f_mo = ovlp_fc @ self.moC
        
        P_proj = ovlp_f_mo.conj().T @ np.linalg.inv(ovlp_ff) @ ovlp_f_mo
        
        # diagonalize 
        s,u = np.linalg.eigh(P_proj)
        s[s < 0] = 0
        
        # select indices of active orbitals
        if self.cutoff_type.lower() in ['overlap','pop','population']:
            mask_act = s >= self.cutoff
            
        elif self.cutoff_type.lower() in ['spade']:
            ds = s[1:] - s[0:-1]
            indx_max = np.argmax(ds)
            mask_act = np.zeros(len(s), dtype=bool)
            mask_act[indx_max+1:] = True
            
        elif self.cutoff_type.lower() in ['pct_occ','occ']:
            cumsum = np.cumsum(s[::-1]/np.sum(s))[::-1]
            mask_act = cumsum < self.cutoff
            
        elif self.cutoff_type.lower() in ['norb','norb_act']:
            assert type(self.cutoff)==int
            mask_act = np.zeros(len(s), dtype=bool)
            mask_act[np.argsort(s)[0:self.cutoff]] = True 

        else:
            raise ValueError("Incorrect cutoff type. Must be one of 'overlap', 'pct_occ' or 'Norb'" )
        
        # construct unitary projector 
        self.P_act = u[:,mask_act]
        self.P_frz = u[:,~mask_act]
        self.Norb_act = np.sum(mask_act)
        
        # set more object attributes if debugger is called
        if debug:
            self.P_proj = P_proj
            self.s_proj = s
            self.ds_proj = ds
            self.mask_act = mask_act
        
        return self.P_act, self.P_frz

rSPADE = rRegionalActiveSpace

# standard regional embedding
class rRegionalEmbedding(rWVFEmbedding):
    
    def __init__(self, mf, frag_inds, frag_inds_type='atom', basis_occ='minao', 
                 basis_vir=None, cutoff_occ=0.1, cutoff_vir=0.1, 
                 cutoff_type='overlap', orth=None, frozen_core=False):
        
        self.occ_calc = rRegionalActiveSpace(mf, frag_inds, 'occupied', 
            frag_inds_type=frag_inds_type, basis=basis_occ, cutoff=cutoff_occ, 
            cutoff_type=cutoff_type, orth=orth,frozen_core=frozen_core)
        
        self.vir_calc = rRegionalActiveSpace(mf, frag_inds, 'virtual', 
            frag_inds_type=frag_inds_type, basis=basis_vir, cutoff=cutoff_vir, 
            cutoff_type=cutoff_type, orth=orth,frozen_core=False)
         
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
    
    mol = pyscf.M(atom=coords,basis='ccpvdz',verbose=4)
    mol.build()
    mf = mol.RHF().run()
    
    # calculate localized orbital energies/coefficients
    frag_inds=[0,1]
    basis_occ='minao'
    basis_vir=mol.basis
    cutoff_occ=0.1
    cutoff_vir=0.1
    
    # traditional regional embedding
    re = rRegionalEmbedding(mf, frag_inds, 'atom', basis_occ, basis_vir, cutoff_occ, cutoff_vir, orth=False, frozen_core=True)
    moE_re, moC_new, indx_frz_re = re.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz_re
    mycc.run()
    print(mycc.e_corr)
    
    # spade embedding
    occ_calc = rRegionalActiveSpace(mf, frag_inds, 'occ', basis='minao', cutoff_type='spade', frozen_core=True)
    vir_calc = rRegionalActiveSpace(mf, frag_inds, 'vir', basis=mol.basis, cutoff_type='spade', frozen_core=False)
    embed = rWVFEmbedding(occ_calc, vir_calc)
    moE_spade, moC_new, indx_frz_spade = embed.calc_mo()
    
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz_spade
    mycc.run()
    print(mycc.e_corr)
    
    #%%
    # compare regional embedding versus spade
    occ_calc.calc_projection(debug=True)
    s = occ_calc.s_proj
    ds = occ_calc.ds_proj
    mask_act = occ_calc.mask_act

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(s)
    plt.plot(ds)
    plt.plot(mask_act)
    
    vir_calc.calc_projection(debug=True)
    s = vir_calc.s_proj
    ds = vir_calc.ds_proj
    mask_act = vir_calc.mask_act

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(s)
    plt.plot(ds)
    plt.plot(mask_act)


    

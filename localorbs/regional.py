#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regional.py

Regional Embedding and Regional Active Space / SPADE code

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from lib import rUnitaryActiveSpace, rActiveSpaceEmbedding, FC_AO_Ints


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
class rRegionalActiveSpace(rUnitaryActiveSpace):
    
    def __init__(self, mf, frag_inds, mo_occ_type, 
                 frag_inds_type="atom", basis='minao', cutoff=0.1, 
                 cutoff_type='absolute', orth=None):
        """
        
        Parameters
        ----------
        mol : PySCF Molecule object
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
        cutoff : Float
            Eigenvalue cutoff for active orbitals. Default: 0.1
        orth : Method to orthogonalize atomic orbitals. 
            Options are lowdin, meta-lowdin, nao, or schmidt. Default: None
        """
        super().__init__(mf,mo_occ_type)
        self.cutoff=cutoff
        self.cutoff_type = cutoff_type
        self.fc_ints = FC_AO_Ints(mf.mol, frag_inds, frag_inds_type=frag_inds_type, basis_frag=basis, orth=orth)
        
    def calc_projection(self):
        ovlp_ff, ovlp_fc = self.fc_ints.calc_ao_ovlp()
        ovlp_f_mo = ovlp_fc @ self.moC
        
        #
        P_proj = ovlp_f_mo.conj().T @ np.linalg.inv(ovlp_ff) @ ovlp_f_mo
        self.P_proj = P_proj
        s,u = np.linalg.eigh(P_proj)
        s[s < 0] = 0
        
        # indices of active orbitals
        if self.cutoff_type.lower() in ['pct_occ','occ']:
            cumsum = np.cumsum(s[::-1]/np.sum(s))[::-1]
            mask_act = cumsum < self.cutoff
            mask_frz = ~mask_act
        else:
            mask_act = s >= self.cutoff
            mask_frz = ~mask_act

        self.P_act = u[:,mask_act]
        self.P_frz = u[:,mask_frz]
        self.Norb_act = np.sum(mask_act)
        
        return self.P_act, self.P_frz
   
# standard regional embedding
class rRegionalEmbedding(rActiveSpaceEmbedding):
    
    def __init__(self, mf, frag_inds, frag_inds_type='atom', basis_occ='minao', 
                 basis_vir=None, cutoff_occ=0.1, cutoff_vir=0.1, 
                 cutoff_type='absolute', orth=None):
        
        self.occ_calc = rRegionalActiveSpace(mf, frag_inds, 'occupied', 
            frag_inds_type=frag_inds_type, basis=basis_occ, cutoff=cutoff_occ, 
            cutoff_type=cutoff_type, orth=orth)
        
        self.vir_calc = rRegionalActiveSpace(mf, frag_inds, 'virtual', 
            frag_inds_type=frag_inds_type, basis=basis_vir, cutoff=cutoff_vir, 
            cutoff_type=cutoff_type, orth=orth)
         
    def kernel(self):
        super().kernel()
        return self.moE_proj, self.moC_proj, self.indx_frz
        
# DMET/AVAS embedding
class rDMET(rActiveSpaceEmbedding):
    
    def __init__(self, mf, frag_inds, frag_inds_type='atom', basis='minao', cutoff=0.1, orth=None):
        self.occ_calc = rRegionalActiveSpace(mf, frag_inds, 'occupied', frag_inds_type=frag_inds_type, basis=basis, cutoff=cutoff, orth=orth)
        self.vir_calc = rRegionalActiveSpace(mf, frag_inds, 'virtual', frag_inds_type=frag_inds_type, basis=basis, cutoff=cutoff, orth=orth)
        
    def kernel(self):
        # occupied space follows same rules as regional embedding
        self.moC_occ,self.moE_occ,self.mask_occ_act = self.occ_calc.calc_mo()
        
        # virtual space should be equal size to occupied space
        ovlp_ff, ovlp_fc = self.vir_calc.fc_ints.calc_ao_ovlp()
        ovlp_f_mo = ovlp_fc @ self.vir_calc.moC
        P_proj = ovlp_f_mo.T @ np.linalg.inv(ovlp_ff) @ ovlp_f_mo
        s, P = np.linalg.eigh(P_proj)
        s[s < 0] = 0
        
        isort = np.flip(np.argsort(s))
        iact = isort[0:self.occ_calc.Norb_act]
        ifrz = isort[self.occ_calc.Norb_act:]
        
        P_act = P[:,iact]
        P_frz = P[:,ifrz]
        
        moE_vir_act,moC_vir_act = self.vir_calc.fock_rotate(self.vir_calc.moE, self.vir_calc.moC, P_act)
        moE_vir_frz,moC_vir_frz = self.vir_calc.fock_rotate(self.vir_calc.moE, self.vir_calc.moC, P_frz)
        
        self.moC_vir = np.hstack([moC_vir_frz,moC_vir_frz])
        self.moE_vir = np.hstack([moE_vir_act,moE_vir_frz])
        self.mask_vir_act = np.arange(len(self.moE_vir)) < self.occ_calc.Norb_act
            
        # reorder by energy (makes analysis easier)
        order = np.argsort(self.moE_vir)
        self.moE_vir = self.moE_vir[order]
        self.moC_vir = self.moC_vir[:,order]
        self.mask_vir_act = self.mask_vir_act[order]
        
        # concatenate virtual and occupied orbitals and output
        self.moE_proj = np.hstack([self.moE_occ,self.moE_vir])
        self.moC_proj = np.hstack([self.moC_occ,self.moC_vir])
        self.mask_act = np.hstack([self.mask_occ_act,self.mask_vir_act])
        self.mask_frz = ~self.mask_act
        
        self.indx_act = np.argwhere(self.mask_act)[:,0]
        self.indx_frz = np.argwhere(~self.mask_act)[:,0]
            
        return self.moE_new, self.moC_new, self.indx_frz
        


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
    #mf = mol.RKS(xc='b3lyp').run()
    mf = mol.RHF().run()
    # calculate localized orbital energies/coefficients
    frag_inds=[0,1]
    basis_occ='minao'
    basis_vir=mol.basis
    cutoff_occ=0.0
    cutoff_vir=0.1
    re = rRegionalEmbedding(mf, frag_inds, 'atom', basis_occ, basis_vir, cutoff_occ, cutoff_vir)
    moE_new, moC_new, indx_frz = re.kernel()
    print(len(indx_frz))
    # CCSD(T) energies
    
    # full CCSD
    # mycc = pyscf.cc.CCSD(mf)
    # mycc.run()

    # embedded
    mycc = pyscf.cc.CCSD(mf)
    mycc.mo_coeff = moC_new
    mycc.frozen = indx_frz
    mycc.run()
    #%%
    # mo1 = re.occ_calc.moC_occ @ re.occ_calc.P_act 
    # mo2 = re.occ_calc.moC_occ @ re.occ_calc.P_frz
    # mo3 = re.vir_calc.moC_vir @ re.vir_calc.P_act 
    # mo4 = re.vir_calc.moC_vir @ re.vir_calc.P_frz
    
    # mo_test = np.hstack([mo1,mo2,mo3,mo4])
    # mask_frz = np.hstack( [[False]*mo1.shape[1],[True]*mo2.shape[1],[False]*mo3.shape[1],[True]*mo4.shape[1]] )
    # indx_frz = np.argwhere(mask_frz)[:,0]
    
    # mycc = pyscf.cc.CCSD(mf)
    # mycc.mo_coeff = mo_test
    # mycc.frozen = indx_frz
    # mycc.run()

    # check localization of orbitals
    # cubegen.orbital(mol,"./lo_homo.cube", re.moC_occ[:,re.mask_occ_act][:,-1])
    # cubegen.orbital(mol,"./lo_homo.cube", re.moC_vir[:,re.mask_vir_act][:,0])

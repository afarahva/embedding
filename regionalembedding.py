#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regionalembedding.py

Regional-Embedding code
"""

import numpy as np
import pyscf
from lib import FockProjection, FC_AO_Ints

def FragmentPops(mol, mo_coeff, frag_inds, basis_frag, frag_inds_type="atom",  orth=None):
    """
    Calculates populations of canonical orbitals of a composite system on a 
    fragment.
    """
    AO_ints = FC_AO_Ints(mol)
    ovlp_ff, ovlp_fc = FC_AO_Ints.calc_ovlp(frag_inds, frag_inds_type, basis_frag=basis_frag, orth=orth)
    pop = AO_ints.population(ovlp_ff, ovlp_fc, mo_coeff)
    
    return pop

class rRegionalEmbedding(FockProjection,FC_AO_Ints):
    """
    Class for One-Shot Regional Embedding for non-periodic systems with RHF refernce. 
    
    Calculates projection operators for occupied/virtual canonical orbitals onto a minimal
    basis on a small fragment. 
    
    Diagonalizes projection operators to find unitary operators which seperate MOs into those
    with/withot ovelap with fragment. 
    
    Uses those unitary operators to select valence active space.
    """
    
    def __init__(self, mol, mf, frag_inds, 
                 frag_inds_type="atom", frag_inds_vir=None,
                 basis_occ='minao', basis_vir=None, 
                 cutoff_occ=0.1, cutoff_vir=None, balanced=False, orth=None):
        """
        
        Parameters
        ----------
        mol : PySCF Molecule object
        mf : PySCF Mean Field object
        frag_inds : Iterable
            Indices of fragment atoms.
            
        OPTIONAL: 
        ----------
        frag_inds_type : String.
            Specify 'orbital' if supplying a list of orbital indices in frag_inds instead of atom indices
        frag_inds_vir : Iterable
            Specify different fragment indices for virtual orbitals. Uses frag_inds if not specified. 
        basis : String.
            Fragment basis set for occupied orbitals. Uses minao if not spcified
        basis_occ : String.
            Fragment basis set for virtual orbitals. Uses mol.basis if not spcified
        cutoff_occ : Float
            Eigenvalue cutoff for active occupied orbitals. Default is 0.1
        cutoff_vir : Float
            Eigenvalue cutoff for active virtual orbitals. Default is to use same as cutoff_occ
        balanced : Bool
            Whether to use the enforce number of active occupied/virtual orbitals (default is False)
        orth : Method to orthogonalize atomic orbitals. 
            Options are lowdin, meta-lowdin, nao, or schmidt. The default is None.
        """
        FockProjection.__init__(self, mf)

        self.mol = mol
        self.frag_inds_type = frag_inds_type
        self.frag_inds_occ = frag_inds
        if frag_inds_vir is None:
            frag_inds_vir = frag_inds
        self.frag_inds_vir = frag_inds_vir
        
        if cutoff_vir is None:
            cutoff_vir = cutoff_occ
        
        if basis_vir is None:
            basis_vir=self.mol.basis
            
        self.basis_occ = basis_occ
        self.basis_vir = basis_vir
        self.cutoff_occ = cutoff_occ
        self.cutoff_vir = cutoff_vir
        self.balanced=balanced
        self.orth=orth
    
    def kernel(self):
        """
        Run calculation

        Returns
        -------
        moC_proj : Numpy Array
            molecular orbital coefficients of active/frozen orbtials.
            columns are ordered as [nocc_frz,nocc_act,nvir_act,nvir_frz]
        moE_proj : Numpy Array
            molecular orbital energies of active/frozen orbtials.
            ordered as [nocc_frz,nocc_act,nvir_act,nvir_frz]
        indx_act : Numpy Array
            Indices of active orbtials.
        indx_frz : Numpy Array
            Indices of frozen orbtials.
        """
        
        print("Calculating fragment projected orbitals")
        # occupied
        ovlp_ff, ovlp_fc = self.calc_ovlp(self.frag_inds_occ, self.frag_inds_type, 
                                          basis_frag=self.basis_occ, orth=self.orth)
        
        _, self.U_occ, self.indx_occ_act = self.calc_projection(ovlp_ff, ovlp_fc, self.moC_occ, self.cutoff_occ)
        
        # virtual
        ovlp_ff, ovlp_fc = self.calc_ovlp(self.frag_inds_vir, self.frag_inds_type, 
                                          basis_frag=self.basis_vir, orth=self.orth)

        _, self.U_vir, self.indx_vir_act = self.calc_projection(ovlp_ff, ovlp_fc, self.moC_vir, self.cutoff_vir)
        if self.balanced:
            self.indx_vir_act = np.arange(self.Nvir-len(self.indx_occ_act),self.Nvir)            

        #
        self.indx_occ_frz = np.delete(np.arange(self.Nocc),self.indx_occ_act)
        self.indx_vir_frz = np.delete(np.arange(self.Nvir),self.indx_vir_act)
        
        print("Constructing new active-space mo-coefficients, energies")
        self.calc_mo(self.U_occ[:,self.indx_occ_act],self.U_occ[:,self.indx_occ_frz],
                     self.U_vir[:,self.indx_vir_act],self.U_vir[:,self.indx_vir_frz])
        
        return self.moC_proj, self.moE_proj, self.indx_act, self.indx_frz
        
        
    def calc_projection(self, ovlp_ff, ovlp_fc, mo_coeff, cutoff):
        """
        Calculate projection operators
        """

        # overlap between canonical MOs and fragment orbitals
        ovlp_f_mo = ovlp_fc @ mo_coeff

        # Projection operator and unitary rotation into projected space
        P = ovlp_f_mo.T @ np.linalg.inv(ovlp_ff) @ ovlp_f_mo
        e, u = np.linalg.eigh(P)
        e[e < 0] = 0
        
        # indices of active orbitals
        indx_act = np.argwhere(e >= cutoff)[:,0]
        
        return P, u, indx_act
  
class rSCRegionalEmbedding:
    
    def __init__(self,highlevelsolver,lowlevelsolvr,re):
        pass
    # TO-DO
    
# class krRegionalEmbedding

#%%
if __name__ == '__main__':
    from pyscf.tools import cubegen
    from ase.io import Trajectory, read, write, xyz, lammpsdata, lammpsrun
    from ase.visualize import view
    import matplotlib.pyplot as plt
    from pyscf.scf.hf import mulliken_pop
    
    ##########  NICE PLOTS  ###########
    plt.rcParams["figure.figsize"] = (8,6)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams['axes.labelsize'] = 32
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] =  28
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.labelsize'] = 28
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.right'] = True
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['figure.titlesize'] = 32
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \usepackage{braket}"
    ###################################


    xyz_file = "./tests/benzenethiol.xyz"
    
    mol = pyscf.M(atom=xyz_file,basis='6-31G')
    mf = mol.RHF().run()
    mf_e = mf.e_tot

    # mycc = pyscf.cc.CCSD(mf)
    # mycc.kernel()
    # ecorr_full = mycc.e_corr * 27.211399
    
    # Plot Local Orbitals
    re = rRegionalEmbedding(mol, mf, [0], basis_occ='iao', cutoff_occ=1e-5, cutoff_vir=1e-5)
    moC_proj, moE_proj, indx_act, indx_frz = re.kernel()
        
    mycc = pyscf.cc.CCSD(mf, mo_coeff=moC_proj, frozen=indx_frz)
    mycc.kernel()
    e_corr_re = mycc.e_corr * 27.211399
    
    cubegen.orbital(mol,"./tests/lo_homo.cube", re.moC_occ_act[:,-1])
    cubegen.orbital(mol,"./tests/lo_lumo.cube", re.moC_vir_act[:,-1])
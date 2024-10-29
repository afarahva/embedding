import numpy as np
import pyscf
from pyscf import lo


class ValenceActiveSpace:
    """
    Class for freezing non-canonical orbitals by seperate rotations of 
    occupied/virtual orbitals. 
    
    Canonical orbitals are eigevectors of the Fock matrix. 
    F psi = E S psi
    Given a set of frozen non-canonical orbitals {\phi}, we can re-diagonalize 
    the Fock operator in the subspace of non-frozen orbitals. 
    
    This defines a new set of active MOs spanning the non-frozen subspace, 
    which can subsequently be correlated with post-HF methods. 
    """
    
    def __init__(self,mf):
        """
        Initial Active Space Selector 
        
        Parameters
        ----------
        mf : PySCF Mean Field Object

        Returns
        -------
        None.

        """
        
        mo_coeff, mo_energy, mo_occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
        
        self.mask_occ = mo_occ > 1e-10
        self.mask_vir = (self.mask_occ==False)
        
        self.moE_occ = mo_energy[self.mask_occ]
        self.moE_vir = mo_energy[self.mask_vir]
        self.moC_occ = mo_coeff[:,self.mask_occ]
        self.moC_vir = mo_coeff[:,self.mask_vir]
        
        self.Nmo, self.Nocc, self.Nvir = len(mo_energy), len(self.moE_occ), len(self.moE_vir)
        
    def project(self, P, fock):
        """
        Project Fock Matrix into a basis given by the columns of U.
        
        Returns MO coefficients in projected/rotated basis

        Parameters
        ----------
        P : Numpy Array
            Projection matrix in MO basis, columns should be orthonormal. 
        indx_act : 
            Indices of active orbitals
            
        Returns
        -------
        moE_U : projected mo energy
        evec_F_mo : eigenvectors of F in terms of canonical MOs
        """
        
        # rotate fock matrix into basis of U
        F_P = P.conj().T @ fock @ P
        
        # diagonalize Fock operator in this basis
        moE, evec_F = np.linalg.eigh(F_P)
        
        # express eigenvectors of F in canonical orbitals
        evec_F_mo = P @ evec_F
        
        return moE, evec_F_mo
        
        
    def calc_mo(self, U_occ, U_vir, indx_occ_act, indx_occ_frz, indx_vir_act, indx_vir_frz):
        """
        Calculated new MO energies and coefficients from rotating 
        occupied/virtual orbitals by P_occ and P_vir and then only holding 
        active those orbitals in indx_occ_act and indx_vir_act. 
        
        If a transformation only on occupied or virtual is desired specify none
        as argument. 
        
        Parameters
        ----------
        U_occ : Numpy Array.
            Unitary matrix for occupied MOs.
        U_vir : Numpy Array.
            Unitary matrix for virtual MOs.
        indx_occ_act : Numpy Array.
            Indices of occupied orbitals to keep active
        indx_vir_act : Numpy Array.
            Indices of virtual orbitals to keep active

        Returns
        -------
        mo_energy_act : active-space MO energies
        mo_coeff_act : active-space MO coefficients
        """

            
        N_occ_act = len(indx_occ_act)
        N_occ_frz = len(indx_occ_frz)
        N_vir_act = len(indx_vir_act)
        N_vir_frz = len(indx_vir_frz)
        
        # project occupied orbitals
        if U_occ is not None:
            # set active orbitals
            self.moE_occ_act, evec_F_mo = self.project(U_occ[:,indx_occ_act], np.diag(self.moE_occ))
            self.moC_occ_act = self.moC_occ @ evec_F_mo
            
            # set frozen orbitals
            self.moE_occ_frz, evec_F_mo = self.project(U_occ[:,indx_occ_frz], np.diag(self.moE_occ))
            self.moC_occ_frz = self.moC_occ @ evec_F_mo
        else:
            self.moE_occ_act, self.moC_occ_act = self.moE_occ, self.moC_occ
            self.moE_occ_frz, self.moE_occ_frz = [],[]
            

        # project virtual orbitals
        if U_vir is not None and indx_vir_act is not None:
            # set active orbitals
            self.moE_vir_act, evec_F_mo = self.project(U_vir[:,indx_vir_act], np.diag(self.moE_vir))
            self.moC_vir_act = self.moC_vir @ evec_F_mo
            
            # set frozen orbitals, note frozen VIRTUAL orbital energies/coefficients do not affect observables
            self.moE_vir_frz, evec_F_mo = self.project(U_vir[:,indx_vir_frz], np.diag(self.moE_vir))
            self.moC_vir_frz = self.moC_vir @ evec_F_mo
            # self.moE_vir_frz,self.moC_vir_frz = np.zeros(N_vir_frz),np.zeros(self.Nmo, N_vir_frz)
            
        else:
            self.moE_vir_act, self.mo_vir_act = self.moE_vir, self.moC_vir
            self.moE_vir_act, self.moC_vir_act = [], []

        # concatenate orbitals
        self.moE_proj = np.hstack([self.moE_occ_frz,self.moE_occ_act,self.moE_vir_act,self.moE_vir_frz])
        self.moC_proj = np.hstack([self.moC_occ_frz,self.moC_occ_act,self.moC_vir_act,self.moC_vir_frz])
        
        # indices of frozen orbitals
        self.indx_act = np.hstack([np.arange(N_occ_frz,self.Nocc),np.arange(self.Nocc,self.Nocc+N_vir_act)])
        self.indx_frz = np.hstack([np.arange(0,N_occ_frz),np.arange(self.Nocc+N_vir_act,self.Nmo)])


class rDMET(ValenceActiveSpace):
    """
    Class for One-Shot Regional Embedding for non-periodic systems with RHF refernce. 
    
    Calculates projection operators for occupied/virtual canonical orbitals onto a minimal
    basis on a small fragment. 
    
    Diagonalizes projection operators to find unitary operators which seperate MOs into those
    with/withot ovelap with fragment. 
    
    Uses those unitary operators to select valence active space.
    """
    
    def __init__(self, mol, mf, indx_frag, basis_occ='minao', basis_vir=None, 
                 cutoff_occ=0.1, cutoff_vir=None, balanced=False, orth=None):
        """
        
        Parameters
        ----------
        mol : PySCF Molecule object
        mf : PySCF Mean Field object
        indx_frag : Numpy Array
            Indices of fragment orbitals.
        basis_occ : String.
            Fragment basis set for occupied orbitals.
        basis_occ : String.
            Fragment basis set for virtual orbitals.
        cutoff_occ : Float
            Eigenvalue cutoff for active occupied orbitals. Default is 0.1
        cutoff_vir : Float
            Eigenvalue cutoff for active virtual orbitals. Default is to use same as cutoff_occ
        balanced : Bool
            Whether to use the enforce number of active occupied/virtual orbitals (default is False)
        ortho : Method to orthogonalize atomic orbitals. 
            options are mf, lowdin, meta-lowdin, iao, or None
            The default is None.
        """
        ValenceActiveSpace.__init__(self, mf)
        
        self.mol = mol
        self.indx_frag = indx_frag
        
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
    
    def kernel(self,fragocc_ao_inds=None,fragvir_ao_inds=None):
        """
        Run calculation

        Parameters
        ----------
        fragocc_ao_inds : TYPE, optional
            indices of fragment orbitals for occupied projection basis. The default is None.
        fragvir_ao_inds : TYPE, optional
            indices of fragment orbitals for virtual projection basis. The default is None.

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
        ovlp_ff, ovlp_fc = self.calc_ovlp(self.basis_occ, orth=self.orth, frag_ao_inds=fragocc_ao_inds)
        _, self.U_occ, self.indx_occ_act = self.calc_projection(ovlp_ff, ovlp_fc, self.moC_occ, self.cutoff_occ)
        
        # virtual
        ovlp_ff, ovlp_fc = self.calc_ovlp(self.basis_vir, orth=self.orth, frag_ao_inds=fragvir_ao_inds)
        _, self.U_vir, self.indx_vir_act = self.calc_projection(ovlp_ff, ovlp_fc, self.moC_vir, self.cutoff_vir)
        if self.balanced:
            self.indx_vir_act = np.arange(self.Nvir-len(self.indx_occ_act),self.Nvir)            



        self.indx_occ_frz = np.delete(np.arange(self.Nocc),self.indx_occ_act)
        self.indx_vir_frz = np.delete(np.arange(self.Nvir),self.indx_vir_act)
        
        print("Constructing new active-space mo-coefficients, energies")
        self.calc_mo(self.U_occ,self.U_vir,
                     self.indx_occ_act,self.indx_occ_frz,self.indx_vir_act,self.indx_vir_frz)
        
        return self.moC_proj, self.moE_proj, self.indx_act, self.indx_frz
        
    def calc_ovlp(self, basis, orth=None, frag_ao_inds=None):
        """
        calculate overlap matrices between fragment/composite AOs
        """
        
        # for a normal minimal basis
        if basis != 'iao':
            
            # molecule in minimal basis
            mol_proj = pyscf.M(atom=self.mol._atom, basis=basis, spin=0, unit='Bohr')
            
            # indices of fragment atomic orbitals
            if frag_ao_inds is None:
                frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                          mol_proj.aoslice_by_atom()[self.indx_frag]]).astype(int)
            
            # overlap  matrices 
            s_ff = mol_proj.intor_symmetric('int1e_ovlp')[np.ix_(frag_ao_inds,frag_ao_inds)]  # nfrag x nfrag
            s_fc = pyscf.gto.mole.intor_cross('int1e_ovlp', mol_proj, self.mol)[frag_ao_inds] # nfrag x ncomposite
            
        # for an IAO basis
        else:
            # molecule in minimal basis
            mol_proj = pyscf.M(atom=self.mol._atom, basis='minao', spin=0, unit='Bohr')
            mf_proj = mol_proj.RHF.run()
            
            # indices of fragment atomic orbitals
            if frag_ao_inds is None:
                frag_ao_inds = np.concatenate([range(p0,p1) for b0,b1,p0,p1 in
                                          mol_proj.aoslice_by_atom()[self.indx_frag]]).astype(int)
            
            # fragment intrinsic atomic orbitals in minao basis
            c_iao = lo.iao.iao(mol_proj, mf_proj.mo_coeff[:,mf_proj.mo_occ>0])[:,frag_ao_inds]
            
            # fragment iao overlap matrix
            s_ff = c_iao.T @ mol_proj.intor_symmetric('int1e_ovlp') @ c_iao
            
            # fragment iao | composite ao ovelap matrix
            s_fc = pyscf.gto.mole.intor_cross('int1e_ovlp', mol_proj, self.mol)
            print(c_iao.shape,s_fc.shape)
            s_fc = c_iao.T @ s_fc
            
        # orthogonalize
        if orth is not None:
            s_ff, s_fc = self.orthogonalize(basis, orth, s_ff, s_fc)
            
        return s_ff, s_fc
    
    def orthogonalize(self, basis, orth, s_ff, s_fc):
        """
        Orthogonalize AOs, returns overlap matrices for orthogonalized orbitals
        """
            
        try:
            atom = [ self.mol._atom[i] for i in self.indx_frag ]
            frag=pyscf.M(atom=atom, basis=basis, spin=0, unit='Bohr')
        except:
            atom = [ self.mol._atom[i] for i in self.indx_frag ]
            frag=pyscf.M(atom=atom, basis=basis, spin=1, unit='Bohr')
            
        # s_lf is overlap between orthogonalized fragment orbitals and fragment AOs
        if orth.lower() == 'lowdin':
            s_lf = lo.lowdin(s_ff)
            
        elif orth.lower() == 'meta-lowdin':
            s_lf = lo.orth_ao(frag,method='meta-lowdin')
            
        elif orth.lower() == 'nao':
            mf_frag = frag.RHF().run()
            s_lf = lo.orth_ao(frag,method='nao')
            
        elif orth.lower() == 'mf':
            mf_frag = frag.RHF().run()
            print(mf_frag)
            s_lf = mf_frag.mo_coeff
        
        else:
            raise ValueError("Incorrect argument for 'ortho'")
            
        # re-express fragment/composite ovelap matrix in terms of orthogonal orbitals
        s_ff = np.eye(s_lf.shape[0])
        s_fc = s_lf @ s_fc
        
        return s_ff, s_fc
        
        
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
    
    
# class krRegionalEmbedding
    
if __name__ == '__main__':
    from pyscf.tools import cubegen
    from ase.io import Trajectory, read, write, xyz, lammpsdata, lammpsrun
    from ase.visualize import view
    import matplotlib.pyplot as plt
    
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
    re = rRegionalEmbedding(mol, mf, [0], cutoff_occ=1e-5, cutoff_vir=1e-5)
    moC_proj, moE_proj, indx_act, indx_frz = re.kernel()
        
    mycc = pyscf.cc.CCSD(mf, mo_coeff=moC_proj, frozen=indx_frz)
    mycc.kernel()
    e_corr_re = mycc.e_corr * 27.211399
    
    cubegen.orbital(mol,"./tests/lo_homo.cube", re.moC_occ_act[:,-1])
    cubegen.orbital(mol,"./tests/lo_lumo.cube", re.moC_vir_act[:,-1])
#%%
    # Convergence w.r.t atoms in fragment (minao)
    ecorr_size=[]
    for i in range(len(mol._atom)):
        re = rRegionalEmbedding(mol,mf,'minao',np.arange(i+1), 0.1, 0.1)
        if len(re.indx_frz) > 0:
            mycc = pyscf.cc.CCSD(re.mf_proj, mo_coeff=re.moC_proj, frozen=re.indx_frz)
        else:
            mycc = pyscf.cc.CCSD(re.mf_proj, mo_coeff=re.moC_proj)
        mycc.kernel()
        ecorr_size.append(mycc.e_corr*27.211399)
    
    plt.figure()
    plt.plot(1+np.arange(len(ecorr_size)), ecorr_size, label="re-CCSD")
    plt.plot(1+np.arange(len(ecorr_size)), [ecorr_full]*len(ecorr_size), label="full-CCSD")
    plt.xlabel("Atoms defining local space")
    plt.ylabel("E corr (eV)")
    plt.legend()
    plt.savefig("./tests/conv_minao.pdf",bbox_inches='tight')

    # energy difference 
    xyz_file = "./tests/phenol.xyz"
    mol = pyscf.M(atom=xyz_file,basis='6-31G')
    mf = mol.RHF().run()
    mf_e_ph = mf.e_tot
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()
    ecorr_full_phe = mycc.e_corr * 27.211399
    
    re = rRegionalEmbedding(mol,mf,'6-31G',[0], 0.1, 0.1)
    mycc = pyscf.cc.CCSD(mf, mo_coeff=re.moC_proj, frozen=re.indx_frz)
    mycc.kernel()
    e_corr_re_phe = mycc.e_corr * 27.211399
    print(mf_e - mf_e_ph)
    print(e_corr_re-e_corr_re_phe)
    print(ecorr_full-ecorr_full_phe)
    
    
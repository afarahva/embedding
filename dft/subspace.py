#!/usr/bin/env python3# -*- coding: utf-8 -*-"""subspace.pyImplements Manby & Miller Projection Based Embedding withot a level-shiftingoperator by diagonalizing the Fock matrix only within the active block. author: Ardavan Farahvash, github.com/afarahva"""import numpy as npfrom scipy.linalg import eighfrom pyscf_embedding.dft import prbefrom pyscf import scf, dft, dfclass rSubspace(prbe.rPRBE):    def eig_wrapper(self,P):        def eig(h,s):            if P is None:                e, c = eigh(h, s)            else:                h_new = P.T @ h @ P                s_new = P.T @ s @ P                e,c = eigh(h_new,s_new)                c = P @ c                            idx = np.argmax(abs(c.real), axis=0)            c[:,c[idx,np.arange(len(e))].real<0] *= -1            return e,c        return eig            def kernel(self, xc_embed=None, deltadm_corr=False, **kwargs):                mf_B = self.mf_B                # copy input mf        ovlp = mf_B.get_ovlp()        mol = mf_B.mol.copy()        mol.nelectron = int(2 * self.moC_occ_A.shape[1])                # build embedding potential and effective core potential        dm_A0 = self.make_rdmA_init()        f_ab = mf_B.get_fock()        v_a = mf_B.get_veff(dm=dm_A0)        hcore = f_ab - v_a                # construct huzinaga projector        projector = self.get_huzinaga(f_ab, ovlp)        self.projector = projector                # get electronic energy for MOs in active subsystem        self.energy_a, _ = mf_B.energy_elec(dm=dm_A0, vhf=v_a, h1e=hcore)        # make embedding mean field object                if xc_embed is None:            self.mf_A = scf.RHF(mol,**kwargs)                    else:            self.mf_A = dft.RKS(mol)            self.mf_A.xc = xc_embed                    if hasattr(mf_B, 'with_df'):            self.mf_A = df.density_fit(self.mf_A)            self.mf_A.with_df.auxbasis = mf_B.with_df.auxbasis                    mf_A = self.mf_A        mf_A.diis_space = mf_B.diis_space        mf_A.max_cycle = mf_B.max_cycle                # set special embedding methods        mf_A.get_hcore = lambda *args: hcore        mf_A.eig = self.eig_wrapper(np.hstack([self.moC_occ_A,self.moC_vir_A]))                # run embedded SCF        mf_A.kernel()                    self.energy_embed = mf_A.e_tot        # find indices of bath/frozen MOs        epsilon = 1e-5        mask_act = np.linalg.norm( \          mf_A.mo_coeff.T @ ovlp @ np.hstack([self.moC_occ_B,self.moC_vir_B])\                    ,axis=1) <= epsilon                    mf_A.mo_energy = mf_A.mo_energy[mask_act]        mf_A.mo_coeff = mf_A.mo_coeff[:,mask_act]        mf_A.mo_occ = mf_A.mo_occ[mask_act]                # calculate total energy        energy_a_in_b = self.energy_embed - mf_A.energy_nuc()                        # recombined energy with embedded part        self.e_tot = mf_B.e_tot - self.energy_a + energy_a_in_b                if deltadm_corr:            self.deltaE_dm = self.energy_deltadm()            self.e_tot = self.e_tot + self.deltaE_dm                return self.e_tot, mf_A.mo_energy, mf_A.mo_coeff, mf_A.mo_occ                    #%%if __name__ == '__main__':    import matplotlib.pyplot as plt    import pyscf    from pyscf.tools import cubegen    from pyscf_embedding.local import regional as re    coords = \    """    O         -3.65830        0.00520       -0.94634    H         -4.10550        1.27483       -1.14033    C         -2.05632        0.04993       -0.35355    C         -1.42969        1.27592       -0.14855    C         -0.12337        1.31114        0.33487    C          0.54981        0.12269        0.61082    C         -0.08157       -1.10218        0.40403    C         -1.38785       -1.13991       -0.07931    H         -1.93037        2.15471       -0.35367    H          0.34566        2.21746        0.48856    H          1.51734        0.14971        0.96884    H          0.41837       -1.98145        0.60889    H         -1.85763       -2.04579       -0.23330    """    mol = pyscf.M(atom=coords,basis='ccpvdz',verbose=4)    mf = mol.RHF(max_cycles=100).run()    e_tot_hf = mf.e_tot.copy()        mol = pyscf.M(atom=coords,basis='ccpvdz',verbose=4)    mf = mol.RKS(xc='pbe',max_cycles=100).run()    e_tot_pbe = mf.e_tot.copy()        mol = pyscf.M(atom=coords,basis='ccpvdz',verbose=4)    mf = mol.RKS(xc='b3lyp',max_cycles=100).run()    e_tot_b3lyp = mf.e_tot.copy()        #%%    ##### Test DFT - in - DFT    mol = pyscf.M(atom=coords,basis='ccpvdz',verbose=4)    # mf = mol.RKS(xc='pbe',max_cycles=100).run()    mf = mol.RHF(max_cycles=100).run()    e_tot_arr1 = []    e_tot_arr2 = []    for i in range(1,9):        occ_calc = re.rRegionalActiveSpace(mf, np.arange(0,i+1), 'occ', basis='minao')        vir_calc = re.rRegionalActiveSpace(mf, np.arange(0,i+1), 'vir', basis='ccpvdz')                #         _,moC_occ,mask_occ_act = occ_calc.calc_mo()        _,moC_vir,mask_vir_act = vir_calc.calc_mo()        #                embed = rSubspace(mf, moC_occ, moC_vir, mask_occ_act, mask_vir_act)        e_tot,mo_energy,mo_coeff,mo_occ = embed.kernel(xc_embed='b3lyp',deltadm_corr=False)        print(e_tot)        deltaE_dm = embed.energy_deltadm()                e_tot_arr1.append(e_tot)        e_tot_arr2.append(e_tot-deltaE_dm)        plt.plot( (e_tot_arr1 - e_tot_pbe)*27.2114, color="red")    plt.plot( (e_tot_arr2 - e_tot_pbe)*27.2114, color="blue")    plt.plot( [e_tot_b3lyp*27.2114 - e_tot_pbe*27.2114]*len(e_tot_arr1), color="black")        #%%    ##### Test CCSD - in - DFT    # plt.ylabel("$E_c$ (hartree)")    # plt.legend(frameon=False)
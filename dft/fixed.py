#!/Users/ardavan/miniconda3/envs/pyscf_custom/bin/python
# -*- coding: utf-8 -*-
"""
fixed_density.py

Projection Based Embedding without a level-shift operator using a fixed density
approximation for the embedded orbital space

author: Ardavan Farahvash, github.com/afarahva
"""

import numpy as np
from pyscf_embedding.dft import prbe
from pyscf import scf, dft, df


class rFixedDensityEmbedding(prbe.rPRBE):

    def kernel(self, xc_embed=None, **kwargs):
        # copy input mf
        mf_B = self.mf_B
        mol = mf_B.mol.copy()

        # build embedding potential and effective core potential
        dm_A = self.make_rdmA_init()
        f_ab = mf_B.get_fock()
        v_a = mf_B.get_veff(dm=dm_A)
        hcore = f_ab - v_a

        # get electronic energy for MOs in active subsystem with bath method
        self.energy_a, _ = mf_B.energy_elec(dm=dm_A, vhf=v_a, h1e=hcore)

        # make embedding mean field object
        mol.nelectron = int(2 * self.moC_occ_A.shape[1])
        if xc_embed is None:
            self.mf_A = scf.RHF(mol, **kwargs)
        else:
            self.mf_A = scf.RKS(mol)
            self.mf_A.xc = xc_embed

        if hasattr(mf_B, 'with_df'):
            self.mf_A = df.density_fit(self.mf_A)
            self.mf_A.with_df.auxbasis = mf_B.with_df.auxbasis

        mf_A = self.mf_A
        mf_A.get_hcore = lambda *args: hcore

        # get embedded MO energies and coefficients
        moC_A = np.hstack([self.moC_occ_A, self.moC_vir_A])
        fock = moC_A.T @ mf_A.get_fock(dm=dm_A) @ moC_A
        e, u = np.linalg.eigh(fock)

        mf_A.mo_energy = e
        mf_A.mo_coeff = moC_A
        mf_A.mo_occ = np.zeros(len(e), dtype=np.int64)
        mf_A.mo_occ[0:self.moC_occ_A.shape[1]] = 2

        # set special embedding methods
        self.energy_embed = mf_A.energy_tot(dm=dm_A)

        # calculate total energy
        energy_a_in_b = self.energy_embed - mf_A.energy_nuc()

        # recombined energy with embedded part; subtract fragment dispersion
        # already captured in mf_B.e_tot to avoid double-counting
        self.e_tot = mf_B.e_tot - self.energy_a + energy_a_in_b - self.calc_disp_embed()
        return self.e_tot, mf_A.mo_energy, mf_A.mo_coeff, mf_A.mo_occ


class uFixedDensityEmbedding(prbe.uPRBE):

    def kernel(self, xc_embed=None, **kwargs):
        mf_B = self.mf_B
        mol = mf_B.mol.copy()

        # build spin-dependent embedding potential and effective core potential
        dm_A = self.make_rdmA_init()          # (2, nao, nao)
        f_ab = mf_B.get_fock()               # (2, nao, nao)
        v_a  = mf_B.get_veff(dm=dm_A)        # (2, nao, nao), carries ecoul/exc
        hcore = f_ab - v_a                   # (2, nao, nao)

        # electronic energy of subsystem A computed at the bath level
        self.energy_a, _ = self.energy_elec(mf_B, dm_A, hcore, v_a)

        # set electron count and spin on the embedded mol
        n_alpha = self.moC_occ_A[0].shape[1]
        n_beta  = self.moC_occ_A[1].shape[1]
        mol.nelectron = int(n_alpha + n_beta)
        mol.spin      = int(n_alpha - n_beta)

        if xc_embed is None:
            self.mf_A = scf.UHF(mol, **kwargs)
        else:
            self.mf_A = dft.UKS(mol)
            self.mf_A.xc = xc_embed

        if hasattr(mf_B, 'with_df'):
            self.mf_A = df.density_fit(self.mf_A)
            self.mf_A.with_df.auxbasis = mf_B.with_df.auxbasis

        mf_A = self.mf_A
        mf_A.get_hcore = lambda *args: hcore

        # project Fock into the alpha and beta active-space bases
        moC_A_a = np.hstack([self.moC_occ_A[0], self.moC_vir_A[0]])
        moC_A_b = np.hstack([self.moC_occ_A[1], self.moC_vir_A[1]])
        fock = mf_A.get_fock(dm=dm_A)        # (2, nao, nao)
        e_a, _ = np.linalg.eigh(moC_A_a.T @ fock[0] @ moC_A_a)
        e_b, _ = np.linalg.eigh(moC_A_b.T @ fock[1] @ moC_A_b)

        mf_A.mo_energy = np.array([e_a, e_b])
        mf_A.mo_coeff  = np.array([moC_A_a, moC_A_b])
        mo_occ = np.zeros((2, moC_A_a.shape[1]), dtype=np.int64)
        mo_occ[0, :n_alpha] = 1
        mo_occ[1, :n_beta]  = 1
        mf_A.mo_occ = mo_occ

        # embedded electronic energy using spin-aware helper
        vhf_A = mf_A.get_veff(dm=dm_A)
        energy_a_in_b, _ = self.energy_elec(mf_A, dm_A, hcore, vhf_A)
        self.energy_embed = energy_a_in_b + mf_A.energy_nuc()

        # recombined energy; subtract fragment dispersion to avoid double-counting
        self.e_tot = mf_B.e_tot - self.energy_a + energy_a_in_b - self.calc_disp_embed()
        return self.e_tot, mf_A.mo_energy, mf_A.mo_coeff, mf_A.mo_occ


 #%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pyscf
    from pyscf.tools import cubegen
    from pyscf_embedding.local import regional as re

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
    mol = pyscf.M(atom=coords, basis='ccpvdz', verbose=4)
    mf = mol.RHF(max_cycles=100).run()
    e_tot_hf = mf.e_tot.copy()

    mol = pyscf.M(atom=coords, basis='ccpvdz', verbose=4)
    mf = mol.RKS(xc='pbe', max_cycles=100).run()
    e_tot_pbe = mf.e_tot.copy()

    mol = pyscf.M(atom=coords, basis='ccpvdz', verbose=4)
    mf = mol.RKS(xc='b3lyp', max_cycles=100).run()
    e_tot_b3lyp = mf.e_tot.copy()
    #%%
    ##### Test DFT - in - DFT
    mol = pyscf.M(atom=coords, basis='ccpvdz', verbose=4)
    mf = mol.RKS(xc='pbe', max_cycles=100).run()

    e_tot_arr1 = []
    for i in range(1, 9):
        occ_calc = re.RegionalActiveSpace(mf, np.arange(0, i+1), 'occ', basis='minao')
        vir_calc = re.RegionalActiveSpace(mf, np.arange(0, i+1), 'vir', basis='ccpvdz')

        embed = rFixedDensityEmbedding(mf, occ_calc, vir_calc)
        e_tot, _, _, _ = embed.kernel(xc_embed='b3lyp')
        print(embed.mf_A.mo_coeff.shape)
        e_tot_arr1.append(e_tot)

    plt.plot((e_tot_arr1 - e_tot_hf)*27.2114, color="red")
    plt.plot([e_tot_b3lyp*27.2114 - e_tot_hf*27.2114]*len(e_tot_arr1), color="black")

    #%%
    ##### Test CCSD - in - DFT
    mol = pyscf.M(atom=coords, basis='3-21g', verbose=4)
    mf = mol.RHF(max_cycles=100).run()
    mycc = pyscf.cc.CCSD(mf).run()
    e_tot_ccsd = mycc.e_tot.copy()

    mol = pyscf.M(atom=coords, basis='3-21g', verbose=4)
    mf = mol.RKS(xc='b3lyp', max_cycles=100).run()
    e_tot_dft = mf.e_tot.copy()

    e_tot_arr1 = []
    for i in range(1, 13):
        occ_calc = re.RegionalActiveSpace(mf, np.arange(0, i+1), 'occ', basis='minao')
        vir_calc = re.RegionalActiveSpace(mf, np.arange(0, i+1), 'vir', basis='3-21g')

        embed = rFixedDensityEmbedding(mf, occ_calc, vir_calc)
        e_tot, _, _, _ = embed.kernel(xc_embed=None)
        mycc = pyscf.cc.CCSD(embed.mf_A, mo_coeff=embed.mf_A.mo_coeff)
        mycc.kernel()

        e_tot_arr1.append(e_tot + mycc.e_corr)

    plt.plot((e_tot_arr1 - e_tot_dft)*27.2114, color="red")
    plt.plot([e_tot_ccsd*27.2114 - e_tot_dft*27.2114]*len(e_tot_arr1), color="black")

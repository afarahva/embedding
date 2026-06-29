#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mp2testing.py

Cheap restricted MP2 diagnostics for PRBE-like objects.

This module decomposes a semicanonical DF-MP2 doubles energy into A-only,
B-only, and mixed A/B pieces.  For DFT bath references, it also evaluates a
Brillouin-singles diagnostic after semicanonicalizing the bath occupied and
virtual spaces with an HF Fock built from the bath density.

Doubles bucketing
-----------------
Each MP2 doubles term is built from two integrals,

    direct    (ia|jb)            -> clouds (i->a) and (j->b)
    exchange  (ib|ja)            -> clouds (i->b) and (j->a)

with spin-summed (closed-shell) energy density

    e_comb[a,b]   = (ia|jb) [2(ia|jb) - (ib|ja)] / D          (full RMP2)
    e_direct[a,b] = (ia|jb)^2 / D                              (OS / direct only)

so that  E_OS = sum e_direct,  E_SS = sum e_comb - E_OS,  E = sum e_comb.

The *direction of charge transfer is a per-cloud property*: cloud (i->a) moves
an electron from frag(i) to frag(a).  Reading the integral (ia|jb):

    (AA|BB) : i_A->a_A and j_B->b_B   -> both neutral -> DISPERSION (net 0)
    (AB|AB) : i_A->a_B and j_A->b_B   -> both leave A -> IONIC (net +2)
    (AB|BA) : i_A->a_B and j_B->b_A   -> opposite CT  -> exchange-dispersion (net 0)

Note the index-string convention used internally is "ijab" = label(i)label(j)
label(a)label(b).  As an integral that string is (i a | j b), i.e. the string
"ABAB" is the integral (AA|BB) (dispersion), and the string "AABB" is the
integral (AB|AB) (ionic).  Both labels are printed to avoid confusion.

Set 1 (combined, direct+exchange) is classified by NET inter-fragment charge
transfer only -- pure / dispersion (net-0 cross) / single_ct / ionic -- with no
directional (A->B vs B->A) or dispersion-vs-bidirectional sub-split, since those
are direct-vs-exchange artifacts rather than amplitude properties.  Set 2
(direct/OS only) uses the same net-charge buckets, evaluated on (ia|jb)^2/D.
"""

from collections import defaultdict

import numpy as np
from pyscf import df, lib, scf
from pyscf.lib import logger
from pyscf.mp import dfmp2


def _dft_xc_energy(mf, dm):
    veff = mf.get_veff(mf.mol, dm)
    if not hasattr(veff, 'exc'):
        raise TypeError('DFT nonadditive XC requires a DFT mf_B object')
    return float(np.asarray(veff.exc).real)


def _mf_total_energy(mf, dm):
    vhf = mf.get_veff(mf.mol, dm)
    return float(mf.energy_tot(dm=dm, h1e=mf.get_hcore(mf.mol), vhf=vhf).real)


def dft_nonadditive(prbe):
    """
    DFT bath and nonadditive energies from the original PRBE A/B densities.

    E_nad[D_A,D_B] = E[D_A + D_B] - E[D_A] - E[D_B], with all terms
    evaluated by ``prbe.mf_B``.  The A/B densities are the input partition
    densities, before embedded optimization.
    """
    mf = prbe.mf_B
    dm_A = 2.0 * prbe.moC_occ_A @ prbe.moC_occ_A.T
    dm_B = 2.0 * prbe.moC_occ_B @ prbe.moC_occ_B.T
    dm_AB = dm_A + dm_B

    e_AB = _mf_total_energy(mf, dm_AB)
    e_A = _mf_total_energy(mf, dm_A)
    e_B = _mf_total_energy(mf, dm_B)

    e_xc_AB = _dft_xc_energy(mf, dm_AB)
    e_xc_A = _dft_xc_energy(mf, dm_A)
    e_xc_B = _dft_xc_energy(mf, dm_B)

    return {
        'dft_bath_xc_energy': e_xc_B,
        'dft_nonadditive_xc_energy': e_xc_AB - e_xc_A - e_xc_B,
    }


dft_nonadditive_xc = dft_nonadditive


def _hf_exchange_energy(mf, dm):
    """Closed-shell HF exchange energy for a spin-summed restricted density."""
    mf_hf = scf.RHF(mf.mol)
    with_df = getattr(mf, 'with_df', None)
    if with_df is not None:
        mf_hf = df.density_fit(mf_hf)
        mf_hf.with_df.auxbasis = getattr(with_df, 'auxbasis', None)
    vk = mf_hf.get_jk(mf_hf.mol, dm, with_j=False, with_k=True)
    if isinstance(vk, tuple):
        vk = vk[1]
    return float((-0.25 * np.einsum('ij,ji->', dm, vk)).real)


def _masked_mo_blocks(moC_occ, moC_vir, mask_occ_act, mask_vir_act):
    moC_occ = np.asarray(moC_occ)
    moC_vir = np.asarray(moC_vir)
    mask_occ_act = np.asarray(mask_occ_act, dtype=bool)
    mask_vir_act = np.asarray(mask_vir_act, dtype=bool)

    if moC_occ.ndim != 2 or moC_vir.ndim != 2:
        raise ValueError('moC_occ and moC_vir must be restricted 2D arrays')
    if mask_occ_act.size != moC_occ.shape[1]:
        raise ValueError('mask_occ_act length does not match moC_occ')
    if mask_vir_act.size != moC_vir.shape[1]:
        raise ValueError('mask_vir_act length does not match moC_vir')

    return (
        moC_occ[:, mask_occ_act],
        moC_vir[:, mask_vir_act],
        moC_occ[:, ~mask_occ_act],
        moC_vir[:, ~mask_vir_act],
    )


def _mf_mo_blocks(mf):
    mo_coeff = getattr(mf, 'mo_coeff', None)
    mo_occ = getattr(mf, 'mo_occ', None)
    if mo_coeff is None or mo_occ is None:
        raise AttributeError('mf_A object has no mo_coeff/mo_occ')
    if isinstance(mo_coeff, (tuple, list)):
        raise NotImplementedError('hfindft override is currently restricted-only')

    mo_coeff = np.asarray(mo_coeff)
    occ_mask = np.asarray(mo_occ) > 1e-10
    return mo_coeff[:, occ_mask], mo_coeff[:, ~occ_mask]


def _subspace_mp2_corr(mf, c_occ, c_vir, auxbasis=None, max_memory=None,
                       verbose=None, frozen=0, outcore=False):
    if frozen < 0:
        raise ValueError('frozen must be non-negative')
    if frozen > c_occ.shape[1]:
        raise ValueError('frozen exceeds the number of occupied orbitals')
    c_occ = c_occ[:, frozen:]
    if c_occ.shape[1] == 0 or c_vir.shape[1] == 0:
        return 0.0
    if max_memory is None:
        max_memory = getattr(mf, 'max_memory', 2000)
    if verbose is None:
        verbose = getattr(mf, 'verbose', logger.NOTE)
    if auxbasis is None:
        with_df = getattr(mf, 'with_df', None)
        auxbasis = getattr(with_df, 'auxbasis', None) if with_df is not None else None
        if auxbasis is None:
            auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

    fock = mf.get_fock()
    f_occ = c_occ.T @ fock @ c_occ
    f_occ = 0.5 * (f_occ + f_occ.T)
    e_occ, u_occ = np.linalg.eigh(f_occ)
    c_occ = c_occ @ u_occ

    f_vir = c_vir.T @ fock @ c_vir
    f_vir = 0.5 * (f_vir + f_vir.T)
    e_vir, u_vir = np.linalg.eigh(f_vir)
    c_vir = c_vir @ u_vir
    print(e_occ,e_vir)
    labels_occ = np.array(['S'] * c_occ.shape[1])
    labels_vir = np.array(['S'] * c_vir.shape[1])
    log = logger.new_logger(mf, verbose)
    result = _doubles_decomp(
        mf, c_occ, c_vir, e_occ, e_vir, labels_occ, labels_vir,
        auxbasis, max_memory, log, outcore=outcore
    )
    # 'total' is the full (combined direct+exchange) correlation energy.
    return result['total']


def mp2_nonadditive(mf, moC_occ, moC_vir, mask_occ_act, mask_vir_act,
                    auxbasis=None, max_memory=None, verbose=None,
                    frozen_sys=0, frozen_env=0, outcore=False,
                    hfindft=None):
    """
    HF exchange and MP2 correlation bath/nonadditive energies from a converged
    canonical HF object.
    """
    if auxbasis is None:
        with_df = getattr(mf, 'with_df', None)
        auxbasis = getattr(with_df, 'auxbasis', None) if with_df is not None else None
        if auxbasis is None:
            auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
    if frozen_sys < 0 or frozen_env < 0:
        raise ValueError('frozen_sys and frozen_env must be non-negative')

    moC_occ_A, moC_vir_A, moC_occ_B, moC_vir_B = _masked_mo_blocks(
        moC_occ, moC_vir, mask_occ_act, mask_vir_act
    )
    dm_A = 2.0 * moC_occ_A @ moC_occ_A.T
    dm_B = 2.0 * moC_occ_B @ moC_occ_B.T
    dm_AB = dm_A + dm_B

    e_hf_AB = _mf_total_energy(mf, dm_AB)
    e_hf_A = _mf_total_energy(mf, dm_A)
    e_hf_B = _mf_total_energy(mf, dm_B)

    e_x_AB = _hf_exchange_energy(mf, dm_AB)
    e_x_A = _hf_exchange_energy(mf, dm_A)
    e_x_B = _hf_exchange_energy(mf, dm_B)
    e_x_nad = e_x_AB - e_x_A - e_x_B


    if frozen_sys > moC_occ_A.shape[1]:
        raise ValueError('frozen_sys exceeds the number of system occupied orbitals')
    if frozen_env > moC_occ_B.shape[1]:
        raise ValueError('frozen_env exceeds the number of bath occupied orbitals')

    pt = dfmp2.DFMP2(mf, frozen=frozen_sys + frozen_env)
    pt.max_memory = (
        max_memory if max_memory is not None else getattr(mf, 'max_memory', 2000)
    )
    pt.force_outcore = bool(outcore)
    pt.with_df = df.DF(mf.mol)
    pt.with_df.auxbasis = auxbasis
    e_corr_AB = float(pt.run().e_corr)

    mf_A = getattr(hfindft, 'mf_A', mf)

    e_corr_A = _subspace_mp2_corr(
        mf, moC_occ_A, moC_vir_A, auxbasis=auxbasis,
        max_memory=max_memory, verbose=verbose, frozen=frozen_sys,
        outcore=outcore,
    )

    e_corr_B = _subspace_mp2_corr(
        mf, moC_occ_B, moC_vir_B, auxbasis=auxbasis,
        max_memory=max_memory, verbose=verbose, frozen=frozen_env,
        outcore=outcore,
    )
    e_corr_nad = e_corr_AB - e_corr_A - e_corr_B
    e_corr_B = e_corr_B
    
    if hfindft is not None:
        mf_B = getattr(hfindft, 'mf_B', mf)
        e_corr_hfindft = hfindft.e_tot - e_hf_AB
        e_corr_hfindft_B =  _mf_total_energy(mf_B, dm_B) - e_hf_B
        e_corr_hfindft_nad = e_corr_hfindft - e_corr_hfindft_B 


        e_corrdft_nad = e_corr_nad - e_corr_hfindft_nad
        e_corrdft_B = e_corr_B - e_corr_hfindft_B
        return {
            'hf_exchange_bath_energy': e_x_B,
            'hf_exchange_nonadditive_energy': e_x_nad,
            
            'delmp2_correlation_energy':e_corr_AB-e_corr_A,
            'mp2_correlation_nonadditive_energy': e_corr_nad,
            'mp2_correlation_bath_energy': e_corr_B,
            
            'delmp2dft_energy': e_corr_AB-e_corr_A-e_corr_hfindft,
            'mp2dft_nonadditive_energy': e_corrdft_nad,
            'mp2dft_bath_energy': e_corrdft_B}
    
    else:
        return {
            'hf_exchange_bath_energy': e_x_B,
            'hf_exchange_nonadditive_energy': e_x_nad,
            
            'delmp2_correlation_energy':e_corr_AB-e_corr_A,
            'mp2_correlation_nonadditive_energy': e_corr_nad,
            'mp2_correlation_bath_energy': e_corr_B}

mp2_nonadditive_xc = mp2_nonadditive


def _semicanonicalize_bath(mf, moC_occ_B, moC_vir_B, dft_bath):
    """
    If the bath came from DFT, semicanonicalize B occ/vir spaces with an HF Fock
    built from the bath density only.
    """
    if not dft_bath:
        fock = mf.get_fock()
        parent_coeff = getattr(mf, 'mo_coeff', None)
        parent_energy = getattr(mf, 'mo_energy', None)
        moE_occ_B = None
        moE_vir_B = None
        if parent_coeff is not None and parent_energy is not None and \
                not isinstance(parent_coeff, (tuple, list)):
            ovlp = mf.get_ovlp()
            for coeff, target in ((moC_occ_B, 'occ'), (moC_vir_B, 'vir')):
                if coeff.shape[1] == 0:
                    energy = np.array([])
                else:
                    overlaps = parent_coeff.T @ ovlp @ coeff
                    idx = np.argmax(np.abs(overlaps), axis=0)
                    best = np.abs(overlaps[idx, np.arange(coeff.shape[1])])
                    if np.all(best > 1.0 - 1e-8) and np.unique(idx).size == idx.size:
                        energy = np.asarray(parent_energy)[idx]
                    else:
                        energy = None
                if target == 'occ':
                    moE_occ_B = energy
                else:
                    moE_vir_B = energy
        if moE_occ_B is None:
            moE_occ_B = np.einsum('pi,pq,qi->i', moC_occ_B, fock, moC_occ_B).real
        if moE_vir_B is None:
            moE_vir_B = np.einsum('pi,pq,qi->i', moC_vir_B, fock, moC_vir_B).real
        return moE_occ_B, moC_occ_B, moE_vir_B, moC_vir_B, fock

    dm_B = 2.0 * moC_occ_B @ moC_occ_B.T
    mf_hf = scf.RHF(mf.mol)
    with_df = getattr(mf, 'with_df', None)
    if with_df is not None:
        mf_hf = df.density_fit(mf_hf)
        mf_hf.with_df.auxbasis = getattr(with_df, 'auxbasis', None)
    fock_B = mf_hf.get_hcore(mf.mol) + mf_hf.get_veff(mf.mol, dm_B)

    f_occ_B = moC_occ_B.T @ fock_B @ moC_occ_B
    f_occ_B = 0.5 * (f_occ_B + f_occ_B.T)
    moE_occ_B, u_occ_B = np.linalg.eigh(f_occ_B)
    moC_occ_B = moC_occ_B @ u_occ_B

    f_vir_B = moC_vir_B.T @ fock_B @ moC_vir_B
    f_vir_B = 0.5 * (f_vir_B + f_vir_B.T)
    moE_vir_B, u_vir_B = np.linalg.eigh(f_vir_B)
    moC_vir_B = moC_vir_B @ u_vir_B
    return moE_occ_B, moC_occ_B, moE_vir_B, moC_vir_B, fock_B


def _preporbs(prbe):
    """Extract optimized A orbitals and semicanonical bath orbitals."""
    required = ('mf_B', 'moC_occ_A', 'moC_vir_A', 'moC_occ_B', 'moC_vir_B')
    missing = [name for name in required if not hasattr(prbe, name)]
    if missing:
        raise AttributeError(
            'PRBE-like object is missing required attributes: '
            + ', '.join(missing)
        )
    if getattr(prbe, 'mf_A', None) is None:
        raise AttributeError(
            'MP2 decomposition requires optimized prbe.mf_A orbitals; '
            'run the embedding kernel first'
        )

    mf = prbe.mf_B
    mo_coeff_A = getattr(prbe.mf_A, 'mo_coeff', None)
    mo_occ_A = getattr(prbe.mf_A, 'mo_occ', None)
    mo_energy_A = getattr(prbe.mf_A, 'mo_energy', None)
    if mo_coeff_A is None or mo_occ_A is None:
        raise AttributeError('optimized subsystem MF has no mo_coeff/mo_occ')
    if isinstance(mo_coeff_A, (tuple, list)):
        raise NotImplementedError('MP2 diagnostics are currently restricted-only')
    occ_mask_A = np.asarray(mo_occ_A) > 1e-10
    vir_mask_A = ~occ_mask_A
    moC_occ_A = np.asarray(mo_coeff_A)[:, occ_mask_A]
    moC_vir_A = np.asarray(mo_coeff_A)[:, vir_mask_A]
    moE_occ_A = moE_vir_A = None
    if mo_energy_A is not None:
        mo_energy_A = np.asarray(mo_energy_A)
        moE_occ_A = mo_energy_A[occ_mask_A]
        moE_vir_A = mo_energy_A[vir_mask_A]

    moC_occ_B = prbe.moC_occ_B
    moC_vir_B = prbe.moC_vir_B
    if any(isinstance(block, (tuple, list)) for block in
           (moC_occ_A, moC_vir_A, moC_occ_B, moC_vir_B)):
        raise NotImplementedError('MP2 diagnostics are currently restricted-only')

    xc = getattr(mf, 'xc', None)
    dft_bath = xc is not None and str(xc).strip().lower() not in ('', 'hf')
    moE_occ_B, moC_occ_B, moE_vir_B, moC_vir_B, fock_B = \
        _semicanonicalize_bath(
            mf, moC_occ_B, moC_vir_B, dft_bath
        )

    if moE_occ_A is None or moE_vir_A is None:
        dm_A = 2.0 * moC_occ_A @ moC_occ_A.T
        mf_hf = scf.RHF(mf.mol)
        with_df = getattr(mf, 'with_df', None)
        if with_df is not None:
            mf_hf = df.density_fit(mf_hf)
            mf_hf.with_df.auxbasis = getattr(with_df, 'auxbasis', None)
        fock_A = mf_hf.get_hcore(mf.mol) + mf_hf.get_veff(mf.mol, dm_A)
        if moE_occ_A is None:
            moE_occ_A = np.einsum('pi,pq,qi->i', moC_occ_A, fock_A, moC_occ_A).real
        if moE_vir_A is None:
            moE_vir_A = np.einsum('pi,pq,qi->i', moC_vir_A, fock_A, moC_vir_A).real

    return {
        'mf': mf,
        'dft_bath': dft_bath,
        'moC_occ_A': moC_occ_A,
        'moC_vir_A': moC_vir_A,
        'moC_occ_B': moC_occ_B,
        'moC_vir_B': moC_vir_B,
        'moE_occ_A': moE_occ_A,
        'moE_vir_A': moE_vir_A,
        'moE_occ_B': moE_occ_B,
        'moE_vir_B': moE_vir_B,
        'fock_B': fock_B,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
#
# Doubles fine groups, classified from the cloud charges of the *direct*
# integral (ia|jb) whose clouds are (i->a) and (j->b).  These are used for BOTH
# the combined (direct+exchange) energy and the OS/direct-only energy.

DOUBLES_GROUPS = (
    'pure_system',   # nA==4         (AA|AA)
    'pure_bath',     # nA==0         (BB|BB)
    'dispersion',    # |net|==0 cross (AA|BB),(BB|AA),(AB|BA),(BA|AB)
    'single_ct',     # |net|==1      (AA|AB)-type, either direction
    'ionic',         # |net|==2      (AB|AB),(BA|BA)
)

SINGLES_GROUPS = (
    'pure_system',
    'a_to_b_charge_transfer',
    'b_to_a_charge_transfer',
    'pure_bath',
)


def _integral_label(pattern):
    """ijab index-string -> chemist-notation cloud label '(ia|jb)'."""
    li, lj, la, lb = pattern
    return '(%s%s|%s%s)' % (li, la, lj, lb)


def _classify_pattern(pattern):
    """
    Classify a doubles amplitude by net inter-fragment charge transfer only.

      pattern = label(i) label(j) label(a) label(b)   (holes i,j ; particles a,b)
      net = (#holes in A) - (#particles in A)
          = number of electrons that leave A across the excitation

    Net charge is invariant under the a<->b swap that exchanges the direct and
    exchange integrals, so it is the ONLY well-defined amplitude-level label.
    The dispersion-vs-bidirectional sub-split of the net-0 cross block, and the
    A->B / B->A direction of single/double CT, are direct-vs-exchange / cloud
    artifacts -- not properties of the amplitude -- so they are NOT split here.

      nA==4        -> pure_system   e.g. (AA|AA)
      nA==0        -> pure_bath     e.g. (BB|BB)
      |net|==0 cross -> dispersion  e.g. (AA|BB),(BB|AA),(AB|BA),(BA|AB)
      |net|==1     -> single_ct     e.g. (AA|AB)            (either direction)
      |net|==2     -> ionic         e.g. (AB|AB) = A^2+ B^2- (either direction)
    """
    if len(pattern) != 4:
        return 'unclassified'
    li, lj, la, lb = pattern
    nA = pattern.count('A')
    if nA == 4:
        return 'pure_system'
    if nA == 0:
        return 'pure_bath'
    net = ((li == 'A') + (lj == 'A')) - ((la == 'A') + (lb == 'A'))
    if net == 0:
        return 'dispersion'
    if abs(net) == 1:
        return 'single_ct'
    return 'ionic'


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

def _add_component(result, pattern, value, group=None):
    n_A = pattern.count('A')
    result['by_pattern'][pattern] += value
    if group is not None:
        result['by_group'][group] += value
    if n_A == len(pattern):
        result['system_only'] += value
    elif n_A == 0:
        result['bath_only'] += value
    else:
        result['cross_total'] += value
        result['cross_by_num_A'][n_A] += value
        # Cross energy excluding doubles whose *both* virtuals (a,b) sit on
        # bath B.  pattern = label(i)label(j)label(a)label(b), so the virtual
        # labels are pattern[2] (a) and pattern[3] (b).  The len==4 guard keeps
        # singles (length-2 patterns) from ever contributing here.
        if len(pattern) == 4 and not (pattern[2] == 'B' and pattern[3] == 'B'):
            result['cross_excl_bb_vir'] += value


def _finalize_component(result):
    result['total'] = (
        result['system_only'] + result['bath_only'] + result['cross_total']
    )
    result['composite_missing'] = result['bath_only'] + result['cross_total']
    result['by_pattern'] = {
        k: float(v) for k, v in sorted(result['by_pattern'].items())
    }
    result['cross_by_num_A'] = {
        k: float(result['cross_by_num_A'][k]) for k in sorted(result['cross_by_num_A'])
    }
    result['by_group'] = {
        k: float(v) for k, v in sorted(result['by_group'].items())
    }
    for key in ('total', 'system_only', 'bath_only', 'cross_total',
                'cross_excl_bb_vir', 'composite_missing'):
        result[key] = float(result[key])
    return result


def _empty_component(groups=()):
    by_group = defaultdict(float)
    for group in groups:
        by_group[group] = 0.0
    return {
        'total': 0.0,
        'system_only': 0.0,
        'bath_only': 0.0,
        'cross_total': 0.0,
        'cross_excl_bb_vir': 0.0,   # cross energy minus both-virtuals-on-B doubles
        'composite_missing': 0.0,
        'cross_by_num_A': defaultdict(float),
        'by_pattern': defaultdict(float),
        'by_group': by_group,
    }


def _doubles_decomp(mf, c_occ, c_vir, e_occ, e_vir,
                    occ_labels, vir_labels, auxbasis, max_memory, log,
                    outcore=False):
    """
    Decompose the restricted DF-MP2 doubles energy.

    Returns a dict with two finalized components computed in a single pass:

      'combined' : full RMP2,    e_comb   = (ia|jb)[2(ia|jb)-(ib|ja)] / D
      'direct'   : OS / direct,  e_direct = (ia|jb)^2 / D

    Top-level convenience keys ('total', etc.) mirror the combined component so
    that callers reading result['total'] get the full correlation energy.
    """
    occ_labels = np.asarray(occ_labels)
    vir_labels = np.asarray(vir_labels)

    combined = _empty_component(DOUBLES_GROUPS)
    direct = _empty_component(DOUBLES_GROUPS)

    if c_occ.shape[1] == 0 or c_vir.shape[1] == 0:
        combined = _finalize_component(combined)
        direct = _finalize_component(direct)
        out = dict(combined)
        out['combined'] = combined
        out['direct'] = direct
        return out

    # Virtual-block masks, keyed by the actual label characters present (handles
    # the all-'S' subspace runs as well as A/B production runs).
    vir_label_vals = list(dict.fromkeys(vir_labels.tolist()))
    vir_masks = {lab: (vir_labels == lab) for lab in vir_label_vals}
    vir_blocks = [
        (la, lb, np.outer(vir_masks[la], vir_masks[lb]))
        for la in vir_label_vals for lb in vir_label_vals
    ]

    mo_coeff = np.hstack([c_occ, c_vir])
    mo_occ = np.hstack([
        np.full(c_occ.shape[1], 2.0),
        np.zeros(c_vir.shape[1]),
    ])
    mo_energy = np.hstack([e_occ, e_vir])
    pt = dfmp2.DFMP2(mf, frozen=0, mo_coeff=mo_coeff,
                     mo_occ=mo_occ, mo_energy=mo_energy)
    pt.max_memory = max_memory
    pt.force_outcore = bool(outcore)
    pt.verbose = log.verbose
    pt.stdout = log.stdout
    pt.with_df = df.DF(mf.mol)
    pt.with_df.auxbasis = auxbasis

    eris = pt.ao2mo()
    evv = e_vir[:, None] + e_vir[None, :]
    nocc = c_occ.shape[1]

    for i in range(nocc):
        ints_i = eris.get_occ_blk(i, i + 1)[0]
        li = occ_labels[i]
        for j in range(nocc):
            ints_j = eris.get_occ_blk(j, j + 1)[0]
            lj = occ_labels[j]

            coul = lib.dot(ints_i, ints_j.T)      # coul[a,b] = (ia|jb)
            exch = coul.T                         # exch[a,b] = (ib|ja)
            denom = e_occ[i] + e_occ[j] - evv     # = eps_i + eps_j - eps_a - eps_b

            e_direct = (coul * coul / denom).real
            e_comb = (coul * (2.0 * coul - exch) / denom).real

            # Bin the (a,b) plane by virtual-block; pattern/group depend only on
            # the fragment labels, so a handful of masked reductions per (i,j)
            # replaces the nvir^2 python loop.
            for la, lb, mask in vir_blocks:
                if not mask.any():
                    continue
                pattern = li + lj + la + lb
                group = _classify_pattern(pattern)
                _add_component(combined, pattern,
                               float(e_comb[mask].sum()), group=group)
                _add_component(direct, pattern,
                               float(e_direct[mask].sum()), group=group)

    combined = _finalize_component(combined)
    direct = _finalize_component(direct)

    out = dict(combined)            # mirror combined at top level (back-compat)
    out['combined'] = combined
    out['direct'] = direct
    return out


def _singles_decomp(fock, c_occ, c_vir, e_occ, e_vir,
                    occ_labels, vir_labels, enabled):
    """
    Cheap Brillouin-singles diagnostic.

    For closed-shell RHF spin adaptation, the energy contribution is
    2 * |f_ia|^2 / (eps_i - eps_a).  The user-facing formula is sometimes
    written schematically as |f_ia| / Delta; the square is required for energy
    dimensions.
    """
    result = _empty_component(SINGLES_GROUPS)
    if not enabled or c_occ.shape[1] == 0 or c_vir.shape[1] == 0:
        return _finalize_component(result)

    fov = c_occ.T @ fock @ c_vir
    denom = e_occ[:, None] - e_vir[None, :]
    e_mat = 2.0 * np.abs(fov)**2 / denom

    def _singles_group(pattern):
        if pattern == 'AA':
            return 'pure_system'
        if pattern == 'AB':
            return 'a_to_b_charge_transfer'
        if pattern == 'BA':
            return 'b_to_a_charge_transfer'
        if pattern == 'BB':
            return 'pure_bath'
        return 'unclassified'

    for i in range(c_occ.shape[1]):
        for a in range(c_vir.shape[1]):
            pattern = occ_labels[i] + vir_labels[a]
            if pattern == 'AA':
                continue
            _add_component(
                result, pattern, e_mat[i, a],
                group=_singles_group(pattern)
            )
    return _finalize_component(result)


def mp2_decomp_driver(prbe, auxbasis=None, max_memory=None, verbose=None,
                      frozen_sys=0, frozen_env=0, outcore=False):
    """
    Decompose approximate MP2 corrections for a restricted PRBE-like object.

    The doubles part is a semicanonical DF-MP2 expression, reported two ways:

      Set 1 (combined direct+exchange) : full RMP2 energy bucketed into
             pure_system, pure_bath, single_ct, and the "double" trio
             dispersion / exchange_dispersion / ionic.
      Set 2 (OS / direct only, (ia|jb)^2/D) : the cleanly cloud-resolved
             Coulomb-coupling energy in the same buckets.

    If ``prbe.mf_B`` is a DFT reference, bath occupied/virtual orbitals are
    automatically semicanonicalized with an HF Fock built from the bath density,
    and a singles diagnostic is evaluated with the composite A+B HF Fock.
    """
    data = _preporbs(prbe)
    mf = data['mf']
    if max_memory is None:
        max_memory = getattr(mf, 'max_memory', 2000)
    if verbose is None:
        verbose = getattr(mf, 'verbose', logger.NOTE)
    if auxbasis is None:
        with_df = getattr(mf, 'with_df', None)
        auxbasis = getattr(with_df, 'auxbasis', None) if with_df is not None else None
        if auxbasis is None:
            auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
    if frozen_sys < 0 or frozen_env < 0:
        raise ValueError('frozen_sys and frozen_env must be non-negative')
    if frozen_sys > data['moC_occ_A'].shape[1]:
        raise ValueError('frozen_sys exceeds the number of system occupied orbitals')
    if frozen_env > data['moC_occ_B'].shape[1]:
        raise ValueError('frozen_env exceeds the number of bath occupied orbitals')
    log = logger.new_logger(mf, verbose)

    moC_occ_A_corr = data['moC_occ_A'][:, frozen_sys:]
    moC_occ_B_corr = data['moC_occ_B'][:, frozen_env:]
    moE_occ_A_corr = data['moE_occ_A'][frozen_sys:]
    moE_occ_B_corr = data['moE_occ_B'][frozen_env:]

    c_occ = np.hstack([moC_occ_A_corr, moC_occ_B_corr])
    c_vir = np.hstack([data['moC_vir_A'], data['moC_vir_B']])
    e_occ = np.hstack([moE_occ_A_corr, moE_occ_B_corr])
    e_vir = np.hstack([data['moE_vir_A'], data['moE_vir_B']])
    occ_labels = np.array(
        ['A'] * moC_occ_A_corr.shape[1]
        + ['B'] * moC_occ_B_corr.shape[1]
    )
    vir_labels = np.array(
        ['A'] * data['moC_vir_A'].shape[1]
        + ['B'] * data['moC_vir_B'].shape[1]
    )

    log.info('')
    log.info('******** restricted PRBE MP2 diagnostics ********')
    log.info('DFT bath reference = %s', data['dft_bath'])
    log.info('nocc_A = %d  nvir_A = %d  nocc_B = %d  nvir_B = %d',
             data['moC_occ_A'].shape[1], data['moC_vir_A'].shape[1],
             data['moC_occ_B'].shape[1], data['moC_vir_B'].shape[1])

    singles_enabled = data['dft_bath']
    c_occ_ref = np.hstack([data['moC_occ_A'], data['moC_occ_B']])
    dm_singles = 2.0 * c_occ_ref @ c_occ_ref.T
    mf_hf = scf.RHF(mf.mol)
    with_df = getattr(mf, 'with_df', None)
    if with_df is not None:
        mf_hf = df.density_fit(mf_hf)
        mf_hf.with_df.auxbasis = getattr(with_df, 'auxbasis', None)
    fock_singles = mf_hf.get_hcore(mf.mol) + mf_hf.get_veff(mf.mol, dm_singles)
    singles = _singles_decomp(
        fock_singles, c_occ, c_vir, e_occ, e_vir,
        occ_labels, vir_labels, singles_enabled
    )

    doubles_out = _doubles_decomp(
        mf, c_occ, c_vir, e_occ, e_vir, occ_labels, vir_labels,
        auxbasis, max_memory, log, outcore=outcore
    )
    combined = doubles_out['combined']
    direct = doubles_out['direct']

    e_os = direct['total']                  # opposite-spin / direct
    e_ss = combined['total'] - e_os         # same-spin remainder

    result = {
        'dft_bath': data['dft_bath'],

        # Set 1: combined (direct + exchange) -- full RMP2
        'doubles': combined,                       # back-compat alias
        'doubles_combined': combined,
        'doubles_by_group': combined['by_group'],  # back-compat alias
        'doubles_combined_by_group': combined['by_group'],
        'doubles_total': combined['total'],
        'doubles_cross_excl_bb_vir': combined['cross_excl_bb_vir'],

        # Set 2: OS / direct only -- (ia|jb)^2 / D
        'doubles_os': direct,
        'doubles_os_by_group': direct['by_group'],
        'os_total': e_os,
        'ss_total': e_ss,
        'doubles_os_cross_excl_bb_vir': direct['cross_excl_bb_vir'],

        # singles
        'singles': singles,
        'singles_by_group': singles['by_group'],
        'singles_total': singles['total'],

        # totals
        'total': combined['total'] + singles['total'],
        'system_only': combined['system_only'] + singles['system_only'],
        'bath_only': combined['bath_only'] + singles['bath_only'],
        'cross_total': combined['cross_total'] + singles['cross_total'],
        'composite_missing': (
            combined['composite_missing'] + singles['composite_missing']
        ),
        'cross_by_num_A': combined['cross_by_num_A'],
    }

    log.info('MP2 doubles total      = %.15g', result['doubles_total'])
    log.info('MP2 singles total      = %.15g', result['singles_total'])
    log.info('MP2 total              = %.15g', result['total'])
    log.info('MP2 E_OS (direct)      = %.15g', e_os)
    log.info('MP2 E_SS (combined-OS) = %.15g', e_ss)
    log.info('MP2 system-only        = %.15g', result['system_only'])
    log.info('MP2 bath-only          = %.15g', result['bath_only'])
    log.info('MP2 cross total        = %.15g', result['cross_total'])
    log.info('MP2 bath + cross       = %.15g', result['composite_missing'])
    log.info('MP2 cross (excl B-B vir)    = %.15g',
             result['doubles_cross_excl_bb_vir'])
    log.info('MP2 OS cross (excl B-B vir) = %.15g',
             result['doubles_os_cross_excl_bb_vir'])

    log.info('')
    log.info('--- Set 1: combined direct+exchange (full RMP2 doubles), net-charge buckets ---')
    for key in DOUBLES_GROUPS:
        log.info('  %-14s = %.15g', key, combined['by_group'].get(key, 0.0))

    log.info('')
    log.info('--- Set 2: OS / direct only, (ia|jb)^2/D, net-charge buckets ---')
    for key in DOUBLES_GROUPS:
        log.info('  %-14s = %.15g', key, direct['by_group'].get(key, 0.0))

    log.info('')
    log.info('--- per-pattern (ijab string / (ia|jb) integral / group) ---')
    log.info('  %-6s %-10s %-14s %16s %16s',
             'ijab', '(ia|jb)', 'group', 'combined', 'direct(OS)')
    for pattern in sorted(combined['by_pattern']):
        log.info('  %-6s %-10s %-14s %16.10g %16.10g',
                 pattern, _integral_label(pattern), _classify_pattern(pattern),
                 combined['by_pattern'][pattern],
                 direct['by_pattern'].get(pattern, 0.0))

    return result


class rMP2Decomposition:
    """Object wrapper for PRBE MP2 decomposition diagnostics."""

    def __init__(self, prbe, auxbasis=None, max_memory=None,
                 frozen_sys=0, frozen_env=0, outcore=False):
        self.prbe = prbe
        self.auxbasis = auxbasis
        self.max_memory = max_memory
        self.frozen_sys = frozen_sys
        self.frozen_env = frozen_env
        self.outcore = outcore
        self.results = None

    def kernel(self):
        self.results = mp2_decomp_driver(
            self.prbe,
            auxbasis=self.auxbasis,
            max_memory=self.max_memory,
            frozen_sys=self.frozen_sys,
            frozen_env=self.frozen_env,
            outcore=self.outcore,
        )
        return self.results

    run = kernel
#%%
if __name__ == '__main__':
    from pyscf_embedding.local import regional as re
    from pyscf_embedding.utils import chemcore
    from pyscf import gto, scf, cc, grad
    from pyscf_embedding.dft.vprbe import rPRBE, uPRBE
    from pyscf_embedding.dft.vprbe import rVPRBE, uVPRBE

    coords1 = \
    """
    H           1.83033035325714     -1.55914459515730      0.85018703157270
    C           0.90263631613210     -0.08578242736390      0.14894116637203
    O           1.92201300538061      0.51665324420946     -0.08503290170897
    O           0.90171075857076     -1.32072671224472      0.70739805133646
    C          -0.44742060897905      0.43887835522198     -0.13930696034172
    C          -1.57631378519914     -0.21501349238018      0.11262246365465
    H          -0.44354482529961      1.42537195115242     -0.58435608510154
    H          -1.57016561414080     -1.20100172981207      0.55763852778426
    H          -2.53835159972200      0.22238440637431     -0.12155029356787
    """

    coords2 = \
    """
      C           0.93501939826289     -0.16731316159968      0.18715060054646
      O           1.88694577197618      0.59099117395221     -0.12029735496945
      O           0.96364675571253     -1.30644525101509      0.70374527602260
      C          -0.45331566554332      0.41330164264060     -0.12783548248701
      C          -1.60070946799917     -0.21426796566953      0.11128233993953
      H          -0.44856062322574      1.40525848237891     -0.57525090759213
      H          -1.57811875566373     -1.20278092786669      0.55806286649746
      H          -2.56689441351964      0.22480300717926     -0.12390333795747
    """


    #%%
    ##### Calculation One, Test rPRBE
    basis='ccpvdz'
    xc = 'pbe'
    #%%
    # run CCSD(T)

    mol1 = gto.Mole(atom=coords1,basis=basis,spin=0,verbose=4)
    mf_hf1 = mol1.RHF(max_cycles=100).density_fit().run()
    mycc1 = cc.CCSD(mf_hf1,frozen=chemcore(mol1)).density_fit()
    mycc1 = mycc1.run()
    et = mycc1.ccsd_t()
    ecc1 = mycc1.e_tot + et
    del mycc1

    mol2 = gto.Mole(atom=coords2,basis=basis,spin=0,verbose=4,charge=-1)
    mf_hf2 = mol2.RHF(max_cycles=100).density_fit().run()
    mycc2 = cc.CCSD(mf_hf2,frozen=chemcore(mol2)).density_fit()
    mycc2 = mycc2.run()
    et = mycc2.ccsd_t()
    ecc2 = mycc2.e_tot + et

    del mycc2
    #%%
    # run DFT
    mol1 = gto.Mole(atom=coords1,basis=basis,spin=0,verbose=4)
    mf1 = mol1.RKS(xc=xc,max_cycles=100).density_fit().run()
    edft1 = mf1.e_tot

    mol2 = gto.Mole(atom=coords2,basis=basis,spin=0,verbose=4,charge=-1)
    mf2 = mol2.RKS(xc=xc,max_cycles=100).density_fit().run()
    edft2 = mf2.e_tot

    #%%
    ##### standard PRBE

    # run embedding 1
    occ_calc = re.RegionalActiveSpace(mf1, np.arange(0,4), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf1, np.arange(0,4), 'vir', basis=mol1.basis, cutoff=1e-6)
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed1 = rPRBE(mf1, moC_occ, moC_vir, mask_occ_act, mask_vir_act)
    e_mf,mo_energy,mo_coeff,mo_occ = embed1.kernel(xc_embed=None)
    mycc1 = cc.CCSD(embed1.mf_A)
    mycc1.kernel()
    e_embed1 = e_mf + mycc1.e_corr

    # run embedding 2
    occ_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'occ', basis='minao', cutoff=0.25)
    vir_calc = re.RegionalActiveSpace(mf2, np.arange(0,3), 'vir', basis=mol1.basis, cutoff=1e-6)
    _,moC_occ,mask_occ_act = occ_calc.calc_mo()
    _,moC_vir,mask_vir_act = vir_calc.calc_mo()

    embed2 = rPRBE(mf2, moC_occ, moC_vir, mask_occ_act, mask_vir_act)
    e_mf,mo_energy,mo_coeff,mo_occ = embed2.kernel(xc_embed=None)
    mycc2 = cc.CCSD(embed2.mf_A)
    mycc2.kernel()
    e_embed2 = e_mf + mycc2.e_corr

    print((ecc2-ecc1)*627.5095, (edft2-edft1)*627.5095,
          (e_embed2-e_embed1)*627.5095)
    #%%
    ##### standard vPRBE

    # # run embedding 1
    # mp_1 = rCrossMP2(embed1,recanonicalize_bath=True)
    # mp_1.kernel()
    # e_mpembed1 =e_embed1 - mp_1.e_corr

    # mp_2 = rCrossMP2(embed2,recanonicalize_bath=True)
    # mp_2.kernel()
    # e_mpembed2 =e_embed2 - mp_2.e_corr
    # print((ecc2-ecc1)*627.5095, (edft2-edft1)*627.5095,
    #       (e_mpembed2-e_mpembed1)*627.5095, (e_embed2-e_embed1)*627.5095)
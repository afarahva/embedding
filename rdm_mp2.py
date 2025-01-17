import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.mp.mp2 import _mo_splitter, make_rdm1, _make_eris
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

def make_fno(mp, dm=None, thresh=1e-6, pct_occ=None, nvir_act=None, t2=None, eris=None, oN4=False):
    r'''
    Frozen natural orbitals. 
    
    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-6 (very conservative).
        pct_occ : float
            Percentage of total occupation number. Default is None. If present, overrides `thresh`.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh` and `pct_occ`.
        oN4 : bool
            Whether to use N4 scaling algorithm for mp2 reduced density matrix

    Returns:
        frozen : list or ndarray
            List of orbitals to freeze
        no_coeff : ndarray
            Semicanonical NO coefficients in the AO basis
    '''
    
    mf = mp._scf
    if dm is None:
        if oN4:
            dm = _make_rdm1_oN4(mp, t2=t2, eris=eris, with_frozen=False)
        else:
            dm = make_rdm1(mp,t2=t2, eris=eris, with_frozen=False)

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    n,v = numpy.linalg.eigh(dm[nocc:,nocc:])
    idx = numpy.argsort(n)[::-1]
    n,v = n[idx], v[:,idx]

    if nvir_act is None:
        if pct_occ is None:
            nvir_keep = numpy.count_nonzero(n>thresh)
        else:
            cumsum = numpy.cumsum(n/numpy.sum(n))
            lib.logger.debug(mp, 'Sum(pct_occ): %s', cumsum)
            nvir_keep = numpy.count_nonzero(cumsum<pct_occ)
    else:
        nvir_keep = min(nvir, nvir_act)

    masks = _mo_splitter(mp)
    moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[m] for m in masks]
    orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[:,m] for m in masks]

    fvv = numpy.diag(moevir)
    fvv_no = numpy.dot(v.T, numpy.dot(fvv, v))
    _, v_canon = numpy.linalg.eigh(fvv_no[:nvir_keep,:nvir_keep])

    orbviract = numpy.dot(orbvir, numpy.dot(v[:,:nvir_keep], v_canon))
    orbvirfrz = numpy.dot(orbvir, v[:,nvir_keep:])
    no_comp = (orboccfrz0, orbocc, orbviract, orbvirfrz, orbvirfrz0)
    no_coeff = numpy.hstack(no_comp)
    nocc_loc = numpy.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
    no_frozen = numpy.hstack((numpy.arange(nocc_loc[0], nocc_loc[1]),
                              numpy.arange(nocc_loc[3], nocc_loc[5]))).astype(int)

    return no_frozen, no_coeff  

def _make_rdm1_oN4(mp, t2=None, eris=None, ao_repr=False, with_frozen=True, with_occupied=False):
    r'''
    Approximate virtual-virtual block of spin-traced one-particle density matrix. 
    The ov response is not included, the oo response can be included with the flag 'with_occupied=True'
    
    The virtual-virtual block of the one-particle density matrix is given by:
    D_{ab} = \sum_{cij}  [2*<cb | ij > < ij | ca> - < cb | ji > <ij | ca>] / [ (e_i + e_j - e_c - e_b) * (e_i + e_j - e_c - e_a)]
    where i,j go over occupied orbitals and a,b,c go over virtual orbitals. This has N^5 scaling
    
    We can approximate this expression as:
    D_{ab} = \sum_{ci}  <cb | ii > < ii | ca> / [ (2*e_i - e_c - e_b) * (2*e_i - e_c - e_a)]
    which has N^4 scaling.
    
    Ref: Gruneis et al. JCTC (2011), https://pubs.acs.org/doi/10.1021/ct200263g
    '''
    
    # get oo and vv blocks of rdm1
    doo, dvv = _gamma1_intermediates_N4(mp, t2, eris, with_occupied=with_occupied)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = numpy.zeros((nocc,nvir), dtype=doo.dtype)
    dvo = dov.T

    # make full rdm1 from blocks
    nocc, nvir = dov.shape
    nmo = nocc + nvir
    dm1 = numpy.empty((nmo,nmo), dtype=doo.dtype)
    dm1[:nocc,:nocc] = doo + doo.conj().T
    dm1[:nocc,nocc:] = dov + dvo.conj().T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].conj().T
    dm1[nocc:,nocc:] = dvv + dvv.conj().T

    # frozen orbitals
    if with_frozen and mp.frozen is not None:
        nmo = mp.mo_occ.size
        nocc = numpy.count_nonzero(mp.mo_occ > 0)
        rdm1 = numpy.zeros((nmo,nmo), dtype=dm1.dtype)
        moidx = numpy.where(mp.get_frozen_mask())[0]
        rdm1[moidx[:,None],moidx] = dm1
        dm1 = rdm1

    # change to ao basis
    if ao_repr:
        mo = mp.mo_coeff
        dm1 = lib.einsum('pi,ij,qj->pq', mo, dm1, mo.conj())
    
    return dm1


def _gamma1_intermediates_N4(mp, t2=None, eris=None, with_occupied=False):
    if t2 is None: t2 = mp.t2
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    
    if t2 is None:
        if eris is None:
            eris = _make_eris(mp, mp._scf.mo_coeff, verbose=mp.verbose)
        mo_energy = eris.mo_energy
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
        dtype = eris.ovov.dtype
    else:
        dtype = t2.dtype

    dm1occ = numpy.zeros((nocc,nocc), dtype=dtype)
    dm1vir = numpy.zeros((nvir,nvir), dtype=dtype)

    for i in range(nocc):
        # if occupied-occupied block requested
        if with_occupied:
            if t2 is None:
                gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
                gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
                t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
            else:
                t2i = t2[i]
            l2i = t2i.conj()
            dm1vir += lib.einsum('ca,cb->ba', l2i[i], t2i[i]) * 2 \
                    - lib.einsum('ca,bc->ba', l2i[i], t2i[i])
            dm1occ += lib.einsum('iab,jab->ij', l2i, t2i) * 2 \
                    - lib.einsum('iab,jba->ij', l2i, t2i)
                    
        # if only v-v block requested
        else:
            if t2 is None:
                gii = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir,i*nvir:(i+1)*nvir])
                t2ii = gii.conj()/lib.direct_sum('b+a->ba', eia[i], eia[i])
            else:
                t2ii = t2[i,i]
            l2ii = t2ii.conj()
            dm1vir += lib.einsum('ca,cb->ba', l2ii, t2ii) * 2 \
                    - lib.einsum('ca,bc->ba', l2ii, t2ii)
                
    return -dm1occ, dm1vir

def _ao2mo_ovov(mp, orbo, orbv, feri, max_memory=2000, verbose=None):
    from pyscf.scf.hf import RHF
    assert isinstance(mp._scf, RHF)
    time0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    mol = mp.mol
    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]
    nbas = mol.nbas
    assert (nvir <= nao)

    ao_loc = mol.ao_loc_nr()
    dmax = max(4, min(nao/3, numpy.sqrt(max_memory*.95e6/8/(nao+nocc)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao))
    ftmp = lib.H5TmpFile()
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, nocc**2*(nao*(nao+dmax)/2+nvir**2)*8/1e6)

    buf_i = numpy.empty((nocc*dmax**2*nao))
    buf_li = numpy.empty((nocc**2*dmax**2))
    buf1 = numpy.empty_like(buf_li)

    fint = gto.moleintor.getints4c
    jk_blk_slices = []
    count = 0
    time1 = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip+1]:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                jk_blk_slices.append((i0,i1,j0,j1))

                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=(0,nbas,ish0,ish1, jsh0,jsh1,0,nbas),
                           aosym='s1', ao_loc=ao_loc, cintopt=ao2mopt._cintopt,
                           out=eribuf)
                tmp_i = numpy.ndarray((nocc,(i1-i0)*(j1-j0)*nao), buffer=buf_i)
                tmp_li = numpy.ndarray((nocc,nocc*(i1-i0)*(j1-j0)), buffer=buf_li)
                lib.ddot(orbo.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao), c=tmp_i)
                lib.ddot(orbo.T, tmp_i.reshape(nocc*(i1-i0)*(j1-j0),nao).T, c=tmp_li)
                tmp_li = tmp_li.reshape(nocc,nocc,(i1-i0),(j1-j0))
                save(str(count), tmp_li.transpose(1,0,2,3))
                buf_li, buf1 = buf1, buf_li
                count += 1
                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = buf_i = buf_li = buf1 = None

    h5dat = feri.create_dataset('ovov', (nocc*nvir,nocc*nvir), 'f8',
                                chunks=(nvir,nvir))
    occblk = int(min(nocc, max(4, 250/nocc, max_memory*.9e6/8/(nao**2*nocc)/5)))
    def load(i0, eri):
        if i0 < nocc:
            i1 = min(i0+occblk, nocc)
            for k, (p0,p1,q0,q1) in enumerate(jk_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = ftmp[str(k)][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(ftmp[str(k)][:,i0:i1])
                    eri[:i1-i0,:,q0:q1,p0:p1] = dat.transpose(1,0,3,2)

    def save(i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i*nvir:(i+1)*nvir] = dat[i-i0].reshape(nvir,nocc*nvir)

    orbv = numpy.asarray(orbv, order='F')
    buf_prefecth = numpy.empty((occblk,nocc,nao,nao))
    buf = numpy.empty_like(buf_prefecth)
    bufw = numpy.empty((occblk*nocc,nvir**2))
    bufw1 = numpy.empty_like(bufw)
    with lib.call_in_background(load) as prefetch:
        with lib.call_in_background(save) as bsave:
            load(0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocc, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(i1, buf_prefecth)
                
                eri = buf[:i1-i0].reshape((i1-i0)*nocc,nao,nao)

                dat = _ao2mo.nr_e2(eri, orbv, (0,nvir,0,nvir), 's1', 's1', out=bufw)
                bsave(i0, i1, dat.reshape(i1-i0,nocc,nvir,nvir).transpose(0,2,1,3))
                bufw, bufw1 = bufw1, bufw
                time1 = log.timer_debug1('pass2 ao2mo [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)
    return h5dat


#%%
if __name__ == '__main__':
    import timeit
    import time
    import pyscf
    from pyscf import cc, mp, scf, gto
    from pyscf import ao2mo
    
    water = [["O" , (0. , 0.     , 0.)],
             ["H" , (0. , -0.757 , 0.587)],
             ["H" , (0. , 0.757  , 0.587)]]


    benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
               [ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
               [ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
               [ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
               [ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
               [ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
               [ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
               [ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
               [ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
               [ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
               [ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
               [ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]

    mol = gto.M(atom=benzene)
    mol.basis = 'cc-pvdz'
    mol.build()
    
    # do hf
    mf = scf.RHF(mol).run()
    
    # precompute eris, can use in all subsequent calls to make_rdm1
    mymp = mp.mp2.MP2(mf)
    eris = _make_eris(mymp, mf.mo_coeff, verbose=mymp.verbose)
    
    ## compare eigenvalues/timings between approximate and full rdm1
    nmo = mymp.nmo
    nocc = mymp.nocc
    nvir = nmo - nocc    
    co = mf.mo_coeff[:,0:nocc]
    cv = mf.mo_coeff[:,nocc:]
    
    # approximate rdm1
    start = time.time()
    dm = _make_rdm1_oN4(mymp,eris=eris,with_occupied=False)
    end = time.time()
    time1 = end-start
    
    dm_vv = dm[nocc:,nocc:]
    n1 = numpy.linalg.eigvalsh(dm_vv[nocc:,nocc:])
    
    # full rdm1
    start = time.time()
    dm = mp.mp2.make_rdm1(mymp,eris=eris)
    end = time.time()
    time2 = end-start
    
    dm_vv = dm[nocc:,nocc:]
    n2 = numpy.linalg.eigvalsh(dm_vv[nocc:,nocc:])
    
    print("\nRDM time N^5 scaling:",time2,", RDM time N^4 scaling:",time1,"\n")

    ## compare natural orbitals
    print("Frozen NOs, Approximate Method")
    no_frozen1, no_coeff1 = make_fno(mymp,eris=eris,pct_occ=0.98,oN4=True)
    print(no_frozen1)
    
    print("Frozen NOs, Full Method ")
    no_frozen2, no_coeff2 = make_fno(mymp,eris=eris,pct_occ=0.98,oN4=False)
    print(no_frozen2)
    
    print("\nMP2/CCSD energy using approximate FNOs")
    pt_no = mp.MP2(mf, frozen=no_frozen1, mo_coeff=no_coeff1).run()
    mycc = cc.CCSD(mf, frozen=no_frozen1, mo_coeff=no_coeff1).run()
    
    print("\nMP2/CCSD energy using full FNOs")
    pt_no = mp.MP2(mf, frozen=no_frozen2, mo_coeff=no_coeff2).run()
    mycc = cc.CCSD(mf, frozen=no_frozen2, mo_coeff=no_coeff2).run()



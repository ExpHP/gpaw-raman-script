# coding: utf-8

# General
import numpy as np
import sparse
import typing as tp
from dataclasses import dataclass
from scipy import signal
from math import pi

# GPAW/ASE
from gpaw import GPAW, PW, FermiDirac
from gpaw.fd_operators import Gradient

from ase.phonons import Phonons
from ase.parallel import parprint

def get_elph_elements(atoms, gpw_name, calc_fd, sc=(1, 1, 1), basename=None, phononname='phonon'):
    """
        Evaluates the dipole transition matrix elements

        Input
        ----------
        params_fd : Calculation parameters used for the phonon calculation
        sc (tuple): Supercell, default is (1,1,1) used for gamma phonons
        basename  : If you want give a specific name (gqklnn_{}.pckl)

        Output
        ----------
        gqklnn.pckl, the electron-phonon matrix elements
    """
    from ase.phonons import Phonons
    from gpaw.elph.electronphonon import ElectronPhononCoupling

    calc_gs = GPAW(gpw_name)
    world = calc_gs.wfs.world

    #calc_fd = GPAW(**params_fd)
    calc_gs.initialize_positions(atoms)
    kpts = calc_gs.get_ibz_k_points()
    nk = len(kpts)
    gamma_kpt = [[0, 0, 0]]
    nbands = calc_gs.wfs.bd.nbands
    qpts = gamma_kpt

    # calc_fd.get_potential_energy()  # XXX needed to initialize C_nM ??????

    # Phonon calculation, We'll read the forces from the elph.run function
    # This only looks at gamma point phonons
    ph = Phonons(atoms=atoms, name=phononname, supercell=sc)
    ph.read()
    frequencies, modes = ph.band_structure(qpts, modes=True)

    if world.rank == 0:
        print("Phonon frequencies are loaded.")

    # Find el-ph matrix in the LCAO basis
    elph = ElectronPhononCoupling(atoms, calc=None, supercell=sc)

    elph.set_lcao_calculator(calc_fd)
    elph.load_supercell_matrix()
    if world.rank == 0:
        print("Supercell matrix is loaded")

    # Non-root processes on GD comm seem to be missing kpoint data.
    assert calc_gs.wfs.gd.comm.size == 1, "domain parallelism not supported"  # not sure how to fix this, sorry

    gcomm = calc_gs.wfs.gd.comm
    kcomm = calc_gs.wfs.kd.comm
    if gcomm.rank == 0:
        # Find the bloch expansion coefficients
        c_kn = np.empty((nk, nbands, calc_gs.wfs.setups.nao), dtype=complex)
        for k in range(calc_gs.wfs.kd.nibzkpts):
            c_k = calc_gs.wfs.collect_array("C_nM", k, 0)
            if kcomm.rank == 0:
                c_kn[k] = c_k
        kcomm.broadcast(c_kn, 0)

        # And we finally find the electron-phonon coupling matrix elements!
        g_qklnn = elph.bloch_matrix(c_kn=c_kn, kpts=kpts, qpts=qpts, u_ql=modes)

    if world.rank == 0:
        print("Saving the elctron-phonon coupling matrix")
        np.save("gqklnn{}.npy".format(make_suffix(basename)), np.array(g_qklnn))

def get_dipole_transitions(calc, momname=None, basename=None):
    """
    Finds the dipole matrix elements:
    <\psi_n|\nabla|\psi_m> = <u_n|nabla|u_m> + ik<u_n|u_m> where psi_n = u_n(r)*exp(ikr).

    Input:
        atoms           Relevant ASE atoms object
        momname         Suffix for the dipole transition file
        basename        Suffix used for the gs.gpw file

    Output:
        dip_vknm.npy    Array with dipole matrix elements
    """

    # par = MPI4PY()  # FIXME: use a comm from gpaw
    #calc = atoms.calc

    bzk_kc = calc.get_ibz_k_points()
    n = calc.wfs.bd.nbands
    nk = np.shape(bzk_kc)[0]

    wfs = {}

    parprint("Distributing wavefunctions.")

    kcomm = calc.wfs.kd.comm
    world = calc.wfs.world
    if not calc.wfs.positions_set:
        calc.initialize_positions()
    for k in range(nk):
        # Collects the wavefunctions and the projections to rank 0. Periodic -> u_n(r)
        spin = 0 # FIXME
        wf = np.array([calc.wfs.get_wave_function_array(
            i, k, spin, realspace=True, periodic=True) for i in range(n)], dtype=complex)
        P_nI = calc.wfs.collect_projections(k, spin)

        # Distributes the information to rank k % size.
        if kcomm.rank == 0:
            if k % kcomm.size == kcomm.rank:
                wfs[k] = wf, P_nI
            else:
                kcomm.send(P_nI, dest=k % kcomm.size, tag=nk+k)
                kcomm.send(wf, dest=k % kcomm.size, tag=k)
        else:
            if k % kcomm.size == kcomm.rank:
                nproj = sum(setup.ni for setup in calc.wfs.setups)
                if not calc.wfs.collinear:
                    nproj *= 2
                P_nI = np.empty((calc.wfs.bd.nbands, nproj), calc.wfs.dtype)
                shape = () if calc.wfs.collinear else(2,)
                wf = np.tile(calc.wfs.empty(
                    shape, global_array=True, realspace=True), (n, 1, 1, 1))

                kcomm.receive(P_nI, src=0, tag=nk + k)
                kcomm.receive(wf, src=0, tag=k)

                wfs[k] = wf, P_nI

    parprint("Evaluating dipole transition matrix elements.")

    dip_vknm = np.zeros((3, nk, n, n), dtype=complex)
    overlap_knm = np.zeros((nk, n, n), dtype=complex)

    nabla_v = [Gradient(calc.wfs.gd, v, 1.0, 4,
                        complex).apply for v in range(3)]
    phases = np.ones((3, 2), dtype=complex)
    grad_nv = calc.wfs.gd.zeros((n, 3), complex)

    for k, (wf, P_nI) in wfs.items():
        # Calculate <phit|nabla|phit> for the pseudo wavefunction
        for v in range(3):
            for i in range(n):
                nabla_v[v](wf[i], grad_nv[i, v], phases)

        dip_vknm[:, k] = np.transpose(
            calc.wfs.gd.integrate(wf, grad_nv), (2, 0, 1))

        overlap_knm[k] = [calc.wfs.gd.integrate(wf[i], wf) for i in range(n)]
        k_v = np.dot(calc.wfs.kd.ibzk_kc[k], calc.wfs.gd.icell_cv) * 2 * pi
        dip_vknm[:, k] += 1j*k_v[:, None, None]*overlap_knm[None, k, :, :]

        # The PAW corrections are added - see https://wiki.fysik.dtu.dk/gpaw/dev/documentation/tddft/dielectric_response.html#paw-terms
        I1 = 0
        # np.einsum is slow but very memory efficient.
        for a, setup in enumerate(calc.wfs.setups):
            I2 = I1 + setup.ni
            P_ni = P_nI[:, I1:I2]
            dip_vknm[:, k, :, :] += np.einsum('ni,ijv,mj->vnm',
                                              P_ni.conj(), setup.nabla_iiv, P_ni)
            I1 = I2

    world.sum(dip_vknm)

    if world.rank == 0:
        np.save('dip_vknm{}.npy'.format(make_suffix(momname)), dip_vknm)


def L(w, gamma=10/8065.544):
    # Lorentzian
    lor = 0.5*gamma/(pi*((w.real)**2+0.25*gamma**2))
    return lor

def gaussian(w, sigma=3/8065.544):
    return (sigma * (2*pi)**0.5) ** -1 * np.exp(-w**2 / (2 * sigma**2))

def make_suffix(s):
    if s is None:
        return ''
    else:
        return '_' + s

TermId = tp.Literal['spl', 'slp', 'psl', 'pls', 'lsp', 'lps']

@dataclass
class RamanOutput:
    raman_lw: np.ndarray
    contributions_lktnnn_parts: tp.Optional[tp.List[sparse.GCXS]]
    terms_t: tp.List[TermId]
    nphonons: int
    nibzkpts: int
    nbands: int

    def term_index_from_str(self, term: str):
        return self.terms_t.index(term)

def calculate_raman(
    calc,
    w_ph,
    permutations='original',
    w_cm=None,
    ramanname=None,
    momname=None,
    basename=None,
    w_l=2.54066,
    gamma_l=0.2,
    d_i=0,
    d_o=0,
    shift_step=1,
    shift_type='stokes',
    phonon_sigma=3,
    write_mode_intensities=False,
    write_contributions=False,
):
    """
    Calculates the first order Raman spectre

    Input:
        w_ph            Gamma phonon energies in eV.
        permutations    Use only resonant term (None) or all terms ('original' or 'fast')
        ramanname       Suffix for the raman.npy file
        momname         Suffix for the momentumfile
        basename        Suffix for the gqklnn.npy files
        w_cm            Raman shift frequencies to compute at.
        w_l, gamma_l    Laser energy, broadening factor for the electron energies
        d_i, d_o        Laser polarization in, out (0, 1, 2 for x, y, z respectively)
    Output:
        RI.npy          Numpy array containing the raman spectre
    """

    assert permutations in [None, 'fast', 'original']
    assert shift_type in ['stokes', 'anti-stokes']

    parprint("Calculating the Raman spectra: Laser frequency = {}".format(w_l))

    bzk_kc = calc.get_ibz_k_points()
    nbands = calc.wfs.bd.nbands
    nibzkpts = np.shape(bzk_kc)[0]
    cm = 1/8065.544

    if w_cm is None:
        w_cm = np.arange(0, int(w_ph.max()/cm) + 201, shift_step) * 1.0  # Defined in cm^-1
    w_shift = w_cm*cm

    if shift_type == 'stokes':
        w_scatter = w_l - w_shift
    elif shift_type == 'anti-stokes':
        w_scatter = w_l + w_shift
    nphonons = len(w_ph)

    assert calc.wfs.gd.comm.size == 1, "domain parallelism not supported"  # not sure how to fix this, sorry

    # ab is in and out polarization
    # l is the phonon mode and w is the raman shift
    output = RamanOutput(
        raman_lw = np.zeros((nphonons, len(w_shift)), dtype=complex),
        contributions_lktnnn_parts = [] if write_contributions else None,
        nphonons = nphonons,
        nibzkpts = nibzkpts,
        nbands = nbands,
        # ordering found in Dresselhaus
        terms_t = ['lps', 'lsp', 'spl', 'slp', 'pls', 'psl']
    )

    parprint("Evaluating Raman sum")

    kcomm = calc.wfs.kd.comm
    world = calc.wfs.world
    if kcomm.rank == 0:
        mom_vknn = np.load("dip_vknm{}.npy".format(make_suffix(momname)))  # [:,k,:,:]dim, k
        elph_klnn = np.load("gqklnn{}.npy".format(make_suffix(basename)))[0]  # [0,k,l,n,m]
    else:
        mom_vknn = None
        elph_klnn = None

    for info_at_k in _distribute_bands_by_k(calc, mom_vknn, elph_klnn, nphonons):
        print("For k = {}".format(info_at_k.k_index))
        _add_raman_terms_at_k(output, w_l, gamma_l, (d_i, d_o), w_ph, w_scatter, info_at_k, shift_type, permutations)

    kcomm.sum(output.raman_lw)

    contributions_lktnnn = None
    if output.contributions_lktnnn_parts is not None:
        contributions_lktnnn = _sum_sparse_coo(output.contributions_lktnnn_parts)
        _mpi_sum_sparse_coo(contributions_lktnnn, kcomm)

    if write_mode_intensities:
        # write values without the gaussian on shift
        if world.rank == 0:
            np.save("ModeI{}.npy".format(make_suffix(ramanname)), output.raman_lw[:, 0])

    if contributions_lktnnn is not None:
        if world.rank == 0:
            _write_raman_contributions(calc, contributions_lktnnn, output.terms_t, "Contrib{}.npz".format(make_suffix(ramanname)))

    RI = np.zeros(len(w_shift))
    for l in range(nphonons):
        if w_ph[l].real >= 0:
            parprint(
                "Phonon {} with energy = {} registered".format(l, w_ph[l]))
            RI += (np.abs(output.raman_lw[l])**2)*np.array(gaussian(w_shift-w_ph[l], sigma=phonon_sigma * cm))

    raman = np.vstack((w_cm, RI))
    if world.rank == 0:
        np.save("RI{}.npy".format(make_suffix(ramanname)), raman)

def _mpi_sum_sparse_coo(arr: sparse.COO, comm):
    """ Sum a sparse.COO array onto rank 0 of an MPI communicator. """
    if comm.rank == 0:
        coords_parts = [arr.coords]
        data_parts = [arr.data]
        for rank in range(1, comm.size):
            this_nnz = np.empty((), dtype=int)
            comm.receive(this_nnz, src=rank, tag=rank)

            this_coords = np.empty([arr.coords.shape[0], this_nnz], dtype=int)
            comm.receive(this_coords, src=rank, tag=comm.size + rank)
            coords_parts.append(this_coords)

            this_data = np.empty([this_nnz], dtype=arr.dtype)
            comm.receive(this_data, src=rank, tag=2*comm.size + rank)
            data_parts.append(this_data)

        return _sum_sparse_coo_impl(coords_parts, data_parts, shape=arr.shape)

    else:
        rank = comm.rank
        comm.send(np.array(arr.data.shape[0], dtype=int), dest=0, tag=rank)
        comm.send(np.array(arr.coords, dtype=int, order='C'), dest=0, tag=comm.size + rank)
        comm.send(np.array(arr.data, dtype=complex, order='C'), dest=0, tag=2*comm.size + rank)

def _sum_sparse_coo(arrs: tp.Iterable[sparse.COO]):
    """ Efficiently sum a large number of sparse COO arrays.

    This is faster than using the + operator, which would repeatedly sort the data. """
    arrs = list(arrs)
    if not arrs:
        raise ValueError("cannot sum no arrays")  # can't infer shape from no arrays!
    if not all(arr.shape == arrs[0].shape for arr in arrs):
        raise ValueError("shapes do not match: {}".format(set(arr.shape for arr in arrs)))

    coords_parts = [arr.coords for arr in arrs]
    data_parts = [arr.data for arr in arrs]
    return _sum_sparse_coo_impl(coords_parts, data_parts, shape=arrs[0].shape)

def _sum_sparse_coo_impl(coords_parts: tp.Iterable[np.ndarray], data_parts: tp.Iterable[np.ndarray], shape: tuple):
    """ Efficiently sum a large number of sparse COO arrays, given their parts. """
    all_coords = np.concatenate(list(coords_parts), axis=1)
    all_data = np.concatenate(list(data_parts), axis=0)
    return sparse.COO(all_coords, all_data, shape=shape)

def _distribute_bands_by_k(calc, mom_vknn, elph_klnn, nphonons):
    nibzkpts = np.shape(calc.get_ibz_k_points())[0]
    nbands = calc.wfs.bd.nbands
    kcomm = calc.wfs.kd.comm
    E_kn = calc.band_structure().todict()["energies"][0]

    out = []
    parprint("Distributing coupling terms")
    for k in range(nibzkpts):
        is_mine = k % kcomm.size == kcomm.rank
        weight = calc.wfs.collect_auxiliary("weight", k, 0)
        f_n = calc.wfs.collect_occupations(k, 0)

        if kcomm.rank == 0:
            # FIXME: why is this / weight?!  Is f_n not what it seems to be?
            f_n = np.array(f_n / weight, dtype=float)
            elph_lnn = weight * np.array(elph_klnn[k], dtype=complex)
            mom_vnn = np.array(mom_vknn[:, k], dtype=complex)
            if is_mine:
                out.append(_InfoAtK(
                    k_index=k,
                    band_energy_n=E_kn[k],
                    band_occupation_n=f_n,
                    band_momentum_vnn=mom_vnn,
                    band_elph_lnn=elph_lnn,
                ))
            else:
                kcomm.send(elph_lnn, dest=k % kcomm.size, tag=k)
                kcomm.send(mom_vnn, dest=k % kcomm.size, tag=nibzkpts + k)
                kcomm.send(f_n, dest=k % kcomm.size, tag=2*nibzkpts + k)
        else:
            if is_mine:
                elph_lnn = np.empty((nphonons, nbands, nbands), dtype=complex)
                mom_vnn = np.empty((3, nbands, nbands), dtype=complex)
                f_n = np.empty(nbands, dtype=float)

                kcomm.receive(elph_lnn, src=0, tag=k)
                kcomm.receive(mom_vnn, src=0, tag=nibzkpts + k)
                kcomm.receive(f_n, src=0, tag=2*nibzkpts + k)
                out.append(_InfoAtK(
                    k_index=k,
                    band_energy_n=E_kn[k],
                    band_occupation_n=f_n,
                    band_momentum_vnn=mom_vnn,
                    band_elph_lnn=elph_lnn,
                ))
    return out

def _write_raman_contributions(calc, contributions_lktnnn: sparse.COO, terms_t: tp.List[TermId], outpath):
    np.savez_compressed(
        outpath,
        k_weight=calc.get_k_point_weights(),
        k_coords=calc.get_ibz_k_points(),
        num_phonons=contributions_lktnnn.shape[0],
        num_ibzkpoints=contributions_lktnnn.shape[1],
        num_terms=contributions_lktnnn.shape[2],
        num_bands=contributions_lktnnn.shape[3],
        contrib_phonon=contributions_lktnnn.coords[0, :],
        contrib_band_k=contributions_lktnnn.coords[1, :],
        contrib_term=contributions_lktnnn.coords[2, :],
        contrib_band_1=contributions_lktnnn.coords[3, :],
        contrib_band_2=contributions_lktnnn.coords[4, :],
        contrib_band_3=contributions_lktnnn.coords[5, :],
        contrib_value=contributions_lktnnn.data,
        term_str=terms_t,
    )

@dataclass
class _InfoAtK:
    k_index: int
    band_occupation_n: np.ndarray
    band_energy_n: np.ndarray
    band_momentum_vnn: np.ndarray
    band_elph_lnn: np.ndarray

def _add_raman_terms_at_k(
    output: RamanOutput,
    w_l,
    gamma_l,
    polarizations,
    w_ph,
    w_s,
    info_at_k,
    shift_type: tp.Literal['stokes', 'anti-stokes'],
    permutations: tp.Literal[None, 'fast', 'original'],
):
    assert permutations in [None, 'fast', 'original']
    d_i, d_o = polarizations
    k = info_at_k.k_index
    E_el = info_at_k.band_energy_n
    f_n = info_at_k.band_occupation_n
    mom = info_at_k.band_momentum_vnn
    elph = info_at_k.band_elph_lnn

    # This is a refactoring of some code by Ulrik Leffers in https://gitlab.com/gpaw/gpaw/-/merge_requests/563,
    # which appears to be an implementation of Equation 10 in https://www.nature.com/articles/s41467-020-16529-6
    # (though it most certainly does not map 1-1 to the symbols in that equation).
    #
    # Third-order perturbation theory produces six terms based on the ordering of three events:
    # light absorption, phonon creation, light emission.
    # In the original code, each term manifested as a tensor product over three tensors.  Each of these
    # tensors took on one of three forms depending on which event it represented (though this was somewhat
    # obfuscated by arbitrary differences in how some of the denominators were written, or in the ordering
    # of arguments to einsum).
    #
    # We will start by factoring out these tensors.
    #
    # But first: Some parts common to many of the tensors.
    Ediff_el = E_el[None,:]-E_el[:,None]  # antisymmetric tensor that shows up in all denominators
    occu1 = f_n[:,None] * (1-f_n[None,:])  # occupation-based part that always appears in the 1st tensor
    # FIXME: didn't I conclude earler that this should have f_n[None,:]?
    #        (NOTE: be careful!!!  f_n is divided by symmetry weight...)
    occu3 = (1-f_n[:,None]) * np.ones((1, len(f_n)))  # occupation-based part that always appears in the 3rd tensor

    def cannot_compute(**_kw):
        assert False, '(bug) a function was unexpectedly called with permutations=None'

    # Anti-stokes simply flips the sign of w_ph wherever it appears in the denominators.
    # We can represent this with an "effective" w_ph.
    w_ph_eff = {
        'stokes': w_ph,
        'anti-stokes': -w_ph,
    }[shift_type]

    # In the code by Leffers, some denominators had explicit dependence on the scattered frequency,
    # making them significantly more expensive to compute.
    #
    # We suspect that this is unnecessary, allowing a much faster computation.
    get_w_s = {
        'original': lambda w: w_s[w],
        'fast': lambda l: w_l - w_ph_eff[l],
        None: cannot_compute,
    }[permutations]

    # There may be many bands that are fully occupied or unoccupied and therefore incapable of appearing
    # in one or more of the axes that we sum over.  Computing these elements is a waste of time.
    #
    # Define three lambdas that each mask a (nbands,nbands) matrix to only have bands appropriate in that position.
    not0 = abs(f_n) > 1e-20
    not1 = f_n != 1
    mask1 = lambda mat: mat[not0][:, not1]
    mask2 = lambda mat: mat[not1][:, not1]
    mask3 = lambda mat: mat[not1][:, not0]
    valence_indices = np.nonzero(not0)[0]
    conduction_indices = np.nonzero(not1)[0]

    # And now, the 9 tensors.
    #
    # In the original code, some of these tensors were VERY LARGE;  over 50 GB for 17-agnr.
    # Thus, to reduce memory requirements, I have rewritten them to not include axes for the phonon mode or
    # raman shift;  Instead they are all lambdas that produce a matrix with two band axes, and we'll
    # evaluate them at a single phonon/raman shift at a time.
    f1_in_ = lambda: mask1(occu1) * mask1(mom[d_i]) / (w_l-mask1(Ediff_el) + 1j*gamma_l)
    f1_elph_ = lambda l: mask1(occu1) * mask1(elph[l]) / (-w_ph_eff[l]-mask1(Ediff_el) + 1j*gamma_l)
    f1_out_ = {
        'original': lambda w: mask1(occu1) * mask1(mom[d_o]) / (-get_w_s(w=w)-mask1(Ediff_el) + 1j*gamma_l),
        'fast': lambda l: mask1(occu1) * mask1(mom[d_o]) / (-get_w_s(l=l)-mask1(Ediff_el) + 1j*gamma_l),
        None: cannot_compute,
    }[permutations]

    f2_in_ = lambda: mask2(mom[d_i])
    f2_elph_ = lambda l: mask2(elph[l])
    f2_out_ = lambda: mask2(mom[d_o])

    f3_in_ = {
        'original': lambda w, l: mask3(occu3) * mask3(mom[d_i]) / (-get_w_s(w=w)-w_ph_eff[l]-mask3(Ediff_el.T) + 1j*gamma_l),
        'fast': lambda l: mask3(occu3) * mask3(mom[d_i]) / (-w_l-mask3(Ediff_el.T) + 1j*gamma_l),
        None: cannot_compute,
    }[permutations]
    f3_elph_ = {
        'original': lambda w, l: mask3(occu3) * mask3(elph[l]) / (w_l-get_w_s(w=w)-mask3(Ediff_el.T) + 1j*gamma_l),
        'fast': lambda l: mask3(occu3) * mask3(elph[l]) / (w_ph_eff[l]-mask3(Ediff_el.T) + 1j*gamma_l),
        None: cannot_compute,
    }[permutations]
    f3_out_ = lambda l: mask3(occu3) * mask3(mom[d_o]) / (w_l-w_ph_eff[l]-mask3(Ediff_el.T) + 1j*gamma_l)

    add_term = lambda fac1_VC, fac2_CC, fac3_CV, l, w, id: _do_sum_over_bands_for_single_term(
        output,
        fac1_VC=fac1_VC,
        fac2_CC=fac2_CC,
        fac3_CV=fac3_CV,
        k=k,
        l=l,
        w=w,
        term_id=id,
        conduction_indices=conduction_indices,
        valence_indices=valence_indices,
    )

    # Some of these factors don't depend on anything and can be evaluated right now.
    f1_in = f1_in_()
    f2_in = f2_in_()
    f2_out = f2_out_()
    for l in range(len(w_ph)):
        # Work with factors for a single phonon mode.
        f1_elph = f1_elph_(l=l)
        f2_elph = f2_elph_(l=l)
        f3_out = f3_out_(l=l)

        # Resonant term.
        add_term(f1_in, f2_elph, f3_out, l=l, w=None, id='lps')

        # Include non-resonant terms?
        if permutations:
            # compared to gpaw!563, I have rearranged the order of the terms to group together
            # the two that don't depend on the shift.
            #
            # This is the second of those terms.
            add_term(f1_elph, f2_in, f3_out, l=l, w=None, id='pls')

            # The remaining four terms depend on the raman shift in the original code.
            if permutations == 'fast':
                # For permutations == 'fast', they still only depend on the phonon
                f1_out = f1_out_(l=l)
                f3_in = f3_in_(l=l)
                f3_elph = f3_elph_(l=l)

                add_term(f1_in, f2_out, f3_elph, l=l, w=None, id='lsp')
                add_term(f1_out, f2_in, f3_elph, l=l, w=None, id='slp')
                add_term(f1_elph, f2_out, f3_in, l=l, w=None, id='psl')
                add_term(f1_out, f2_elph, f3_in, l=l, w=None, id='spl')

            elif permutations == 'original':
                for w in range(len(w_s)):
                    f1_out = f1_out_(w=w)
                    f3_in = f3_in_(w=w, l=l)
                    f3_elph = f3_elph_(w=w, l=l)

                    add_term(f1_in, f2_out, f3_elph, l=l, w=w, id='lsp')
                    add_term(f1_out, f2_in, f3_elph, l=l, w=w, id='slp')
                    add_term(f1_elph, f2_out, f3_in, l=l, w=w, id='psl')
                    add_term(f1_out, f2_elph, f3_in, l=l, w=w, id='spl')

            else:
                assert False, permutations

# Function for adding a single one of the six raman terms.
def _do_sum_over_bands_for_single_term(
    output: RamanOutput,
    # the three factors to be multiplied
    fac1_VC,
    fac2_CC,
    fac3_CV,
    # information about indices
    k, l, w,
    term_id: TermId,
    conduction_indices,
    valence_indices,
):
    # For terms that don't depend on w, broadcast onto the w axis
    w_eff = slice(None) if w is None else w

    # Here's the main thing we care about.
    output.raman_lw[l, w_eff] += np.einsum('si,ij,js->', fac1_VC, fac2_CC, fac3_CV)

    # NOTE: This repeats the numerical work done by einsum but I'm not too concerned about
    #       the performance of recording contributions...
    if output.contributions_lktnnn_parts is not None:
        assert w is None

        intermediate_truncate_threshold = 1e-13

        nphonons, nibzkpts, nbands = output.nphonons, output.nibzkpts, output.nbands
        nterms = len(output.terms_t)

        # NOTE: It'd probably be more efficient to make the input masked arrays sparse rather
        #       than sparsifying them here, but don't want to force the code to depend on the
        #       sparse package unless necessary.
        def sparsify_factor(data_XY, X_indices, Y_indices):
            # convert our dense array on conduction/valance bands into a sparse array on all bands
            data_nn = np.zeros((nbands, nbands), dtype=complex)
            data_nn[tuple(np.meshgrid(X_indices, Y_indices, indexing='ij'))] = data_XY
            sparse_nn = sparse.GCXS.from_numpy(data_nn)

            return _truncate_sparse(sparse_nn, rel=intermediate_truncate_threshold)

        sparse1_nn = sparsify_factor(fac1_VC, valence_indices, conduction_indices)
        sparse2_nn = sparsify_factor(fac2_CC, conduction_indices, conduction_indices)
        sparse3_nn = sparsify_factor(fac3_CV, conduction_indices, valence_indices)
        prod_nnn = sparse1_nn[:, :, None] * sparse2_nn[None, :, :] * sparse3_nn.T[:, None, :]
        prod_nnn = prod_nnn.tocoo()
        prod_nnn = _truncate_sparse(prod_nnn, rel=intermediate_truncate_threshold)

        # Add in the remaining axes by prepending constant indices.
        # This can be done using kron with an array that has a single nonzero element.
        t = output.terms_t.index(term_id)
        delta_lkt = np.zeros((nphonons, nibzkpts, nterms))
        delta_lkt[l][k][t] = 1   # put our data at these fixed coordinates
        output.contributions_lktnnn_parts.append(sparse.kron(
            delta_lkt[:, :, :, None, None, None],
            prod_nnn[None, None, None, :, :, :],
        ))

def _truncate_sparse(arr, abs=None, rel=None):
    assert (abs is not None) != (rel is not None)
    abs_arr = np.absolute(arr)
    cutoff = abs if abs is not None else rel * abs_arr.max()
    return sparse.where(abs_arr < cutoff, 0, arr)

def plot_raman(yscale="linear", figname="Raman.png", relative=False, w_min=None, w_max=None, ramanname=None):
    """
        Plots a given Raman spectrum

        Input:
            yscale: Linear or logarithmic yscale
            figname: Name of the generated figure
            relative: Scale to the highest peak
            w_min, w_max: The plotting range wrt the Raman shift
            ramanname: Suffix used for the file containing the Raman spectrum

        Output:
            ramanname: image containing the Raman spectrum.

    """
    import matplotlib
    matplotlib.use('Agg')  # FIXME: Evil, none of this function's business
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    from ase.parallel import world

    # Plotting function

    if world.rank == 0:
        legend = isinstance(ramanname, (list, tuple))
        if ramanname is None:
            RI_name = ["RI.npy"]
        elif type(ramanname) == list:
            RI_name = ["RI_{}.npy".format(name) for name in ramanname]
        else:
            RI_name = ["RI_{}.npy".format(ramanname)]

        ylabel = "Intensity (arb. units)"
        inferno = cm = plt.get_cmap('inferno')
        cNorm = colors.Normalize(vmin=0, vmax=len(RI_name))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        peaks = None
        for i, name in enumerate(RI_name):
            RI = np.real(np.load(name))
            if w_min == None:
                w_min = np.min(RI[0])
            if w_max == None:
                w_max = np.max(RI[0])
            r = RI[1][np.logical_and(RI[0] >= w_min, RI[0] <= w_max)]
            w = RI[0][np.logical_and(RI[0] >= w_min, RI[0] <= w_max)]
            cval = scalarMap.to_rgba(i)
            if relative:
                ylabel = "I/I_max"
                r = r/np.max(r)
            if peaks is None:
                peaks = signal.find_peaks(
                    r[np.logical_and(w >= w_min, w <= w_max)])[0]
                locations = np.take(
                    w[np.logical_and(w >= w_min, w <= w_max)], peaks)
                intensities = np.take(
                    r[np.logical_and(w >= w_min, w <= w_max)], peaks)
            if legend:
                plt.plot(w, r, color=cval, label=ramanname[i])
            else:
                plt.plot(w, r, color=cval)
        for i, loc in enumerate(locations):
            if intensities[i]/np.max(intensities) > 0.05:
                plt.axvline(x=loc,  color="grey", linestyle="--")

        # FIXME: usage of pyplot API
        plt.yscale(yscale)
        plt.minorticks_on()
        if legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title("Raman intensity")
        plt.xlabel("Raman shift (cm$^{-1}$)")
        plt.ylabel(ylabel)
        if not relative:
            plt.yticks([])
        plt.savefig(figname, dpi=300)
        plt.clf()

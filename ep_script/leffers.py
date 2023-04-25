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
import gpaw.kpoint

from ase.phonons import Phonons
from ase.parallel import parprint

def get_elph_elements(calc_gs, calc_fd, supercell, phononname='phonon'):
    """
        Evaluates the dipole transition matrix elements

        Input
        ----------
        calc_gs:    Ground state calculator on unit cell.
        calc_fd:    Finite displacement calculator on supercell.
        supercell (tuple): Supercell, default is (1,1,1) used for gamma phonons

        Output
        ----------
        g_sqklnn, the electron-phonon matrix elements
    """
    from ase.phonons import Phonons
    from gpaw.raman.elph import EPC

    calc_gs.initialize_positions(calc_gs.get_atoms())

    phonon = Phonons(atoms=calc_gs.get_atoms(), name=phononname, supercell=supercell)
    epc = EPC(calc_gs.atoms, calc=calc_fd, supercell=supercell)
    g_sqklnn = epc.get_elph_matrix(calc_gs, phonon, savetofile=False)
    return g_sqklnn

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

PARTICLE_TYPES = ['electron', 'hole']
ParticleType = tp.Literal.__getitem__(tuple(PARTICLE_TYPES))
@dataclass
class RamanOutput:
    raman_lw: np.ndarray
    contributions_lksptnnn_parts: tp.Optional[tp.List[sparse.GCXS]]
    nphonons: int
    nibzkpts: int
    nbands: int
    nspins: int
    # ordering of the term and particle axes
    terms_t: tp.List[TermId]
    particles_p: tp.List[ParticleType]

    def term_index_from_str(self, term: TermId):
        return self.terms_t.index(term)

    def particle_index_from_str(self, particle_type: ParticleType):
        return self.particles_p.index(particle_type)

    @property
    def nterms(self):
        return len(self.terms_t)

    @property
    def nparticles(self):
        return len(self.particles_p)

def calculate_raman(
    calc,
    *,
    w_ph,
    d_i,
    d_o,
    w_l,
    mom_skvnn,
    elph_sklnn,
    permutations='original',
    w_cm=None,
    ramanname=None,
    gamma_l=0.2,
    shift_step=1,
    shift_type='stokes',
    phonon_sigma=3,
    particle_types=PARTICLE_TYPES,
    kpoint_symmetry_form=False,
    write_mode_amplitudes=False,
    write_contributions=False,
):
    """
    Calculates the first order Raman spectre

    Input:
        w_ph            Gamma phonon energies in eV.
        permutations    Use only resonant term (None) or all terms ('original' or 'fast')
        ramanname       Suffix for the raman.npy file
        mom_skvnn       Momentum transition elements
        elph_sklnn      Elph elements for gamma phonons
        w_cm            Raman shift frequencies to compute at.
        w_l, gamma_l    Laser energy, broadening factor for the electron energies (eV)
        d_i, d_o        Laser polarization in, out (0, 1, 2 for x, y, z respectively)
    Output:
        RI.npy          Numpy array containing the raman spectre
    """

    assert permutations in [None, 'fast', 'original']
    assert shift_type in ['stokes', 'anti-stokes']

    parprint("Calculating the Raman spectra: Laser frequency = {}".format(w_l))

    bzk_kc = calc.get_ibz_k_points()
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
        contributions_lksptnnn_parts = [] if write_contributions else None,
        nphonons = nphonons,
        nibzkpts = nibzkpts,
        nbands = calc.wfs.bd.nbands,
        nspins = calc.wfs.nspins,
        # ordering found in Dresselhaus
        terms_t = ['lps', 'lsp', 'spl', 'slp', 'pls', 'psl'],
        particles_p = ['electron', 'hole'],
    )

    parprint("Reading matrix elements")

    kcomm = calc.wfs.kd.comm
    world = calc.wfs.world

    parprint("Evaluating Raman sum")

    for info_at_k in _distribute_bands_by_k(calc, mom_skvnn=mom_skvnn, elph_sklnn=elph_sklnn, kpoint_symmetry_form=kpoint_symmetry_form):
        print(f"For (k, s) = ({info_at_k.kpt.k}, {info_at_k.kpt.s})".format(info_at_k.kpt.s))
        _add_raman_terms_at_k(output, w_l, gamma_l, (d_i, d_o), w_ph, w_scatter, info_at_k, shift_type, permutations, particle_types)

    kcomm.sum(output.raman_lw)

    contributions_lksptnnn = None
    if output.contributions_lksptnnn_parts is not None:
        contributions_lksptnnn = _sum_sparse_coo(output.contributions_lksptnnn_parts)
        contributions_lksptnnn = _mpi_sum_sparse_coo(contributions_lksptnnn, kcomm)

    if write_mode_amplitudes:
        # write values without the gaussian on shift
        if world.rank == 0:
            if permutations == 'original':
                np.save(
                    "ModeA_lw{}.npy".format(make_suffix(ramanname)),
                    np.vstack([[w_cm], output.raman_lw[:, :]]),
                )
            else:
                np.save(
                    "ModeA_l{}.npy".format(make_suffix(ramanname)),
                    output.raman_lw[:, 0],
                )
    world.barrier()

    if contributions_lksptnnn is not None:
        if world.rank == 0:
            _write_raman_contributions(
                "Contrib{}.npz".format(make_suffix(ramanname)),
                calc,
                contributions_lksptnnn,
                output.terms_t,
                particles_p=output.particles_p,
            )
    world.barrier()

    RI = np.zeros(len(w_shift))
    for l in range(nphonons):
        if w_ph[l].real >= 0:
            parprint(
                "Phonon {} with energy = {} registered".format(l, w_ph[l]))
            RI += (np.abs(output.raman_lw[l])**2)*np.array(gaussian(w_shift-w_ph[l], sigma=phonon_sigma * cm))

    raman = np.vstack((w_cm, RI))
    if world.rank == 0:
        np.save("RI{}.npy".format(make_suffix(ramanname)), raman)
    world.barrier()

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

def _distribute_bands_by_k(calc, mom_skvnn, elph_sklnn, kpoint_symmetry_form: bool):
    E_skn = calc.band_structure().todict()["energies"]

    def get_symmetry_data(kpt):
        if kpoint_symmetry_form == 'mult':
            # naively account for multiplicity
            bad_func = lambda value: value * kpt.weight
            do_proper_conj = False
        else:
            # elph and moment matrix elements are both conjugated under time inversion
            assert not calc.symmetry.point_group, "general point group K-symmetry not supported"
            if (
                calc.symmetry.time_reversal
                and not np.allclose(calc.wfs.kd.ibzk_kc[kpt.k], [0, 0, 0])
            ):
                if kpoint_symmetry_form == 'badconj':
                    # Naively do tensor + tensor.conj(), which accidentally conjugates the denominator as well.
                    bad_func = lambda tensor: 2 * tensor.real
                    do_proper_conj = False
                elif kpoint_symmetry_form == 'conj':
                    # This one is handled differently.  We need a second einsum for the mirrored K.
                    bad_func = lambda tensor: tensor
                    do_proper_conj = True
                else:
                    assert False, kpoint_symmetry_form
            else:
                # Point has no symmetry partner.
                bad_func = lambda tensor: tensor
                do_proper_conj = False

        return bad_func, do_proper_conj

    out = []
    parprint("Distributing coupling terms")
    for kpt in calc.wfs.kpt_u:
        # NOTE: The reason for this '/ weight' is because the occupancies returned by collect_occupations() for any given 'k'
        #       lie in the interval '[0, weight]', while we want values in the interval [0, 1].
        #
        #       Why does gpaw scale them like this?  Well, as far as I can tell:
        #
        #       - The kpoint weights are chosen to sum to 2.0 when summed over symmetry-reduced kpoints.
        #       - The 'weight' of a given ibzkpoint k is  2 x (size of k's symmetry star) / (total number of kpoints).
        #
        #       The purpose of the scaling by gpaw thus appears to be in order to make is so that, when the occupancies are summed
        #       over all symmetry-reduced kpoints and all bands, the total is equal to the number of valence electrons.
        f_n = np.array(kpt.f_n / kpt.weight, dtype=float)
        # NOTE: the original script accounted for symmetry multiplicity in elph
        #       by multiplying in weight here
        elph_lnn = np.array(elph_sklnn[kpt.s, kpt.k], dtype=complex)
        mom_vnn = np.array(mom_skvnn[kpt.s, kpt.k], dtype=complex)

        apply_bad_sum_over_equivalent_k, do_proper_conj = get_symmetry_data(kpt)
        out.append(_InfoAtK(
            kpt=kpt,
            apply_bad_sum_over_equivalent_k=apply_bad_sum_over_equivalent_k,
            do_proper_conj=do_proper_conj,
            band_energy_n=E_skn[kpt.s, kpt.k],
            band_occupation_n=f_n,
            band_momentum_vnn=mom_vnn,
            band_elph_lnn=elph_lnn,
        ))
    return out

def _write_raman_contributions(outpath, calc, contributions_lksptnnn: sparse.COO, terms_t: tp.List[TermId], particles_p: tp.List[ParticleType]):
    np.savez_compressed(
        outpath,
        k_weight=calc.get_k_point_weights(),
        k_coords=calc.get_ibz_k_points(),
        num_phonons=contributions_lksptnnn.shape[0],
        num_ibzkpoints=contributions_lksptnnn.shape[1],
        num_spins=contributions_lksptnnn.shape[2],
        num_particles=contributions_lksptnnn.shape[3],
        num_terms=contributions_lksptnnn.shape[4],
        num_bands=contributions_lksptnnn.shape[5],
        contrib_phonon=contributions_lksptnnn.coords[0, :],
        contrib_band_k=contributions_lksptnnn.coords[1, :],
        contrib_band_spin=contributions_lksptnnn.coords[2, :],
        contrib_particle=contributions_lksptnnn.coords[3, :],
        contrib_term=contributions_lksptnnn.coords[4, :],
        contrib_band_1=contributions_lksptnnn.coords[5, :],
        contrib_band_2=contributions_lksptnnn.coords[6, :],
        contrib_band_3=contributions_lksptnnn.coords[7, :],
        contrib_value=contributions_lksptnnn.data,
        particle_str=particles_p,
        term_str=terms_t,
    )

@dataclass
class _InfoAtK:
    kpt: gpaw.kpoint.KPoint
    apply_bad_sum_over_equivalent_k: tp.Callable[[np.ndarray], np.ndarray]
    do_proper_conj: bool
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
    info_at_k: _InfoAtK,
    shift_type: tp.Literal['stokes', 'anti-stokes'],
    permutations: tp.Literal[None, 'fast', 'original'],
    particle_types: tp.List[ParticleType],
):
    assert permutations in [None, 'fast', 'original']
    assert len(set(particle_types)) == len(particle_types), "duplicate particle type"
    assert not (set(particle_types) - set(PARTICLE_TYPES)), "bad particle type"
    d_i, d_o = polarizations
    k = info_at_k.kpt.k
    s = info_at_k.kpt.s
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
    occu3 = (1-f_n[:,None]) * f_n[None,:]  # occupation-based part that always appears in the 3rd tensor

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
    def cannot_compute(**_kw):
        assert False, '(bug) a function was unexpectedly called with permutations=None'
    get_w_s = {
        'original': lambda w: w_s[w],
        'fast': lambda l: w_l - w_ph_eff[l],
        None: cannot_compute,
    }[permutations]

    # There may be many bands that are fully occupied or unoccupied and therefore incapable of appearing
    # in one or more of the axes that we sum over.  Computing these elements is a waste of time.
    #
    # Define three lambdas that each mask a (nbands,nbands) matrix to only have bands appropriate in that position.
    isV_n = abs(f_n) > 1e-20
    isC_n = f_n != 1
    mask1 = lambda mat: mat[isV_n][:, isC_n]
    mask3 = lambda mat: mat[isC_n][:, isV_n]
    valence_indices = np.nonzero(isV_n)[0]
    conduction_indices = np.nonzero(isC_n)[0]

    # And now, the 9 tensors.
    #
    # In the original code, some of these tensors were VERY LARGE;  over 50 GB for 17-agnr.
    # Thus, to reduce memory requirements, I have rewritten them to not include axes for the phonon mode or
    # raman shift;  Instead they are all lambdas that produce a matrix with two band axes, and we'll
    # evaluate them at a single phonon/raman shift at a time.
    #
    # First event tensors
    f1_in_ = lambda: (
        mask1(occu1) * mask1(mom[d_i]),  # numerator
        w_l-mask1(Ediff_el) + 1j*gamma_l,  # denominator
    )
    f1_elph_ = lambda l: (
        mask1(occu1) * mask1(elph[l]),
        -w_ph_eff[l]-mask1(Ediff_el) + 1j*gamma_l,
    )
    f1_out_ = {
        'original': lambda w: (
            mask1(occu1) * mask1(mom[d_o]),
            -get_w_s(w=w)-mask1(Ediff_el) + 1j*gamma_l,
        ),
        'fast': lambda l: (
            mask1(occu1) * mask1(mom[d_o]),
            -get_w_s(l=l)-mask1(Ediff_el) + 1j*gamma_l,
        ),
        None: cannot_compute,
    }[permutations]

    # Third event tensors
    f3_in_ = {
        'original': lambda w, l: (
            mask3(occu3) * mask3(mom[d_i]),
            -get_w_s(w=w)-w_ph_eff[l]-mask3(Ediff_el.T) + 1j*gamma_l,
        ),
        'fast': lambda l: (
            mask3(occu3) * mask3(mom[d_i]),
            -w_l-mask3(Ediff_el.T) + 1j*gamma_l,
        ),
        None: cannot_compute,
    }[permutations]
    f3_elph_ = {
        'original': lambda w, l: (
            mask3(occu3) * mask3(elph[l]),
            w_l-get_w_s(w=w)-mask3(Ediff_el.T) + 1j*gamma_l,
        ),
        'fast': lambda l: (
            mask3(occu3) * mask3(elph[l]),
            w_ph_eff[l]-mask3(Ediff_el.T) + 1j*gamma_l,
        ),
        None: cannot_compute,
    }[permutations]
    f3_out_ = lambda l: (
        mask3(occu3) * mask3(mom[d_o]),
        w_l-w_ph_eff[l]-mask3(Ediff_el.T) + 1j*gamma_l,
    )

    # Second event tensors
    #
    # Unlike the others, we don't mask these ahead of time because the masking
    # is different for electron-electron transitions versus hole-hole transitions.
    f2_in_ = lambda: mom[d_i]
    f2_elph_ = lambda l: elph[l]
    f2_out_ = lambda: mom[d_o]

    def add_term(tuple1_VC, fac2_nn, tuple3_CV, l, w, id):
        for particle_type in particle_types:
            _do_sum_over_bands_for_single_term(
                output,
                tuple1_VC=tuple1_VC,
                fac2_nn=fac2_nn,
                tuple3_CV=tuple3_CV,
                k=k, s=s, l=l, w=w,
                particle_type=particle_type,
                term_id=id,
                conduction_indices=conduction_indices,
                valence_indices=valence_indices,
                apply_bad_sum_over_equivalent_k=info_at_k.apply_bad_sum_over_equivalent_k,
                do_proper_conj=info_at_k.do_proper_conj,
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
    tuple1_VC,   # tuple of (numer, denom), already masked to [valence, conduction]
    fac2_nn,   # not masked
    tuple3_CV,   # tuple of (numer, denom), already masked to [conduction, valence]
    # information about indices
    k, s, l, w,
    particle_type: ParticleType,
    term_id: TermId,
    conduction_indices,
    valence_indices,
    apply_bad_sum_over_equivalent_k,
    do_proper_conj: bool,
):
    # For terms that don't depend on w, broadcast onto the w axis
    w_eff = slice(None) if w is None else w

    numer1_VC, denom1_VC = tuple1_VC
    numer3_CV, denom3_CV = tuple3_CV

    # Here's the main thing we care about.
    if particle_type == 'electron':
        fac2_CC = fac2_nn[conduction_indices][:, conduction_indices]
        # FIXME Good god how do we factor out this logic
        output.raman_lw[l, w_eff] += apply_bad_sum_over_equivalent_k(
            np.einsum('si,ij,js->', numer1_VC / denom1_VC, fac2_CC, numer3_CV / denom3_CV)
        )
        # FIXME good god how to factor out this logic
        if do_proper_conj:
            output.raman_lw[l, w_eff] += (
                np.einsum('si,ij,js->', numer1_VC.conj() / denom1_VC, fac2_CC, numer3_CV.conj() / denom3_CV)
            )
    elif particle_type == 'hole':
        fac2_VV = fac2_nn[valence_indices][:, valence_indices]
        output.raman_lw[l, w_eff] -= apply_bad_sum_over_equivalent_k(
            np.einsum('si,ts,it->', numer1_VC / denom1_VC, fac2_VV, numer3_CV / denom3_CV)
        )
        if do_proper_conj:
            output.raman_lw[l, w_eff] -= (
                np.einsum('si,ts,it->', numer1_VC.conj() / denom1_VC, fac2_VV, numer3_CV.conj() / denom3_CV)
            )
    else:
        assert False, particle_type

    # NOTE: This repeats the numerical work done by einsum but I'm not too concerned about
    #       the performance of recording contributions...
    if output.contributions_lksptnnn_parts is not None:
        assert w is None

        intermediate_truncate_threshold = 1e-13

        # NOTE: It'd probably be more efficient to make the input masked arrays sparse rather
        #       than sparsifying them here, but don't want to force the code to depend on the
        #       sparse package unless necessary.
        # FIXME: The 'sparse' import has since been moved to the top of the file, does that mean
        #        avoiding the dependency is no longer a concern?  (presumably thanks to conda?)
        def sparsify_factor(data_XY, X_indices, Y_indices):
            # convert our dense array on conduction/valance bands into a sparse array on all bands
            data_nn = np.zeros((output.nbands, output.nbands), dtype=complex)
            data_nn[tuple(np.meshgrid(X_indices, Y_indices, indexing='ij'))] = data_XY
            sparse_nn = sparse.GCXS.from_numpy(data_nn)

            return _truncate_sparse(sparse_nn, rel=intermediate_truncate_threshold)

        if particle_type == 'electron':
            # make product whose indices are:   valence conduction1 conduction2
            sparse1_nn = sparsify_factor(numer1_VC / denom1_VC, valence_indices, conduction_indices)
            sparse2_nn = sparsify_factor(fac2_CC, conduction_indices, conduction_indices)
            sparse3_nn = sparsify_factor(numer3_CV / denom3_CV, conduction_indices, valence_indices)

            prod_nnn = sparse1_nn[:, :, None] * sparse2_nn[None, :, :] * sparse3_nn.T[:, None, :]
            if do_proper_conj:
                sparse1B_nn = sparsify_factor(numer1_VC.conj() / denom1_VC, valence_indices, conduction_indices)
                sparse3B_nn = sparsify_factor(numer3_CV.conj() / denom3_CV, conduction_indices, valence_indices)
                prod_nnn += sparse1B_nn[:, :, None] * sparse2_nn[None, :, :] * sparse3B_nn.T[:, None, :]
        elif particle_type == 'hole':
            # make product whose indices are:   valence1 conduction valence2
            sparse1_nn = sparsify_factor(numer1_VC / denom1_VC, valence_indices, conduction_indices)
            sparse2_nn = sparsify_factor(fac2_VV, valence_indices, valence_indices)
            sparse3_nn = sparsify_factor(numer3_CV / denom3_CV, conduction_indices, valence_indices)
            prod_nnn = -1 * sparse1_nn[:, :, None] * sparse2_nn.T[:, None, :] * sparse3_nn[None, :, :]
            if do_proper_conj:
                sparse1B_nn = sparsify_factor(numer1_VC.conj() / denom1_VC, valence_indices, conduction_indices)
                sparse3B_nn = sparsify_factor(numer3_CV.conj() / denom3_CV, conduction_indices, valence_indices)
                prod_nnn -= sparse1B_nn[:, :, None] * sparse2_nn.T[:, None, :] * sparse3B_nn[None, :, :]
        else:
            assert False, particle_type
        prod_nnn = prod_nnn.tocoo()
        prod_nnn = _truncate_sparse(prod_nnn, rel=intermediate_truncate_threshold)
        prod_nnn = apply_bad_sum_over_equivalent_k(prod_nnn)

        # Add in the remaining axes by prepending constant indices.
        # This can be done using kron with an array that has a single nonzero element.
        t = output.term_index_from_str(term_id)
        p = output.particle_index_from_str(particle_type)
        delta_lkspt = np.zeros((output.nphonons, output.nibzkpts, output.nspins, output.nparticles, output.nterms))
        delta_lkspt[l][k][s][p][t] = 1   # put our data at these fixed coordinates
        output.contributions_lksptnnn_parts.append(sparse.kron(
            delta_lkspt[:, :, :, :, :, None, None, None],
            prod_nnn[None, None, None, None, None, :, :, :],
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
    world.barrier()

# calculate_supercell_matrix breaks if parallelized over domains so parallelize over kpt instead
# (note: it prints messages from all processes but it DOES run faster with more processes)
def _GPAW_without_domain_parallel(*args, **kw):
    from ase.parallel import world
    kw['parallel'] = {'domain': (1,1,1), 'band': 1, 'kpt': world.size}
    return GPAW(*args, **kw)

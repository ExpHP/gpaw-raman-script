#!/usr/bin/env python3

from . import symmetry
from . import interop
from . import utils
from . import leffers
from .leffers import _GPAW_without_domain_parallel

import functools
import os
import sys
import copy
import json

from collections import namedtuple
from datetime import datetime
import numpy as np
from ase.parallel import parprint, paropen, world
import phonopy
import ase.build
import ase.phonons
from ase.utils.filecache import MultiFileJSONCache
import gpaw
from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT
import warnings
from contextlib import contextmanager

from ruamel.yaml import YAML
yaml = YAML(typ='rt')

import typing as tp
T = tp.TypeVar("T")

DISPLACEMENT_DIST = 1e-2  # FIXME supply as arg to gpaw

PROG = 'ep_script'

class EarlySuccessfulTermination(BaseException):
    pass

def main():
    import argparse

    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()

    parser.set_defaults(func=lambda args, log: parser.error('missing test name'))

    def add_standard_arguments(p):
        p.add_argument('INPUT', help='.gpw file for unitcell, with structure and relevant parameters')
        p.add_argument('--supercell', type=(lambda s: tuple(map(int, s.split()))), dest='supercell', default=(1,1,1), help='space-separated list of 3 integers with number of repeats along each cell vector')
        p.add_argument('--params-fd', help='json file with GPAW params to modify for finite displacement (supercell)')
        p.add_argument('--symmetry-tol', type=float, default=1e-5, help=
            'Symmetry tolerance for phonopy.  This needs to be provided on every run, even after displacements are done',
        )

    def add_raman_arguments(p):
        p.add_argument('--laser-broadening', type=float, default=0.2, help='broadening in eV (imaginary part added to light freqencies)')
        p.add_argument('--phonon-broadening', type=float, default=3, help='phonon gaussian variance in cm-1')
        p.add_argument('--polarizations', type=lambda s: list(s.split(',')), default=[i+o for i in 'xyz' for o in 'xyz'], help='comma-separated list of raman polarizations to do (e.g. xx,xy,xz)')
        p.add_argument('--no-displacement-symmetry',
            dest='displacement_set', default='symmetry', const='6n', action='store_const', help=
            'explicitly compute data at all 6N cartesian displacements, rather than using symmetry.')
        p.add_argument('--write-mode-intensities', action='store_true', help='deprecated alias for --write-mode-amplitudes.')
        p.add_argument('--write-mode-amplitudes', action='store_true', help=
            'write mode amplitudes to files.  For --permutations=original, these will contain a second axis for the'
            ' raman shift. (this dependence arises from the form of the matrix elements, and does not account for broadening)')
        p.add_argument('--write-spectrum-plots', action='store_true', help='write raman plots')
        p.add_argument('--write-contributions', action='store_true', help='write individual electronic state raman contributions to a NPZ file')
        p.add_argument('--shift-type', choices=['stokes', 'anti-stokes'], default='stokes', help=
            'selects sign of the phonon frequency in the energy conservation equation.'
            " IMPORTANT: results of --shift-type 'anti-stokes' are not physical as they do not account"
            ' for the occupation of the phonon states (there is no temperature dependence).  Currently the purpose of'
            ' this flag is to demonstrate a relation between this sign factor and differences between off-diagonal'
            ' raman tensor elements. (stokes XY is similar to anti-stokes YX, and etc.)')
        p.add_argument('--permutations', choices=['original', 'default', 'fast', 'none'], default='default', help=
            'controls inclusion of nonresonant raman terms in the raman spectral intensity'
            " (i.e. event orderings other than light absorption, phonon emission, light emission)."
            " '--permutations=default' will include all six orderings."
            " '--permutations=none' only includes the resonant ordering."
            " '--permutations=fast' is equivalent to --permutations=default."
            " '--permutations=original' faithfully replicates Ulrik Leffer's original code; it includes all"
            " nonresonant terms but is SIGNIFICANTLY slower than the default setting as it expresses some terms"
            " as a function of the raman shift.")
        p.add_argument('--no-permutations', dest='permutations', action='store_const', const='none', help='alias for --permutations=none')
        p.add_argument('--kpoint-symmetry-bug', action='store_true', help=
            "Simulate a bug in old versions of the script that did not correctly "
            "account for complex conjugation of matrix elements under time-inversion symmetry "
            "when computing raman intensities. "
            "The new behavior is consistent with the implementation in GPAW.")
        DEFAULT_LASER_FREQS = '488,532,633nm'
        p.add_argument('--laser-freqs',
            type=parse_laser_freqs, default=parse_laser_freqs(DEFAULT_LASER_FREQS), help=
            'comma-separated list of laser wavelengths, followed by an optional unit (else assumed nm). '
            f'Default: {repr(DEFAULT_LASER_FREQS)}.  Available units: {", ".join(LASER_UNIT_CONVERSIONS)}')

        p.add_argument('--shift-step', type=int, default=1, help='step for x axis of raman shift (cm-1)')

    def extract_raman_arguments(p, args):
        permutations = None if args.permutations == 'none' else args.permutations
        if permutations == 'default':
            permutations = 'fast'

        write_mode_amplitudes = args.write_mode_intensities
        if args.write_mode_intensities:
            warnings.warn(f"--write-mode-intensities has been renamed to --write-mode-amplitudes")

        return dict(
            laser_broadening=args.laser_broadening,
            phonon_broadening=args.phonon_broadening,
            permutations=permutations,
            polarizations=args.polarizations,
            lasers=args.laser_freqs,
            shift_step=args.shift_step,
            write_mode_amplitudes=write_mode_amplitudes,
            write_plots=args.write_spectrum_plots,
            write_contributions=args.write_contributions,
            shift_type=args.shift_type,
            kpoint_symmetry_bug=args.kpoint_symmetry_bug,
        )

    p = subs.add_parser('ep')
    add_standard_arguments(p)
    add_raman_arguments(p)
    p.add_argument('--disp-split', metavar="IDX,MOD", type=parse_disp_split, default=None, help=
        'Only compute displacements with index IDX modulo MOD.  '
        'If provided, this process will stop after displacements.  '
        'Use --disp-split=stop to run the script up to the point JUST BEFORE doing displacements. '
        '(it is recommended to do one run with --disp-split=stop before starting multiple --disp-split runs, to avoid race conditions.)')

    p.set_defaults(func=lambda args, log: main_elph(
        structure_path=args.INPUT, supercell=args.supercell, params_fd_path=args.params_fd, log=log,
        symmetry_tol=args.symmetry_tol,
        disp_split=DispSplit(0, 1) if args.disp_split is None else args.disp_split,
        stop_after_displacements=args.disp_split is not None,
        displacement_set=args.displacement_set,
        raman_settings=extract_raman_arguments(p, args),
    ))

    p = subs.add_parser('ep-raman')
    add_standard_arguments(p)
    add_raman_arguments(p)
    p.set_defaults(func=lambda args, log: main_elph__after_symmetry(
        structure_path=args.INPUT, supercell=args.supercell, log=log,
        raman_settings=extract_raman_arguments(p, args),
    ))

    p = subs.add_parser('brute-gpw')
    p.add_argument('INPUT', help='.gpw file for unitcell, with structure and relevant parameters')
    p.add_argument('--supercell', type=(lambda s: tuple(map(int, s))), dest='supercell', default=(1,1,1))
    p.set_defaults(func=lambda args, log: main_brute_gpw(structure_path=args.INPUT, supercell=args.supercell, log=log))
    args = parser.parse_args()

    with start_log_entry('gpaw.log') as log:
        try:
            args.func(args, log)
        except EarlySuccessfulTermination:
            pass
        except UserError as e:
            parprint(f'{PROG}: error: {e}', file=log)
            raise

class UserError(RuntimeError):
    def __init__(self, message):
        lines = message.split('\n')
        super().__init__('\n' + ''.join(f'    {line}\n' for line in lines))

def start_log_entry(path):
    logfile = paropen(path, 'a')
    parprint(file=logfile)
    parprint('=====================================', file=logfile)
    parprint('===', datetime.now().isoformat(), file=logfile)

    if world.rank == 0:
        return utils.Tee(logfile, sys.stdout)
    else:
        return utils.Tee()

DispSplit = namedtuple('DispSplit', ['index', 'mod'])
def parse_disp_split(s):
    if s == 'stop':
        return 'stop'
    idx, mod = map(int, s.split(','))
    assert 0 <= idx < mod, "invalid --split-index (should satisfy 0 <= IDX < MOD)"
    return DispSplit(idx, mod)

LASER_UNIT_CONVERSIONS = {
    'nm': lambda x: x,
    # constant = c / (ev / h) / nm
    'eV': lambda x: 1239.84197386209 / x
}
def parse_laser_freqs(s):
    from argparse import ArgumentTypeError
    import re
    unit_match = re.search('[a-zA-Z][a-zA-Z0-9_-]* *$', s)
    if unit_match:
        split = unit_match.start()
        s, unit_str = s[:split].strip(), s[split:].strip()
    else:
        unit_str = 'nm'

    def parse_float(s):
        try:
            return float(s)
        except ValueError:
            raise ArgumentTypeError(f'invalid float value: {s}')

    value_strs = [word.strip() for word in s.split(',')]
    value_floats = [parse_float(word) for word in value_strs]
    value_texts = [f'{word}{unit_str}' for word in value_strs]
    try:
        to_nm = LASER_UNIT_CONVERSIONS[unit_str]
    except KeyError:
        raise ArgumentTypeError(f'unit {repr(unit_str)} is not implemented for lasers')

    return [Laser(text, to_nm(x)) for (text, x) in zip(value_texts, value_floats)]

class Laser:
    wavelength_nm: float
    text: str
    def __init__(self, text, wavelength_nm):
        self.text = text
        self.wavelength_nm = wavelength_nm

# ==============================================================================

def main_brute_gpw(structure_path, supercell, log):
    from gpaw.elph.electronphonon import ElectronPhononCoupling
    from gpaw import GPAW

    calc = GPAW(structure_path)
    supercell_atoms = make_gpaw_supercell(calc, supercell, txt=log)

    # NOTE: confusingly, Elph wants primitive atoms, but a calc for the supercell
    elph = ElectronPhononCoupling(calc.atoms, calc=supercell_atoms.calc, supercell=supercell, calculate_forces=True)
    elph.run()
    supercell_atoms.calc.wfs.gd.comm.barrier()
    elph = ElectronPhononCoupling(calc.atoms, calc=supercell_atoms.calc, supercell=supercell)
    elph.set_lcao_calculator(supercell_atoms.calc)
    elph.calculate_supercell_matrix()
    return

def main_elph(
        structure_path,
        params_fd_path,
        supercell,
        log,
        symmetry_tol,
        disp_split,
        stop_after_displacements,
        raman_settings,
        displacement_set,
):
    main_elph__init(
        structure_path=structure_path,
        params_fd_path=params_fd_path,
        supercell=supercell,
        log=log,
        symmetry_tol=symmetry_tol,
        displacement_set=displacement_set,
    )

    main_elph__run_displacements(
        structure_path=structure_path,
        supercell=supercell,
        log=log,
        symmetry_tol=symmetry_tol,
        disp_split=disp_split,
        stop_after_displacements=stop_after_displacements,
        displacement_set=displacement_set,
    )

    main_elph__do_supercell_matrix(
        structure_path=structure_path,
        log=log,
        supercell=supercell,
    )

    main_elph__after_symmetry(
        structure_path=structure_path,
        supercell=supercell,
        log=log,
        raman_settings=raman_settings,
    )

def main_elph__init(
        structure_path,
        params_fd_path,
        supercell,
        log,
        symmetry_tol,
        displacement_set,
):
    from gpaw import GPAW

    calc = GPAW(structure_path)
    if calc.wfs.kpt_u[0].C_nM is None:
        parprint(f"'{structure_path}': no wavefunctions! You must save your .gpw file with 'mode=\"all\"'!")
        sys.exit(1)

    if calc.parameters['convergence']['bands'] == 'occupied':
        parprint(f"'{structure_path}': WARNING: only occupied bands were converged!  Please converge some conduction band states as these are an explicit part of the electron-phonon computation.")

    if params_fd_path is not None:
        params_fd = json.load(open(params_fd_path))
    else:
        params_fd = {}

    # atoms = Cluster(ase.build.molecule('CH4'))
    # atoms.minimal_box(4)
    # atoms.pbc = Tr
    
    if displacement_set == 'symmetry':
        if os.path.exists('phonopy_disp.yaml'):
            parprint('using saved phonopy_disp.yaml')
        else:
            parprint('computing phonopy_disp.yaml')
            world.barrier()  # avoid race condition where rank 0 creates file before others enter
            phonon = get_minimum_displacements(
                unitcell=ase_atoms_to_phonopy(calc.atoms),
                supercell_matrix=np.diag(supercell),
                displacement_distance=DISPLACEMENT_DIST,
                phonopy_kw=dict(
                    symprec=symmetry_tol,
                ),
            )
            if world.rank == 0:
                phonon.save('phonopy_disp.yaml')
            world.barrier()  # avoid race condition where rank 0 creates file before others enter

    # Structure with initial guess of wavefunctions for displacement calculations.
    if os.path.exists('supercell.eq.gpw'):
        parprint('using saved supercell.eq.gpw')
    else:
        parprint('computing supercell.eq.gpw')
        supercell_atoms = make_gpaw_supercell(calc, supercell, **dict(params_fd, txt=log))
        ensure_gpaw_setups_initialized(supercell_atoms.calc, supercell_atoms)
        supercell_atoms.get_potential_energy()
        supercell_atoms.calc.write('supercell.eq.gpw', mode='all')

def main_elph__run_displacements(
        structure_path,
        supercell,
        log,
        symmetry_tol,
        disp_split,
        stop_after_displacements,
        displacement_set,
):
    if disp_split == 'stop':
        parprint('stopping. (--disp-split=stop)')
        raise EarlySuccessfulTermination

    main_elph__run_split_displacements(
        structure_path=structure_path,
        supercell=supercell,
        log=log,
        symmetry_tol=symmetry_tol,
        disp_split=disp_split,
        displacement_set=displacement_set,
    )

    if stop_after_displacements:
        parprint('stopping here due to --disp-split.  Please resume without it.')
        raise EarlySuccessfulTermination

    main_elph__symmetry_expansion(
        structure_path=structure_path,
        supercell=supercell,
        log=log,
        symmetry_tol=symmetry_tol,
        displacement_set=displacement_set,
    )


def main_elph__run_split_displacements(
        structure_path,
        supercell,
        log,
        symmetry_tol,
        disp_split,
        displacement_set,
):
    calc = GPAW(structure_path)
    supercell_atoms = GPAW('supercell.eq.gpw', txt=log).get_atoms()
    natoms_prim = len(calc.atoms)

    cache = ElphCache('elph')

    if cache.has_all_cartesian_data(natoms=natoms_prim):
        parprint('== skipping all displacement computations (found cached cartesian data)')
        return

    parprint('checking for any missing data at displacements')

    def do_structure(supercell_atoms, name):
        filename = f'elph/cache.{name}.json'
        with cache.lock(name) as handle:
            if handle is None:
                if cache.read(name) is None:
                    raise UserError(
                        f"Detected lockfile (empty file) at  {filename}\n"
                        "If the previous job for this file has terminated, please delete the file."
                    )
                else:
                    parprint(f'== skipping  {filename}  (already computed)')
            if handle is not None:
                parprint(f'== computing  {filename}')
                Vt_sG, dH_all_asp = get_elph_data(supercell_atoms)
                forces = supercell_atoms.get_forces()
                if world.rank == 0:
                    handle.write(ElphDataset(
                        Vt_sG=Vt_sG,
                        dH_all_asp=dH_all_asp,
                        forces=forces,
                    ))

    if displacement_set == 'symmetry':
        phonopy_kw = dict(symprec=symmetry_tol)
        phonon = phonopy.load('phonopy_disp.yaml', produce_fc=False, **phonopy_kw)
        disp_phonopy_sites, disp_carts = get_phonopy_displacements(phonon)
        disp_sites = phonopy_sc_indices_to_ase_sc_indices(disp_phonopy_sites, natoms_prim, supercell)
        disp_keys = [f'sym-{disp_index}' for disp_index in range(len(disp_carts))]

    elif displacement_set == '6n':
        all_displacements = list(interop.AseDisplacement.iter(natoms_prim))
        disp_sites = [disp.atom for disp in all_displacements]
        disp_carts = [disp.cart_displacement(DISPLACEMENT_DIST) for disp in all_displacements]
        disp_keys = all_displacements

    else:
        assert False, displacement_set

    if disp_split.index == 0:
        do_structure(supercell_atoms, 'eq')

    all_displaced_structures = iter_displaced_structures(supercell_atoms, disp_sites, disp_carts)
    for disp_index, (disp_key, displaced_atoms) in enumerate(zip(disp_keys, all_displaced_structures)):
        supercell_atoms.set_positions(displaced_atoms.get_positions())
        if (disp_index + 1) % disp_split.mod == disp_split.index:  # + 1 because equilibrium was zero
            do_structure(supercell_atoms, disp_key)

def main_elph__symmetry_expansion(
        structure_path,
        supercell,
        log,
        symmetry_tol,
        displacement_set,
):
    calc = GPAW(structure_path)
    supercell_atoms = GPAW('supercell.eq.gpw', txt=log).get_atoms()
    natoms_prim = len(calc.atoms)

    if ElphCache('elph').has_all_cartesian_data(natoms_prim):
        # no expansion necessary, either it was computed this run or is already cached
        return

    assert displacement_set == 'symmetry', displacement_set
    parprint('generating data at cartesian displacements by expanding symmetry-reduced data.')

    phonopy_kw = dict(symprec=symmetry_tol)
    phonon = phonopy.load('phonopy_disp.yaml', produce_fc=False, **phonopy_kw)

    disp_phonopy_sites, disp_carts = get_phonopy_displacements(phonon)
    disp_sites = phonopy_sc_indices_to_ase_sc_indices(disp_phonopy_sites, natoms_prim, supercell)

    elph_do_symmetry_expansion(supercell, calc, DISPLACEMENT_DIST, phonon, disp_carts, disp_sites, supercell_atoms)

def main_elph__after_symmetry(
        structure_path,
        supercell,
        log,
        raman_settings
):
    # gqklnn, dip_vknm, and raman all don't support domain parallelism currently
    calc = _GPAW_without_domain_parallel(structure_path)

    if not os.path.exists('gqklnn.npy'):
        supercell_atoms = GPAW('supercell.eq.gpw', txt=log).get_atoms()
        leffers.get_elph_elements(calc.atoms, gpw_name=structure_path, calc_fd=supercell_atoms.calc, sc=supercell, phononname='elph')

    if not os.path.isfile("dip_vknm.npy"):
        leffers.get_dipole_transitions(calc)

    elph_do_raman_spectra(calc, supercell, **raman_settings, phononname='elph')

def elph_do_symmetry_expansion(
        supercell,
        calc,
        displacement_dist,
        phonon,
        disp_carts,
        disp_sites,
        supercell_atoms,
):
    from gpaw.elph.electronphonon import ElectronPhononCoupling

    cache = ElphCache('elph')
    natoms_prim = len(calc.get_atoms())
    disp_values = [cache.read(f'sym-{index}') for index in range(len(disp_sites))]

    if any(x is None for x in disp_values):
        raise UserError(
            "Detected unfinished computations in elph/ cache.\n"
            "Please delete any empty files in elph/ and rerun the displacements step."
        )

    # NOTE: phonon.symmetry includes pure translational symmetries of the supercell
    #       so we use an empty quotient group
    quotient_perms = np.array([np.arange(len(supercell_atoms))])
    super_lattice = supercell_atoms.get_cell()[...]
    super_symmetry = phonon.symmetry.get_symmetry_operations()
    oper_sfrac_rots = super_symmetry['rotations']
    oper_sfrac_trans = super_symmetry['translations']
    oper_cart_rots = np.array([super_lattice.T @ Rfrac @ np.linalg.inv(super_lattice).T for Rfrac in oper_sfrac_rots])
    oper_cart_trans = oper_sfrac_trans @ super_lattice
    oper_phonopy_coperms = phonon.symmetry.get_atomic_permutations()
    oper_phonopy_deperms = np.argsort(oper_phonopy_coperms, axis=1)

    # Convert permutations by composing the following three permutations:   into phonopy order, apply oper, back to ase order
    parprint('phonopy deperms:', oper_phonopy_deperms.shape)
    deperm_phonopy_to_ase = interop.get_deperm_from_phonopy_sc_to_ase_sc(natoms_prim, supercell)
    oper_deperms = [np.argsort(deperm_phonopy_to_ase)[deperm][deperm_phonopy_to_ase] for deperm in oper_phonopy_deperms]
    del oper_phonopy_coperms, oper_phonopy_deperms

    elphsym = symmetry.ElphGpawSymmetrySource.from_setups_and_ops(
        setups=supercell_atoms.calc.wfs.setups,
        lattice=super_lattice,
        oper_cart_rots=oper_cart_rots,
        oper_cart_trans=oper_cart_trans,
        oper_deperms=oper_deperms,
    )

    full_derivatives = symmetry.expand_derivs_by_symmetry(
        disp_sites,       # disp -> atom
        disp_carts,       # disp -> 3-vec
        disp_values,      # disp -> T  (displaced value, optionally minus equilibrium value)
        elph_callbacks_2(supercell_atoms.calc.wfs, elphsym, supercell=supercell),        # how to work with T
        oper_cart_rots,   # oper -> 3x3
        oper_perms=oper_deperms,       # oper -> atom' -> atom
        quotient_perms=quotient_perms,
    )

    # NOTE: confusingly, Elph wants primitive atoms, but a calc for the supercell
    elph = ElectronPhononCoupling(calc.atoms, calc=supercell_atoms.calc, supercell=supercell)
    cache = ElphCache(elph.name)
    displaced_cell_index = elph.offset
    del elph  # that's all we needed it for

    eq_Vt, eq_dH, eq_forces = cache.read('eq')
    for a in range(natoms_prim):
        for c in range(3):
            delta_Vt, delta_dH, delta_forces = full_derivatives[natoms_prim * displaced_cell_index + a][c]
            for sign in [-1, +1]:
                disp = interop.AseDisplacement(atom=a, axis=c, sign=sign)
                with cache.lock(disp) as handle:
                    Vt_sG = eq_Vt + sign * displacement_dist * delta_Vt
                    dH_all_asp = {k: eq_dH[k] + sign * displacement_dist * delta_dH[k] for k in eq_dH}
                    forces = eq_forces + sign * displacement_dist * delta_forces
                    if handle is not None:
                        handle.write(ElphDataset(Vt_sG=Vt_sG, dH_all_asp=dH_all_asp, forces=forces))

def make_gpaw_supercell(calc: GPAW, supercell: tp.Tuple[int, int, int], **new_kw):
    atoms = calc.atoms

    # Take most parameters from the unit cell.
    params = copy.deepcopy(calc.parameters)
    try: del params['txt']
    except KeyError: pass

    # This makes the real space grid points identical to the primitive cell computation.
    # (by increasing the counts by a factor of a supercell dimension)
    params['gpts'] = calc.wfs.gd.N_c * supercell
    try: del params['h']
    except KeyError: pass

    # Decrease kpt count to match density in reciprocal space.
    # FIXME: if gamma is False, the new kpoints won't match the old ones.
    #        However, it doesn't seem appropriate to warn about this because ElectronPhononCoupling itself
    #        warns about gamma calculations for some reason I do not yet understand.  - ML
    old_kpts = params['kpts']
    params['kpts'] = {'size': tuple(np.ceil(calc.wfs.kd.N_c / supercell).astype(int))}  # ceil so that 1 doesn't become 0
    if isinstance(old_kpts, dict) and 'gamma' in old_kpts:
        params['kpts']['gamma'] = old_kpts['gamma']

    # warn if kpoint density could not be preserved (unless it's just one point in an aperiodic direction)
    if any((k % c != 0) and not (k, c, pbc) == (1, 1, False) for (k, c, pbc) in zip(calc.wfs.kd.N_c, supercell, atoms.pbc)):
        warnings.warn('original kpts not divisible by supercell; density in supercell will be different')

    sc_atoms = atoms * supercell
    sc_atoms.calc = GPAW(**dict(params, **new_kw))
    return sc_atoms

def get_elph_data(atoms):
    # This here is effectively what ElectronPhononCoupling.__call__ does.
    # It returns the data that should be pickled for a single displacement.
    atoms.get_potential_energy()

    calc = atoms.calc

    Vt_sG = calc.wfs.gd.collect(calc.hamiltonian.vt_sG, broadcast=True)
    dH_asp = interop.gpaw_broadcast_array_dict_to_dicts(calc.hamiltonian.dH_asp)
    return Vt_sG, dH_asp

def main_elph__do_supercell_matrix(structure_path, log, supercell):
    from gpaw.elph.electronphonon import ElectronPhononCoupling

    calc = GPAW(structure_path)

    supercell_atoms = _GPAW_without_domain_parallel('supercell.eq.gpw', txt=log).get_atoms()

    elph = ElectronPhononCoupling(calc.atoms, supercell=supercell, calc=supercell_atoms.calc)
    elph.set_lcao_calculator(supercell_atoms.calc)
    # to initialize bfs.M_a
    ensure_gpaw_setups_initialized(supercell_atoms.calc, supercell_atoms)
    elph.calculate_supercell_matrix()

    world.barrier()

def elph_do_raman_spectra(
        calc,
        supercell,
        lasers,
        permutations,
        laser_broadening,
        phonon_broadening,
        shift_step,
        shift_type,
        polarizations,
        write_mode_amplitudes,
        write_plots,
        write_contributions,
        kpoint_symmetry_bug,
        phononname='phonons',
):
    from ase.units import _hplanck, _c, J

    parprint('Computing phonons')
    ph = ase.phonons.Phonons(atoms=calc.atoms, name=phononname, supercell=supercell)
    ph.read()
    w_ph = np.array(ph.band_structure([[0, 0, 0]])[0])

    if calc.world.rank == 0:
        np.save('frequencies.npy', w_ph * 8065.544)  # frequencies in cm-1

    # And the Raman spectra are calculated
    for laser in lasers:
        w_l = _hplanck*_c*J/(laser.wavelength_nm * 10**(-9))
        for polarization in polarizations:
            if len(polarization) != 2:
                raise ValueError(f'invalid polarization "{polarization}", should be two characters like "xy"')
            d_i = 'xyz'.index(polarization[0])
            d_o = 'xyz'.index(polarization[1])
            name = f"{laser.text}-{polarization}"
            if shift_type == 'anti-stokes':
                name = f"{name}-antistokes"
            if not os.path.isfile(f"RI_{name}.npy"):
                leffers.calculate_raman(
                    calc=calc, w_ph=w_ph, permutations=permutations,
                    w_l=w_l, ramanname=name, d_i=d_i, d_o=d_o,
                    gamma_l=laser_broadening, phonon_sigma=phonon_broadening,
                    shift_step=shift_step, shift_type=shift_type,
                    write_mode_amplitudes=write_mode_amplitudes,
                    write_contributions=write_contributions,
                    kpoint_symmetry_bug=kpoint_symmetry_bug,
                )

            # And plotted
            if write_plots:
                leffers.plot_raman(relative = True, figname = f"Raman_{name}.png", ramanname = name)

def elph_callbacks(wfs_with_symmetry: gpaw.wavefunctions.base.WaveFunctions, supercell):
    elphsym = symmetry.ElphGpawSymmetrySource.from_wfs_with_symmetry(wfs_with_symmetry)
    return elph_callbacks_2(wfs_with_symmetry, elphsym, supercell)

# FIXME: rename (just different args)
def elph_callbacks_2(wfs: gpaw.wavefunctions.base.WaveFunctions, elphsym: symmetry.ElphGpawSymmetrySource, supercell):
    Vt_part = symmetry.GpawLcaoVTCallbacks(wfs, elphsym, supercell=supercell)
    dH_part = symmetry.GpawLcaoDHCallbacks(wfs, elphsym)
    forces_part = symmetry.GeneralArrayCallbacks(['atom', 'cart'])
    return symmetry.TupleCallbacks(Vt_part, dH_part, forces_part)

# A namedtuple whose dict representation matches the data stored by ElectronPhononCoupling in its cache
class ElphDataset(tp.NamedTuple):
    Vt_sG: np.ndarray
    dH_all_asp: np.ndarray
    forces: np.ndarray

class ElphCache:
    def __init__(self, name):
        self.cache = MultiFileJSONCache(name)

    def read(self, displacement: tp.Union[interop.AseDisplacement, str]):
        """ Reads entry if it exists. Throws if it does not. Returns None on empty JSON. """
        d = self.cache[str(displacement)]
        if d is None:
            return None
        return ElphDataset(**d)

    def has_data(self, displacement: tp.Union[interop.AseDisplacement, str]):
        try:
            return self.cache[str(displacement)] is not None
        except KeyError:
            return False

    def has_all_cartesian_data(self, natoms: int):
        return all(self.has_data(disp) for disp in interop.AseDisplacement.iter(natoms))

    def is_empty_file(self, displacement: tp.Union[interop.AseDisplacement, str]):
        try:
            return self.cache[str(displacement)] is None
        except KeyError:
            return False

    @contextmanager
    def lock(self, displacement: tp.Union[interop.AseDisplacement, str]):
        class MyHandle:
            def __init__(self, handle):
                self._handle = handle
            def write(self, data: ElphDataset):
                self._handle.save(data._asdict())

        with self.cache.lock(str(displacement)) as handle:
            if handle is None:
                yield None
            else:
                yield MyHandle(handle)

def ensure_gpaw_setups_initialized(calc, atoms):
    """ Initializes the Setups of a GPAW instance without running a groundstate computation. """
    calc._set_atoms(atoms)  # FIXME private method
    calc.initialize()
    calc.set_positions(atoms)  # FIXME: Apparently this breaks if there is domain parallelism? What?!?!?!

# ==================================
# Steps of the procedure.  Each function caches their results, for restart purposes.

def relax_atoms(outpath, atoms):
    from ase import optimize

    if os.path.exists(outpath):
        parprint(f'Found existing {outpath}')
        return
    world.barrier()
    parprint(f'Relaxing structure... ({outpath})')

    dyn = optimize.FIRE(atoms)
    dyn.run(fmax=0.05)
    # FIXME: consider using something else to write, like pymatgen.io.vasp.Poscar with significant_figures=15.
    #        ASE always writes {:11.8f} in frac coords, which can be a dangerous amount of rounding
    #        for large unit cells.
    atoms.write(outpath, format='vasp')


# Get displacements using phonopy
def get_minimum_displacements(
        unitcell: phonopy.structure.atoms.PhonopyAtoms,
        supercell_matrix: np.ndarray,
        displacement_distance: float,
        phonopy_kw: dict = {},
):
    # note: applying phonopy_kw on load is necessary because phonopy will recompute symmetry
    parprint(f'Getting displacements... ()')
    phonon = phonopy.Phonopy(unitcell, supercell_matrix, factor=phonopy.units.VaspToTHz, **phonopy_kw)
    phonon.generate_displacements(distance=displacement_distance)
    return phonon


def make_force_sets_and_excitations(cachepath, disp_filenames, phonon, atoms, ex_kw):
    if os.path.exists(cachepath):
        parprint(f'Found existing {cachepath}')
        return np.load(cachepath)
    world.barrier()
    parprint(f'Computing force sets and polarizability data at displacements... ({cachepath})')

    eq_atoms = atoms.copy()
    def iter_displacement_files():
        eq_force_filename = disp_filenames['force']['eq']
        eq_ex_filename = disp_filenames['ex']['eq']
        yield 'eq', eq_force_filename, eq_ex_filename, eq_atoms

        disp_phonopy_sites, disp_carts = get_phonopy_displacements(phonon)
        for i, disp_atoms in enumerate(iter_displaced_structures(atoms, disp_phonopy_sites, disp_carts)):
            force_filename = disp_filenames['force']['disp'].format(i)
            ex_filename = disp_filenames['ex']['disp'].format(i)
            yield 'disp', force_filename, ex_filename, disp_atoms

    # Make files for one displacement at a time
    for disp_kind, force_filename, ex_filename, disp_atoms in iter_displacement_files():
        if os.path.exists(ex_filename):
            continue
        world.barrier()
        atoms.set_positions(disp_atoms.get_positions())

        disp_forces = atoms.get_forces()
        ex = LrTDDFT(atoms.calc, **ex_kw)
        if disp_kind == 'eq':
            # For inspecting the impact of differences in the calculator
            # between ionic relaxation and raman computation.
            parprint('Max equilibrium force during raman:', np.absolute(disp_forces).max())
        if world.rank == 0:
            np.save(force_filename, disp_forces)
        ex.write(ex_filename)

    # combine force sets into one file
    force_sets = np.array([
        np.load(disp_filenames['force']['disp'].format(i))
        for i in range(len(phonon.get_displacements()))
    ])
    np.save(cachepath, force_sets)
    for i in range(len(phonon.get_displacements())):
        os.unlink(disp_filenames['force']['disp'].format(i))
    return force_sets


def expand_raman_by_symmetry(
        cachepath,
        phonon,
        disp_filenames,
        get_polarizability,
        ex_kw,
        subtract_equilibrium_polarizability,
):
    if os.path.exists(cachepath):
        parprint(f'Found existing {cachepath}')
        return np.load(cachepath)
    world.barrier()
    parprint(f'Expanding raman data by symmetry... ({cachepath})')

    disp_phonopy_sites, disp_carts = get_phonopy_displacements(phonon)

    prim_symmetry = phonon.primitive_symmetry.get_symmetry_operations()
    lattice = phonon.primitive.get_cell()[...]
    carts = phonon.primitive.get_positions()

    oper_frac_rots = prim_symmetry['rotations']
    oper_frac_trans = prim_symmetry['translations']
    oper_cart_rots = np.array([np.linalg.inv(lattice).T @ R @ lattice.T for R in oper_frac_rots])
    oper_cart_trans = oper_frac_trans @ lattice

    oper_deperms = []
    for cart_rot, cart_trans in zip(oper_cart_rots, oper_cart_trans):
        carts = phonon.primitive.get_positions()
        transformed_carts = carts @ cart_rot.T + cart_trans
        oper_deperms.append(get_deperm(carts, transformed_carts, lattice))
    oper_deperms = np.array(oper_deperms)

    disp_tensors = np.array([
        get_polarizability(LrTDDFT.read(disp_filenames['ex']['disp'].format(i), **ex_kw))
        for i in range(len(disp_phonopy_sites))
    ])
    if subtract_equilibrium_polarizability:
        disp_tensors -= get_polarizability(LrTDDFT.read(disp_filenames['ex']['eq'], **ex_kw))

    pol_derivs = symmetry.expand_derivs_by_symmetry(
        disp_phonopy_sites,
        disp_carts,
        disp_tensors,
        symmetry.Tensor2Callbacks(),
        oper_cart_rots,
        oper_deperms,
    )
    pol_derivs = np.array(pol_derivs.tolist())  # (n,3) dtype=object --> (n,3,3,3) dtype=complex

    np.save(cachepath, pol_derivs)
    return pol_derivs


def get_eigensolutions_at_q(cachepath, phonon, q):
    if os.path.exists('eigensolutions-gamma.npz'):
        parprint('Found existing eigensolutions-gamma.npz')
        return dict(np.load(cachepath))
    world.barrier()
    parprint('Diagonalizing dynamical matrix at gamma... (eigensolutions-gamma.npz)')

    phonon.produce_force_constants()
    frequencies, eigenvectors = phonon.get_frequencies_with_eigenvectors(q)
    out = dict(
        atom_masses=phonon.masses,
        frequencies=frequencies,
        eigenvectors=eigenvectors.T, # store as rows
    )
    np.savez(cachepath, **out)
    return out


def get_mode_raman(outpath, eigendata, cart_pol_derivs):
    if os.path.exists(outpath):
        parprint(f'Found existing {outpath}')
        return
    world.barrier()
    parprint(f'Computing mode raman tensors... ({outpath})')

    cart_pol_derivs = np.load('raman-cart.npy')
    mode_pol_derivs = []
    for row in eigendata['eigenvectors']:
        mode_displacements = eigendata['atom_masses'].repeat(3) ** -0.5 * row
        mode_displacements /= np.linalg.norm(mode_displacements)

        #  ∂α_ij          ∂α_ij  ∂x_ak
        #  -----  = sum ( -----  ----- )
        #  ∂u_n     a,k   ∂x_ak  ∂u_n
        #
        #         = dot product of (3n-dimensional gradient of ∂α_ij)
        #                     with (3n-dimensional displacement vector of mode n)
        mode_pol_deriv = np.dot(
            # move i and j (axes 2 and 3) to the outside and combine axes 0 and 1 (x components)
            cart_pol_derivs.transpose((2, 3, 0, 1)).reshape((9, -1)),
            mode_displacements,
        ).reshape((3, 3))
        mode_pol_derivs.append(mode_pol_deriv)
    np.save(outpath, mode_pol_derivs)

#----------------

def get_deperm(
        carts_from,  # Nx3
        carts_to,  # Nx3
        lattice,  # 3x3 matrix or ASE Cell, each row is a lattice vector
        tol: float = 1e-5,
):
    from phonopy.structure.cells import compute_permutation_for_rotation

    # Compute the inverse permutation on coordinates, which is the
    # forward permutation on metadata ("deperm").
    #
    # I.e. ``fracs_translated[deperm] ~~ fracs_original``
    fracs_from = carts_from @ np.linalg.inv(lattice)
    fracs_to = carts_to @ np.linalg.inv(lattice)
    return compute_permutation_for_rotation(
        fracs_to, fracs_from, lattice[...].T, tol,
    )

# ==============================================================================

Displacements = tp.List[tp.Tuple[int, tp.List[float]]]
PhonopyScIndex = int  # index of a supercell atom, in phonopy's supercell ordering
AseScIndex = int  # index of a supercell atom, in ASE's supercell ordering

def get_phonopy_displacements(phonon: phonopy.Phonopy):
    """ Get displacements as arrays of ``phonopy_atom`` and ``[dx, dy, dz]`` (cartesian).

    ``phonopy_atom`` is the displaced atom index according to phonopy's supercell ordering convention.
    Mind that this is different from ASE's convention. """
    return tuple(map(list, zip(*[(i, xyz) for (i, *xyz) in phonon.get_displacements()])))

def phonopy_sc_indices_to_ase_sc_indices(phonopy_disp_atoms, natoms, supercell):
    """ Takes an array of atom indices in phonopy's supercell ordering convention and converts it to ASE's convention. """
    # use inverse perm to permute sparse indices
    deperm_phonopy_to_ase = interop.get_deperm_from_phonopy_sc_to_ase_sc(natoms, supercell)  # ase index -> phonopy index
    inv_deperm_phonopy_to_ase = np.argsort(deperm_phonopy_to_ase)  # phonopy index -> ase index
    return inv_deperm_phonopy_to_ase[phonopy_disp_atoms]  # ase indices

def iter_displaced_structures(atoms, disp_sites, disp_carts):
    # Don't use phonon.get_supercells_with_displacements as these may be translated
    # a bit relative to the original atoms if you used something like 'minimum_box'.
    # (resulting in absurd forces, e.g. all components positive at equilibrium)
    eq_atoms = atoms.copy()
    assert len(disp_sites) == len(disp_carts)
    for i, disp in zip(disp_sites, disp_carts):
        disp_atoms = eq_atoms.copy()
        positions = disp_atoms.get_positions()
        positions[i] += disp
        disp_atoms.set_positions(positions)
        yield disp_atoms

# ==============================================================================

def phonopy_atoms_to_ase(atoms):
    atoms = ase.Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
    )
    return atoms

def ase_atoms_to_phonopy(atoms):
    atoms = phonopy.structure.atoms.PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
    )
    return atoms

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import functools
from collections import defaultdict
import os
import sys

from datetime import datetime
import numpy as np
from ase.parallel import parprint, paropen, world
import phonopy
import ase.build
import gpaw
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.lrtddft import LrTDDFT
import scipy.linalg
import pickle

from ruamel.yaml import YAML
yaml = YAML(typ='rt')

from collections.abc import Sequence
from abc import ABC, abstractmethod
import warnings
import typing as tp
T = tp.TypeVar("T")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()

    parser.set_defaults(func=lambda args, log: parser.error('missing test name'))

    # New script
    p = subs.add_parser('diamond')
    p.add_argument('--brute', dest='action', action='store_const', const='brute')
    p.add_argument('--symmetry-test', dest='action', action='store_const', const='symmetry-test')
    p.set_defaults(func=lambda args, log: main__elph_diamond(action=args.action, log=log))

    # Old script is still here so that we can maintain it and work towards
    # factoring out commonalities with the new script.
    p = subs.add_parser('ch4')
    p.set_defaults(func=lambda args, log: main__raman_ch4(log=log))

    args = parser.parse_args()

    with start_log_entry('gpaw.log') as log:
        args.func(args, log)

def start_log_entry(path):
    logfile = paropen(path, 'a')
    parprint(file=logfile)
    parprint('=====================================', file=logfile)
    parprint('===', datetime.now().isoformat(), file=logfile)
    return Tee(logfile, sys.stdout)

def main__elph_diamond(action, log):
    from gpaw.elph.electronphonon import ElectronPhononCoupling
    from gpaw.lrtddft.spectrum import polarizability
    from gpaw.lrtddft import LrTDDFT

    from gpaw import GPAW, FermiDirac

    atoms = Cluster(ase.build.bulk('C'))

    params_fd = dict(
        mode='lcao',
        symmetry={"point_group": False},
        nbands = "nao",
        convergence={"bands":"all"},
        basis='dzp',
        h = 0.25,  # large for faster testing
        # NOTE: normally kpt parallelism is better, but in this script we have code
        #       that has to deal with domain parallelization, and so we need to test it
        parallel = {'domain': world.size },
        # occupations=FermiDirac(width=0.05),
        # kpts={'size': (kx_gs, ky_gs,1), 'gamma': True},
        xc='PBE',
    )
    # FIXME Need to test with an SC that's at least 3 in one direction to test translations
    supercell = (2, 2, 2)

    # ============
    # BRUTE FORCE
    if action == 'brute':
        calc_fd = GPAW(txt=log, **params_fd)
        elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True)
        elph.run()
        return

    # ============
    # SYMMETRY TEST
    # For testing the symmetry expansion of ElectronPhononCoupling data.
    # Takes a few pieces of the data produced by 'brute' and tries to produce all of the rest of the data.
    if action == 'symmetry-test':
        # a supercell exactly like ElectronPhononCoupling makes
        supercell_atoms = atoms * supercell
        quotient_perms = list(ase_repeat_translational_symmetry_perms(len(atoms), supercell))

        def get_wfs_with_sym():
            # Make a supercell exactly like ElectronPhononCoupling makes, but with point_group = True
            params_fd_sym = dict(params_fd)
            if 'symmetry' not in params_fd_sym:
                params_fd_sym['symmetry'] = dict(GPAW.default_parameters['symmetry'])
            params_fd_sym['symmetry']['point_group'] = True
            params_fd_sym['symmetry']['symmorphic'] = False  # enable full spacegroup # FIXME: doesn't work for supercells
            params_fd_sym['symmetry']['tolerance'] = 1e-6

            calc_fd_sym = GPAW(txt=log, **params_fd)
            dummy_supercell_atoms = supercell_atoms.copy()
            dummy_supercell_atoms.calc = calc_fd_sym
            calc_fd_sym._set_atoms(dummy_supercell_atoms)  # FIXME private method
            calc_fd_sym.initialize()
            calc_fd_sym.set_positions(dummy_supercell_atoms)
            return calc_fd_sym.wfs
        wfs_with_sym = get_wfs_with_sym()

        calc_fd = GPAW(txt=log, **params_fd)

        elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True)

        def read_elph(path):
            from gpaw.arraydict import ArrayDict

            array_0, arr_dic = pickle.load(open(path, 'rb'))
            atom_partition = wfs_with_sym.atom_partition
            arr_dic = ArrayDict(
                partition=atom_partition,
                shapes_a=[arr_dic[a].shape for a in range(atom_partition.natoms)],
                dtype=arr_dic[0].dtype,
                d={a:arr_dic[a] for a in atom_partition.my_indices},
            )
            arr_dic.redistribute(atom_partition.as_serial())
            return array_0, arr_dic

        # GPAW displaces the center cell for some reason instead of the first cell
        get_displaced_index = lambda prim_atom: elph.offset + prim_atom

        disp_atoms = [
            get_displaced_index(0),
            get_displaced_index(0),
            # FIXME: can't currently get full spacegroup for supercells so use extra atoms for now.
            #        (we'll need a different material if we want to test the step that generates
            #         new rows using rotational symmetries)
            get_displaced_index(1),
            get_displaced_index(1),
        ]
        disp_carts = [
            np.array([+1e02, 0, 0]),
            np.array([-1e02, 0, 0]),
            np.array([+1e02, 0, 0]),
            np.array([-1e02, 0, 0]),
        ]

        disp_values = [
            # TODO: Also transform the grid data in [0]
            read_elph('elph.0x+.pckl')[1],
            read_elph('elph.0x-.pckl')[1],
            read_elph('elph.1x+.pckl')[1],
            read_elph('elph.1x-.pckl')[1],
        ]

        lattice = supercell_atoms.get_cell()[...]
        oper_cart_rots = np.einsum('ki,slk,jl->sij', lattice, wfs_with_sym.kd.symmetry.op_scc, np.linalg.inv(lattice))
        if world.rank == 0:
            full_values = expand_derivs_by_symmetry(
                disp_atoms,       # disp -> atom
                disp_carts,       # disp -> 3-vec
                disp_values,      # disp -> T  (displaced value, optionally minus equilibrium value)
                GpawLcaoDHCallbacks(wfs_with_sym),        # how to work with T
                oper_cart_rots,   # oper -> 3x3
                oper_perms=wfs_with_sym.kd.symmetry.a_sa,       # oper -> atom' -> atom
                quotient_perms=quotient_perms,
            )
            for a in range(len(full_values)):
                for c in range(3):
                    full_values[a][c] = dict(full_values[a][c])
            pickle.dump(full_values, open('elph-full.pckl', 'wb'), protocol=2)

    # END action 'symmetry-test'

    # ============
    # TODO: test that uses phonopy for displacements.
    pass

def main__raman_ch4(log):
    from ase.build import molecule

    from gpaw.lrtddft.spectrum import polarizability

    from gpaw.cluster import Cluster
    from gpaw import GPAW, FermiDirac

    #=============================================
    # Settings

    # Input structure
    relax_grid_sep = 0.22  # GPAW finite grid size
    vacuum_sep = 3.5
    pbc = False
    def get_unrelaxed_structure():
        atoms = Cluster(molecule('CH4'))
        atoms.minimal_box(vacuum_sep, h=relax_grid_sep)
        return atoms

    # Calculator (general settings)
    make_calc = functools.partial(GPAW,
            occupations=FermiDirac(width=0.1),
            symmetry={'point_group': False},
            txt=log,
    )

    # Relaxation settings
    make_calc_relax = functools.partial(make_calc,
            h=relax_grid_sep,
    )

    # Args for computations on displaced structures
    raman_grid_sep = 0.25 # In the example, they use a larger spacing here than during relaxation.
                          # (TODO: but why? On CH4 I observe that this to leads to equilibrium forces of
                          #        0.067 ev/A, which seems to compromise our "energy minimum" state...)
    num_converged_bands = 10
    num_total_bands = 20
    make_calc_raman = functools.partial(make_calc,
            h=raman_grid_sep,
            convergence={
                'eigenstates': 1.e-5,
                'bands': num_converged_bands,
            },
            eigensolver='cg',
            nbands=num_total_bands,
    )
    supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    displacement_distance = 1e-2

    # ----------
    # Excitation settings (for polarizability)
    ex_kw = {'restrict': {'jend':num_converged_bands-1}}
    omega = 5.0 # eV
    get_polarizability = functools.partial(polarizability, omega=omega, form='v', tensor=True)
    subtract_equilibrium_polarizability = False

    # for testing purposes
    also_try_brute_force_raman = False

    #=============================================
    # Process

    disp_filenames = {
        'ex': {'eq': 'raman-eq.ex.gz', 'disp': 'raman-{:04}.ex.gz'},
        'force': {'eq': 'force-set-eq.npy', 'disp': 'force-set-{:04}.npy'},
    }

    # Relax
    unrelaxed_atoms = get_unrelaxed_structure()
    unrelaxed_atoms.pbc = pbc
    unrelaxed_atoms.calc = make_calc_relax()
    relax_atoms(outpath='relaxed.vasp', atoms=unrelaxed_atoms)

    # Phonopy displacements
    phonon = get_minimum_displacements(cachepath='phonopy_disp.yaml',
            structure_path='relaxed.vasp', supercell_matrix=supercell_matrix,
            displacement_distance=displacement_distance,
    )

    # Computing stuff at displacements
    eq_atoms = Cluster(phonopy_atoms_to_ase(phonon.supercell))
    eq_atoms.pbc = pbc
    if raman_grid_sep != relax_grid_sep:
        eq_atoms.minimal_box(vacuum_sep, h=raman_grid_sep)
    eq_atoms.calc = make_calc_raman()

    force_sets = make_force_sets_and_excitations(cachepath='force-sets.npy',
            disp_filenames=disp_filenames, phonon=phonon,
            atoms=eq_atoms, ex_kw=ex_kw,
    )
    phonon.set_forces(force_sets)

    # Applying symmetry
    cart_pol_derivs = expand_raman_by_symmetry(cachepath='raman-cart.npy',
            phonon=phonon,
            disp_filenames=disp_filenames, get_polarizability=get_polarizability, ex_kw=ex_kw,
            subtract_equilibrium_polarizability=subtract_equilibrium_polarizability,
    )

    # Phonopy part 2
    gamma_eigendata = get_eigensolutions_at_q(cachepath='eigensolutions-gamma.npz',
            phonon=phonon, q=[0, 0, 0],
    )

    # Raman of modes
    get_mode_raman(outpath='mode-raman-gamma.npy',
            eigendata=gamma_eigendata, cart_pol_derivs=cart_pol_derivs,
    )

    if also_try_brute_force_raman:
        eq_atoms = Cluster(phonopy_atoms_to_ase(phonon.supercell))
        eq_atoms.pbc = pbc
        if raman_grid_sep != relax_grid_sep:
            eq_atoms.minimal_box(vacuum_sep, h=raman_grid_sep)
        eq_atoms.calc = make_calc_raman()

        get_mode_raman_brute_force(
            eigendata=gamma_eigendata, atoms=eq_atoms, displacement_distance=displacement_distance,
            get_polarizability=get_polarizability, ex_kw=ex_kw,
        )

# ==================================
# Steps of the procedure.  Each function caches their results, for restart purposes.

def relax_atoms(outpath, atoms):
    from ase import optimize

    if os.path.exists(outpath):
        parprint(f'Found existing {outpath}')
        return
    parprint(f'Relaxing structure... ({outpath})')

    dyn = optimize.FIRE(atoms)
    dyn.run(fmax=0.05)
    # FIXME: consider using something else to write, like pymatgen.io.vasp.Poscar with significant_figures=15.
    #        ASE always writes {:11.8f} in frac coords, which can be a dangerous amount of rounding
    #        for large unit cells.
    atoms.write(outpath, format='vasp')


# Get displacements using phonopy
def get_minimum_displacements(cachepath, structure_path, supercell_matrix, displacement_distance):
    from phonopy.interface.calculator import read_crystal_structure

    if os.path.exists(cachepath):
        parprint(f'Found existing {cachepath}')
        return phonopy.load(cachepath, produce_fc=False)
    parprint(f'Getting displacements... ({cachepath})')

    unitcell, _ = read_crystal_structure(structure_path, interface_mode='vasp')
    phonon = phonopy.Phonopy(unitcell, supercell_matrix, factor=phonopy.units.VaspToTHz)
    phonon.generate_displacements(distance=displacement_distance)
    if world.rank == 0:
        phonon.save(cachepath)
    return phonopy.load(cachepath, produce_fc=False)


def make_force_sets_and_excitations(cachepath, disp_filenames, phonon, atoms, ex_kw):
    if os.path.exists(cachepath):
        parprint(f'Found existing {cachepath}')
        return np.load(cachepath)
    parprint(f'Computing force sets and polarizability data at displacements... ({cachepath})')

    eq_atoms = atoms.copy()
    def iter_displacement_files():
        eq_force_filename = disp_filenames['force']['eq']
        eq_ex_filename = disp_filenames['ex']['eq']
        yield 'eq', eq_force_filename, eq_ex_filename, eq_atoms

        for i, disp_atoms in enumerate(iter_displaced_structures(atoms, phonon)):
            force_filename = disp_filenames['force']['disp'].format(i)
            ex_filename = disp_filenames['ex']['disp'].format(i)
            yield 'disp', force_filename, ex_filename, disp_atoms

    # Make files for one displacement at a time
    for disp_kind, force_filename, ex_filename, disp_atoms in iter_displacement_files():
        if os.path.exists(ex_filename):
            continue
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


def expand_raman_by_symmetry(cachepath,
                             phonon,
                             disp_filenames,
                             get_polarizability,
                             ex_kw,
                             subtract_equilibrium_polarizability):
    if os.path.exists(cachepath):
        parprint(f'Found existing {cachepath}')
        return np.load(cachepath)
    parprint(f'Expanding raman data by symmetry... ({cachepath})')

    disp_atoms, disp_carts = map(np.array, zip(*get_displacements(phonon)))

    symmetry = phonon.primitive_symmetry.get_symmetry_operations()
    lattice = phonon.primitive.get_cell()[...]
    carts = phonon.primitive.get_positions()

    oper_frac_rots = symmetry['rotations']
    oper_frac_trans = symmetry['translations']
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
        for i in range(len(disp_atoms))
    ])
    if subtract_equilibrium_polarizability:
        disp_tensors -= get_polarizability(LrTDDFT.read(disp_filenames['ex']['eq'], **ex_kw))

    pol_derivs = expand_derivs_by_symmetry(
        disp_atoms,
        disp_carts,
        disp_tensors,
        Tensor2Callbacks(),
        oper_cart_rots,
        oper_deperms,
    )

    np.save(cachepath, pol_derivs)
    return pol_derivs


def get_eigensolutions_at_q(cachepath, phonon, q):
    if os.path.exists('eigensolutions-gamma.npz'):
        parprint('Found existing eigensolutions-gamma.npz')
        return dict(np.load(cachepath))
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


# For testing purposes: Compute raman by getting polarizability at +/- displacements along mode
def get_mode_raman_brute_force(eigendata, atoms, displacement_distance, get_polarizability, ex_kw):
    if os.path.exists('mode-raman-gamma-expected.npy'):
        parprint('Found existing mode-raman-gamma-expected.npy')
        return
    parprint('Computing mode raman tensors... (mode-raman-gamma-expected.npy)')

    eq_positions = atoms.get_positions().copy()

    mode_pol_derivs = []
    for i,row in enumerate(eigendata['eigenvectors']):
        mode_displacements = eigendata['atom_masses'][:, None] ** -0.5 * row.reshape(-1, 3)
        mode_displacements /= np.linalg.norm(mode_displacements)

        atoms.set_positions(eq_positions + mode_displacements * displacement_distance)
        atoms.get_forces()
        # FIXME: These seemingly redundant reads served a purpose at some point but I never documented it.
        #        Now that LrTDDFT has this "redesigned API" they might not even do anything at all? Test this.
        LrTDDFT(atoms.calc, **ex_kw).write(f'mode-raman-{i}+.ex.gz')
        pol_plus = get_polarizability(LrTDDFT.read(f'mode-raman-{i}+.ex.gz', **ex_kw))

        atoms.set_positions(eq_positions - mode_displacements * displacement_distance)
        atoms.get_forces()
        LrTDDFT(atoms.calc, **ex_kw).write(f'mode-raman-{i}-.ex.gz')
        pol_minus = get_polarizability(LrTDDFT.read(f'mode-raman-{i}-.ex.gz', **ex_kw))

        mode_pol_derivs.append((pol_plus - pol_minus)/(2*displacement_distance))

    np.save('mode-raman-gamma-expected.npy', mode_pol_derivs)

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

class SymmetryCallbacks(ABC, tp.Generic[T]):
    """ Class that factors out operations needed by ``expand_derivs_by_symmetry`` to make it
    general over all different sorts of data.

    Instances must not be reused for more than one call to ``expand_derivs_by_symmetry``.
    This restriction enables implementations to record data about the shape of their input if necessary.
    """
    def __init__(self):
        self.__already_init = False

    def initialize(self, obj: T):
        """ Record any data needed about the shape of T, if necessary.

        This will always be the first method called, and will be called exactly once on an
        arbitrarily-chosen item from ``disp_values``. """
        if self.__already_init:
            raise RuntimeError('SymmetryCallbacks instances must not be reused')
        self.__already_init = True

    @abstractmethod
    def flatten(self, obj: T) -> np.ndarray:
        """ Convert an object into an ndarray of ndim 1. """
        raise NotImplementedError

    @abstractmethod
    def restore(self, arr: np.ndarray) -> T:
        """ Reconstruct an object from an ndarray of ndim 1. """
        raise NotImplementedError

    @abstractmethod
    def rotate(self, obj: T, sym: int, cart_rot) -> T:
        """ Apply a spacegroup operation. """
        raise NotImplementedError

    @abstractmethod
    def permute_atoms(self, obj: T, site_deperm: np.ndarray) -> T:
        """ Apply a pure translational symmetry represented as a permutation (s' -> s). """
        raise NotImplementedError

class Tensor2Callbacks(SymmetryCallbacks):
    def flatten(self, obj):
        return obj.reshape((9,))

    def restore(self, arr):
        return arr.reshape((3, 3))

    def rotate(self, obj, sym, cart_rot):
        return _rotate_rank_2_tensor(obj, cart_rot=cart_rot)

    def permute_atoms(self, obj, deperm):
        return obj

class GpawArrayDictCallbacks(SymmetryCallbacks):
    """ Callbacks for a gpaw ``ArrayDict``. """
    def __init__(self):
        super().__init__()
        # A copy of one of the arraydicts so that we have access to the correct communicator, partition,
        # and array shapes when unflattening data.
        self.template_arraydict = None

    def initialize(self, obj):
        super().initialize(obj)

        # FIXME: proper way to check this?
        if len(obj) != obj.partition.natoms:
            if len(obj) > 0:
                raise RuntimeError('symmetry expansion must be done in serial; try arr.redistribute(arr.partition.as_serial())')
            else:
                raise RuntimeError('symmetry expansion must only be done on root node')

        template = obj.copy()
        for key in template:
            template[key] = np.empty_like(template[key])
        self.template_arraydict = template

    def flatten(self, obj):
        return np.concatenate([a.reshape(-1) for a in obj.values()])

    def restore(self, arr):
        sizes = [np.product(shape) for shape in self.template_arraydict.shapes_a]
        splits = np.cumsum(sizes)[:-1]
        arrs_a = np.split(arr, splits)

        out_a = self.template_arraydict.copy()
        for a in range(out_a.partition.natoms):
            out_a[a][...] = arrs_a[a].reshape(out_a.shapes_a[a])
        return out_a

    def permute_atoms(self, obj, deperm):
        out_a = self.template_arraydict.copy()
        for anew, aold in enumerate(deperm):
            out_a[anew] = obj[aold]
        return out_a

class GpawLcaoDHCallbacks(GpawArrayDictCallbacks):
    """ Callbacks for ``calc.hamiltonian.dH_asp`` in GPAW LCAO mode. """
    def __init__(self, wfs_with_symmetry: gpaw.wavefunctions.base.WaveFunctions):
        super().__init__()
        self.wfs = wfs_with_symmetry

    def rotate(self, obj, sym, cart_rot):
        from gpaw.utilities import pack, unpack2

        # FIXME: It's *possible* that this is actually applying the inverse of sym instead of sym itself.
        #        It's hard for me to tell since it would not ultimately impact program output, and the conventions
        #        for how gpaw's a_sa and R_sii are stored are unclear to me.

        dH_asp = obj
        out_asp = obj.copy()
        a_a = self.wfs.kd.symmetry.a_sa[sym]

        for adest in dH_asp.keys():
            asrc = a_a[adest]
            R_ii = self.wfs.setups[adest].R_sii[sym]
            for s in range(self.wfs.nspins):
                dH_p = dH_asp[asrc][s]
                dH_ii = unpack2(dH_p)
                tmp_ii = R_ii @ dH_ii @ R_ii.T
                tmp_p = pack(tmp_ii)
                out_asp[adest][s][...] = tmp_p

        return out_asp

# ==============================================================================

def expand_derivs_by_symmetry(
    disp_atoms,       # disp -> atom
    disp_carts,       # disp -> 3-vec
    disp_values,      # disp -> T  (displaced value, optionally minus equilibrium value)
    callbacks,        # how to work with T
    oper_cart_rots,   # oper -> 3x3
    oper_perms,       # oper -> atom' -> atom
    quotient_perms=None,   # oper -> atom' -> atom
) -> np.ndarray:
    """
    Generic function that uses symmetry to expand finite difference data for derivatives of any
    kind of data structure ``T``.

    This takes data computed at a small number of displaced structures that are distinct under
    symmetry, and applies the symmetry operators in the spacegroup (and internal translational
    symmetries for supercells) to compute derivatives with respect to all cartesian coordinates
    of all atoms in the structure.

    E.g. ``T`` could be residual forces of shape (natom,3) to compute the force constants matrix,
    or it could be 3x3 polarizability tensors to compute all raman tensors.  Or it could be something
    else entirely; simply supply the appropriate ``callbacks``.

    :param disp_atoms: shape (ndisp,), dtype int.  Index of the displaced atom for each displacement.

    :param disp_carts: shape (ndisp,3), dtype float.  The displacement vectors, in cartesian coords.

    :param disp_values: sequence type of length ``ndisp`` holding ``T`` objects for each displacement.
        These are either ``T_disp - T_eq`` or ``T_disp``, where ``T_eq`` is the value at equilibrium and
        ``T_disp`` is the value after displacement.

    :param callbacks: ``SymmetryCallbacks`` instance defining how to apply symmetry operations to ``T``,
        and how to convert back and forth between ``T`` and a 1D array of float or complex.

    :param oper_cart_rots: shape (nsym,3,3), dtype float.  For each spacegroup or pointgroup operator,
        its representation as a 3x3 rotation/mirror matrix that operates on column vectors containing
        Cartesian data.  (for spacegroup operators, the translation vectors are not needed, because
        their impact is already accounted for in ``oper_perms``)

    :param oper_perms: shape (nsym,nsite), dtype int.  For each spacegroup or pointgroup operator, its
        representation as a permutation that operates on site metadata (see the notes below).

    :param quotient_perms: shape (nquotient,nsite), dtype int, optional.  If the structure is a supercell
        of a periodic structure, then this should contain the representations of all pure translational
        symmetries as permutations that operate on site metadata (see the notes below).  Note that, as an
        alternative to this argument, it possible to instead include these pure translational symmetries
        in ``oper_perms/oper_cart_rots``... but using ``quotient_perms`` will be more efficient.

    :return:
        Returns a shape ``(natom, 3)`` array of ``T`` where the item at ``(a, k)`` is the derivative of
        the value with respect to cartesian component ``k`` of the displacement of atom ``a``.
        Note that the output is *always* 2-dimensional with ``dtype=object``, even if ``T`` is an array type.
        (so the output may be an array of arrays).  This is done because numpy's overly eager array detection
        could easily lead to data loss if allowed to run unchecked on ``T``.  If you want to reconstruct a
        single array, try ``np.array(output.tolist()).tolist()``.

    ..note::
        This function is serial and requires a process to have access to data for all atoms.

    ..note::
        This function does not require any assumptions of periodicity and should work equally well
        on isolated molecules (even those with spacegroup-incompatible operators like C5).

    ..note::
        For each star of symmetrically equivalent sites, precisely one site must appear in ``disp_atoms``.
        (if more than one site in the star has displacements, some of the input data may be ignored)

    ..note::
        For best results, once the input displacements are expanded by symmetry, there should be data
        at both positive and negative displacements along each of three linearly independent axes for each site.
        Without negative displacements, the results will end up being closer to a forward difference
        rather than a central difference (and will be wholly inaccurate if equilibrium values were not subtracted).

        The displacements chosen by `Phonopy <https://phonopy.github.io/phonopy/>` meet this criterion.

    ..note::
        The precise definition of the permutations is as follows: Suppose that you have an
        array of coordinates ``carts`` (shape ``(nsite,3)``) and an array of data ``data`` (shape ``(nsite,)``).
        Then, for any given spacegroup operation with rotation ``rot``, translation ``trans``, and permutation ``perm`,
        pairing ``carts @ rot.T + trans`` with ``data`` should produce a scenario equivalent to pairing ``carts``
        with ``data[perm]`` (using `integer array indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html#integer-array-indexing>`).
        In this manner, ``perm`` essentially represents the action of the operator
        on metadata when coordinate data is fixed.

        Equivalently, it is the *inverse* of the permutation that operates on the coordinates.
        This is to say that ``(carts @ rot.T + trans)[perm]`` should be equivalent (under lattice translation)
        to the original ``carts``.
    """

    assert len(disp_carts) == len(disp_atoms) == len(disp_values)
    assert len(oper_cart_rots) == len(oper_perms)

    disp_values = list(disp_values)
    callbacks.initialize(disp_values[0])

    # Proper way to apply data permutations to sparse indices
    oper_inv_perms = np.argsort(oper_perms, axis=1) # oper -> atom -> atom'
    permute_index = lambda oper, site: oper_inv_perms[oper, site]

    # For each representative atom that gets displaced, gather all of its displacements.
    representative_disps = defaultdict(list)
    for (disp, representative) in enumerate(disp_atoms):
        representative_disps[representative].append(disp)

    if quotient_perms is None:
        # Just the identity.
        quotient_perms = np.array([np.arange(len(oper_perms[0]))])
    quotient_inv_perms = np.argsort(quotient_perms, axis=1) # quotient -> atom -> atom'

    # Compute derivatives with respect to displaced (representative) atoms
    def compute_representative_row(representative):
        # Expand the available data using the site-symmetry operators to ensure
        # we have enough independent equations for pseudoinversion.
        eq_cart_disps = []  # equation -> 3-vec
        eq_rhses = []  # equation -> flattened T
        for oper, cart_rot in enumerate(oper_cart_rots):
            if permute_index(oper, representative) != representative:
                continue  # not site-symmetry oper

            for disp in representative_disps[representative]:
                rotated = callbacks.rotate(disp_values[disp], oper, cart_rot=cart_rot)
                eq_cart_disps.append(cart_rot @ disp_carts[disp])
                eq_rhses.append(callbacks.flatten(rotated))  # flattened tensor

        # Solve for Q in the overconstrained system   eq_cart_disps   Q   = eq_rhses
        #                                                (?x3)      (3xM) =  (?xM)
        #
        # (M is the length of the flattened representation of T).
        # The columns of Q are the cartesian gradients of each scalar component of T
        # with respect to the representative atom.
        pinv, rank = scipy.linalg.pinv(eq_cart_disps, return_rank=True)
        solved = pinv @ np.array(eq_rhses)
        assert rank == 3, "site symmetry too small! (rank: {})".format(rank)
        assert len(solved) == 3

        # Important not to use array() here because this contains values of type T.
        return [callbacks.restore(x) for x in solved]

    # atom -> cart axis -> T
    # I.e. atom,i -> partial T / partial x_(atom,i)
    site_derivatives = {rep: compute_representative_row(rep) for rep in representative_disps}

    # Fill out more rows (i.e. derivatives w.r.t. other atoms) by applying spacegroup symmetry
    for representative in representative_disps:
        # Find the atoms we can reach from the representative (and a symmetry operation that sends it to each one)
        for oper, cart_rot in enumerate(oper_cart_rots):
            newsite = permute_index(oper, representative)
            if newsite not in site_derivatives:
                # Apply the rotation to the inner dimensions of the gradient (i.e. rotate each T)
                t_derivs_by_axis = [callbacks.rotate(deriv, oper, cart_rot) for deriv in site_derivatives[representative]]
                # Apply the rotation to the outer axis of the gradient
                array_derivs_by_axis = [callbacks.flatten(t) for t in t_derivs_by_axis]
                array_derivs_by_axis = cart_rot @ array_derivs_by_axis
                t_derivs_by_axis = [callbacks.restore(arr) for arr in array_derivs_by_axis]

                site_derivatives[newsite] = t_derivs_by_axis

    # If this is a supercell, fill out all remaining rows by applying pure translational symmetries
    old_site_derivatives = dict(site_derivatives)
    site_derivatives = {}
    for oldsite in old_site_derivatives:
        for quotient in range(len(quotient_perms)):
            newsite = quotient_inv_perms[quotient, oldsite]
            site_derivatives[newsite] = [
                callbacks.permute_atoms(derivative, quotient_perms[quotient])
                for derivative in old_site_derivatives[oldsite]
            ]

    # site_derivatives should now be dense
    natoms = len(oper_perms[0])
    missing_indices = set(range(natoms)) - set(site_derivatives)
    if missing_indices:
        raise RuntimeError(f'no displaced atoms were symmetrically equivalent to these indices: {sorted(missing_indices)}!')

    # Convert to array, in a manner that prevents numpy from detecting the dimensions of T.
    final_out = np.empty((natoms, 3), dtype=object)
    final_out[...] = [site_derivatives[i] for i in range(natoms)]
    return final_out

def _rotate_rank_2_tensor(tensor, cart_rot):
    assert tensor.shape == (3, 3)
    return cart_rot @ tensor @ cart_rot.T

AtomIndex = int
QuotientIndex = int
OperIndex = int

class CombinedOperator(tp.NamedTuple):
    """ Represents a symmetry operation that is the composition of a space group/pointgroup
    operation followed by a pure translation.

    Attributes
        oper      Index of space group/pointgroup operator.
        quotient  Index of pure translation operator (from the quotient group of a primitive lattice and a superlattice).
    """
    oper: OperIndex
    quotient: QuotientIndex

class FromRepInfo(tp.NamedTuple):
    """ Describes the ways to reach a given site from a representative atom.
    
    Attributes
        rep         Atom index of the representative that can reach this site.
        operators   List of operators, each of which individually maps ``rep`` to this site.
                    If this site is itself a representative (and thus is its own ``rep``),
                    then this is effectively the set of site symmetry.
    """
    rep: AtomIndex
    operators: tp.List[CombinedOperator]

class PrecomputedSymmetryIndexInfo:
    """ A class that records how to reach each atom from a predetermined set of symmetry representatives.

    Attributes:
        from_reps   dict. For each atom, a ``FromRepInfo`` describing how to reach that atom.
    """
    from_reps: tp.Dict[AtomIndex, FromRepInfo]

    def __init__(self,
            representatives: tp.Iterable[AtomIndex],
            oper_deperms,      # oper -> site' -> site
            quotient_deperms,  # quotient -> site' -> site
    ):
        redundant_reps = []
        from_reps: tp.Dict[AtomIndex, FromRepInfo] = {}

        # To permute individual sparse indices in O(1), we need the inverse perms
        oper_inv_deperms = np.argsort(oper_deperms, axis=1)  # oper -> site -> site'
        quotient_inv_deperms = np.argsort(quotient_deperms, axis=1)  # quotient -> site -> site'

        for rep in representatives:
            if rep in from_reps:
                redundant_reps.append(rep)
                continue

            for quotient in range(len(quotient_inv_deperms)):
                for oper in range(len(oper_inv_deperms)):
                    # find the site that rep gets sent to
                    site = oper_inv_deperms[oper][rep]
                    site = quotient_inv_deperms[quotient][site]
                    if site not in from_reps:
                        from_reps[site] = FromRepInfo(rep, [])
                    from_reps[site].operators.append(CombinedOperator(oper, quotient))

        if redundant_reps:
            message = ', '.join('{} (~= {})'.format(a, from_reps[a].rep) for a in redundant_reps)
            raise RuntimeError('Redundant atoms in representative list:  {}'.format(message))

        self.from_reps = from_reps

def _rotate_rank_3_tensor(tensor, cart_rot):
    assert tensor.shape == (3, 3, 3)
    return np.einsum('ia,jb,kc,abc->ijk', cart_rot, cart_rot, cart_rot, tensor)

# ==============================================================================

def get_displacements(phonon):
    """ Get displacements as list of [supercell_atom, [dx, dy, dz]] """
    return [[i, xyz] for (i, *xyz) in phonon.get_displacements()]

def iter_displaced_structures(atoms, phonon):
    # Don't use phonon.get_supercells_with_displacements as these may be translated
    # a bit relative to the original atoms if you used something like 'minimum_box'.
    # (resulting in absurd forces, e.g. all components positive at equilibrium)
    assert len(atoms) == len(phonon.supercell)

    eq_atoms = atoms.copy()
    for i, disp in get_displacements(phonon):
        disp_atoms = eq_atoms.copy()
        positions = disp_atoms.get_positions()
        positions[i] += disp
        disp_atoms.set_positions(positions)
        yield disp_atoms

# ==============================================================================

def ase_repeat_translational_symmetry_perms(natoms, repeats):
    """ Get the full quotient group of pure translational symmetries of ``atoms * repeats``.
    
    The order of the output is deterministic but unspecified. """
    if isinstance(repeats, int):
        repeats = (repeats, repeats, repeats)

    # It is unclear whether ASE actually specifies the order of atoms in a supercell anywhere,
    # so be ready for the worst.  (even though this we don't specify order of operators returned,
    # our handling of repeats like (2, 3, 1) will be totally incorrect if the convention is wrong)
    if not np.array_equal(repeats, (1, 1, 1)):
        __check_ase_repeat_convention_hasnt_changed()

    # fastest index is atoms, then repeats[2], then repeats[1], then repeats[0]

    def all_cyclic_perms_of_len(n):
        # e.g. for n=4 this gives [[0,1,2,3], [1,2,3,0], [2,3,0,1], [3,0,1,2]]
        return np.add.outer(np.arange(n), np.arange(n)) % n

    n_a, n_b, n_c = repeats
    atom_perm = np.arange(natoms)  # we never rearrange the atoms within a cell
    for a_perm in all_cyclic_perms_of_len(n_a):
        for b_perm in all_cyclic_perms_of_len(n_b):
            for c_perm in all_cyclic_perms_of_len(n_c):
                yield permutation_outer_product(a_perm, b_perm, c_perm, atom_perm)

def __check_ase_repeat_convention_hasnt_changed():
    # A simple structure with an identity matrix for its cell so that atoms in the supercell
    # have easily-recognizable positions.
    unitcell = ase.Atoms(symbols=['X', 'Y'], positions=[[0, 0, 0], [0.5, 0, 0]], cell=np.eye(3))
    sc_positions = (unitcell * (4, 3, 5)).get_positions()
    if not all([
        np.all(sc_positions[0].round(8) == [0, 0, 0]),
        np.all(sc_positions[1].round(8) == [0.5, 0, 0]),    # fastest index: primitive
        np.all(sc_positions[2].round(8) == [0, 0, 1]),      # ...followed by 3rd cell vector
        np.all(sc_positions[2*5].round(8) == [0, 1, 0]),    # ...followed by 2nd cell vector
        np.all(sc_positions[2*5*3].round(8) == [1, 0, 0]),  # ...followed by 1st cell vector
    ]):
        raise RuntimeError('ordering of atoms in ASE supercells has changed!')

def permutation_outer_product(*perms):
    """ Compute the mathematical outer product of a sequence of permutations.
    
    The result is a permutation that operates on an array whose length is the product of all of the
    input perms.  The last perm will be the fastest index in the output (rearranging items within
    blocks), while the first perm will be the slowest (rearranging the blocks themselves).
    """
    from functools import reduce

    lengths = [len(p) for p in perms]  # na, nb, ..., ny, nz
    strides = np.multiply.accumulate([1] + lengths[1:][::-1])[::-1]   #   ..., nx*ny*nz, ny*nz, nz, 1

    premultiplied_perms = [stride * np.array(perm) for (stride, perm) in zip(strides, perms)]
    permuted_n_dimensional = reduce(np.add.outer, premultiplied_perms)

    # the thing we just computed is basically what you would get if you started with
    #  np.arange(product(lengths)).reshape(lengths) and permuted each axis.
    return permuted_n_dimensional.ravel()

# ==============================================================================

def phonopy_atoms_to_ase(atoms):
    atoms = ase.Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
    )
    return atoms

class Tee :
    def __init__(self, *fds):
        self.fds = list(fds)

    def write(self, text):
        for fd in self.fds:
            fd.write(text)

    def flush(self):
        for fd in self.fds:
            fd.flush()

    def closed(self):
        return False

    def __enter__(self, *args, **kw):
        for i, fd in enumerate(self.fds):
            if fd not in [sys.stdout, sys.stderr] and hasattr(fd, '__enter__'):
                self.fds[i] = self.fds[i].__enter__(*args, **kw)
        return self

    def __exit__(self, *args, **kw):
        for fd in self.fds:
            if fd not in [sys.stdout, sys.stderr] and hasattr(fd, '__exit__'):
                fd.__exit__(*args, **kw)

if __name__ == '__main__':
    main()

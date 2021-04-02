#!/usr/bin/env python3

from . import symmetry
from . import interop
from . import utils

import functools
import os
import sys
import copy

from datetime import datetime
import numpy as np
from ase.parallel import parprint, paropen, world
import phonopy
import ase.build
import gpaw
from gpaw.lrtddft import LrTDDFT
import pickle

from ruamel.yaml import YAML
yaml = YAML(typ='rt')

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
    p.add_argument('--supercell', type=(lambda s: tuple(map(int, s))), dest='supercell', default=(1,1,1))
    p.set_defaults(func=lambda args, log: main__elph_diamond(action=args.action, supercell=args.supercell, log=log))

    # Old script is still here so that we can maintain it and work towards
    # factoring out commonalities with the new script.
    p = subs.add_parser('raman')
    p.set_defaults(structure='ch4')
    p.add_argument('--mos2', action='store_const', dest='structure', const='mos2')
    p.add_argument('--supercell', type=(lambda s: tuple(map(int, s))), dest='supercell', default=(1,1,1))
    p.set_defaults(func=lambda args, log: main__raman_ch4(structure=args.structure, supercell=args.supercell, log=log))

    args = parser.parse_args()

    with start_log_entry('gpaw.log') as log:
        args.func(args, log)

def start_log_entry(path):
    logfile = paropen(path, 'a')
    parprint(file=logfile)
    parprint('=====================================', file=logfile)
    parprint('===', datetime.now().isoformat(), file=logfile)
    return utils.Tee(logfile, sys.stdout)

def main__elph_diamond(supercell, action, log):
    from gpaw.elph.electronphonon import ElectronPhononCoupling
    from gpaw.lrtddft.spectrum import polarizability
    from gpaw.lrtddft import LrTDDFT
    from gpaw import GPAW, FermiDirac
    from gpaw.cluster import Cluster

    atoms = Cluster(ase.build.bulk('C'))

    params_fd = dict(
        mode='lcao',
        symmetry={"point_group": False},
        nbands = "nao",
        convergence={"bands":"all"},
        basis='dzp',
        h = 0.3,  # large for faster testing
        # NOTE: normally kpt parallelism is better, but in this script we have code
        #       that has to deal with domain parallelization, and so we need to test it
        parallel = {'domain': world.size },
        # occupations=FermiDirac(width=0.05),
        kpts={'size': (2, 2, 2), 'gamma': False},
        xc='PBE',
    )

    # ============
    # BRUTE FORCE
    if action == 'brute':
        calc_fd = GPAW(txt=log, **params_fd)

        if not os.path.exists('gs.gpw'):
            # NOTE: original script used different parameters here (params_gs)  _/o\_
            calc_gs = GPAW(**params_fd)
            atoms.calc = calc_gs
            atoms.get_potential_energy()
            atoms.calc.write("gs.gpw", mode="all")
        else:
            calc_gs = GPAW('gs.gpw')

        # This makes the real space grid points identical to the primitive cell computation.
        # (by increasing the counts by a factor of a supercell dimension)
        params_fd['gpts'] = calc_gs.wfs.gd.N_c * list(supercell)
        if 'h' in params_fd:
            del params_fd['h']

        elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True)
        elph.run()
        calc_gs.wfs.gd.comm.barrier()
        elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell)
        elph.set_lcao_calculator(calc_fd)
        elph.calculate_supercell_matrix(dump=1)
        return

    # ============
    # SYMMETRY TEST
    # For testing the symmetry expansion of ElectronPhononCoupling data.
    # Takes a few pieces of the data produced by 'brute' and tries to produce all of the rest of the data.
    if action == 'symmetry-test':
        # a supercell exactly like ElectronPhononCoupling makes
        supercell_atoms = atoms * supercell
        quotient_perms = list(interop.ase_repeat_translational_symmetry_perms(len(atoms), supercell))

        # IMPORTANT: The real space grid of the calculators we use now must match
        #            the ones we had during the brute force computation.
        params_fd['gpts'] = GPAW('gs.gpw').wfs.gd.N_c * list(supercell)
        if 'h' in params_fd:
            del params_fd['h']

        def get_wfs_with_sym():
            # Make a supercell exactly like ElectronPhononCoupling makes, but with point_group = True
            params_fd_sym = copy.deepcopy(params_fd)
            if 'symmetry' not in params_fd_sym:
                params_fd_sym['symmetry'] = dict(GPAW.default_parameters['symmetry'])
            params_fd_sym['symmetry']['point_group'] = True
            params_fd_sym['symmetry']['symmorphic'] = False  # enable full spacegroup # FIXME: doesn't work for supercells
            params_fd_sym['symmetry']['tolerance'] = 1e-6

            calc_fd_sym = GPAW(txt=log, **params_fd_sym)
            dummy_supercell_atoms = supercell_atoms.copy()
            dummy_supercell_atoms.calc = calc_fd_sym
            calc_fd_sym._set_atoms(dummy_supercell_atoms)  # FIXME private method
            calc_fd_sym.initialize()
            calc_fd_sym.set_positions(dummy_supercell_atoms)
            return calc_fd_sym.wfs
        wfs_with_sym = get_wfs_with_sym()

        calc_fd = GPAW(txt=log, **params_fd)

        elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True).offset
        cell_offset = elph.offset
        del elph  # that's all we needed it for

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
        get_displaced_index = lambda prim_atom: cell_offset * len(atoms) + prim_atom

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
            np.array([+1e-2, 0, 0]),
            np.array([-1e-2, 0, 0]),
            np.array([+1e-2, 0, 0]),
            np.array([-1e-2, 0, 0]),
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
            full_values = symmetry.expand_derivs_by_symmetry(
                disp_atoms,       # disp -> atom
                disp_carts,       # disp -> 3-vec
                disp_values,      # disp -> T  (displaced value, optionally minus equilibrium value)
                symmetry.GpawLcaoDHCallbacks(wfs_with_sym),        # how to work with T
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

def main__raman_ch4(structure, supercell, log):
    import ase.build

    from gpaw.lrtddft.spectrum import polarizability
    from gpaw.cluster import Cluster
    from gpaw import GPAW, FermiDirac

    #=============================================
    # Settings

    # Input structure
    relax_grid_sep = 0.22  # GPAW finite grid size
    vacuum_sep = 3.5
    pbc = False
    if structure == 'ch4':
        def get_unrelaxed_structure():
            atoms = Cluster(ase.build.molecule('CH4'))
            atoms.minimal_box(vacuum_sep, h=relax_grid_sep)
            return atoms
    elif structure == 'mos2':
            atoms = Cluster(ase.build.mx2('MoS2'))
            atoms.center(vacuum=vacuum_sep, axis=2)

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
    supercell_matrix = [[supercell[0], 0, 0], [0, supercell[1], 0], [0, 0, supercell[2]]]
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
        for i in range(len(disp_atoms))
    ])
    if subtract_equilibrium_polarizability:
        disp_tensors -= get_polarizability(LrTDDFT.read(disp_filenames['ex']['eq'], **ex_kw))

    pol_derivs = symmetry.expand_derivs_by_symmetry(
        disp_atoms,
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

def phonopy_atoms_to_ase(atoms):
    atoms = ase.Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
    )
    return atoms

if __name__ == '__main__':
    main()

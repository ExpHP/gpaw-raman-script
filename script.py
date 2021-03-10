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

from ruamel.yaml import YAML
yaml = YAML(typ='rt')

from collections.abc import Sequence
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
T = TypeVar("T")

def main():
    if sys.argv[1] == 'diamond':
        # Run new script
        with start_log_entry('gpaw.log') as log:
            main__elph_diamond(log=log)
        return
    if sys.argv[1] == 'ch4':
        # Old script is still here so that we can maintain it and work towards
        # factoring out commonalities with the new script.
        with start_log_entry('gpaw.log') as log:
            main__raman_ch4(log=log)
        return
    assert False, 'invalid test {}'.format(sys.argv[1])

def start_log_entry(path):
    logfile = paropen(path, 'a')
    parprint(file=logfile)
    parprint('=====================================', file=logfile)
    parprint('===', datetime.now().isoformat(), file=logfile)
    return Tee(logfile, sys.stdout)

def main__elph_diamond(log):
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
    supercell = (2, 2, 2)
    params_fd_sym = dict(params_fd)
    params_fd_sym['point_group'] = True

    # BRUTE FORCE
    if True:
        calc_fd = GPAW(txt=log, **params_fd)
        elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True)
        elph.run()
        return

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
    make_calc_relax = functools.partial(GPAW,
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

    pol_derivs = rank_2_tensor_derivs_by_symmetry(
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

class SymmetryCallbacks(ABC, Generic[T]):
    @abstractmethod
    def flatten(self, obj: T) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def restore(self, arr: np.ndarray) -> T:
        raise NotImplementedError

    @abstractmethod
    def rotate(self, obj: T, sym: int, cart_rot) -> T:
        raise NotImplementedError

class Tensor2Callbacks(SymmetryCallbacks):
    def flatten(self, obj):
        return obj.reshape((9,))

    def restore(self, arr):
        return arr.reshape((3, 3))

    def rotate(self, obj, sym, cart_rot):
        return _rotate_rank_2_tensor(obj, cart_rot=cart_rot)

class GpawArrayDictCallbacks(SymmetryCallbacks):
    """ Callbacks for a gpaw ``ArrayDict``. """
    def __self__(self):
        # A copy of one of the arraydicts so that we have access to the correct communicator, partition,
        # and array shapes when unflattening data.
        self.template_arraydict = None

    def flatten(self, obj):
        import gpaw
        assert isinstance(obj, gpaw.arraydict.ArrayDict)

        # FIXME: proper way to check this?
        if len(obj) != obj.partition.natoms:
            if len(obj) > 0:
                raise RuntimeError('symmetry expansion must be done in serial; try arr.redistribute(arr.partition.as_serial())')
            else:
                raise RuntimeError('symmetry expansion must only be done on root node')

        if self.template_arraydict is None:
            self.template_arraydict = obj.copy()
        else:
            assert np.all(self.template_arraydict.shapes_a == obj.shapes_a), "instance of {} cannot be reused for different ArrayDicts".format(type(self).__name__)

        return np.concatenate([a.reshape(-1) for a in obj.values()])

    def restore(self, arr):
        sizes = [np.product(shape) for shape in self.template_arraydict.shapes_a]
        splits = np.cumsum(sizes)[:-1]
        arrs_a = np.split(arr, splits)

        out_a = self.template_arraydict.copy()
        for a in range(out_a.partition.natoms):
            out_a[a][...] = arrs_a[a].reshape(out_a.shapes_a[a])
        return out_a

class GpawLcaoDHCallbacks(GpawArrayDictCallbacks):
    """ Callbacks for ``calc.hamiltonian.dH_asp`` in GPAW LCAO mode. """
    def __init__(self, wfs_with_symmetry: gpaw.setup.Setups):
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

def rank_2_tensor_derivs_by_symmetry(
    disp_atoms,       # disp -> atom
    disp_carts,       # disp -> 3-vec
    disp_values,      # disp -> T  (displaced value, optionally minus equilibrium value)
    callbacks,        # how to work with T
    oper_cart_rots,   # oper -> 3x3
    oper_perms,       # oper -> atom' -> atom
) -> np.ndarray:
    """
    Uses symmetry to expand finite difference data for derivatives of a rank 2 tensor.

    This takes data computed about a rank 2 tensor at a small number of displaced structures that
    are distinct under symmetry, and applies the symmetry operators in the spacegroup to compute
    derivatives with respect to all cartesian coordinates of all atoms in the structure.

    :param disp_atoms: shape (ndisp,), dtype int.  Index of the displaced atom for each displacement.
    :param disp_carts: shape (ndisp,3), dtype float.  The displacement vectors, in cartesian coords.
    :param disp_values: sequence type of length ``ndisp`` holding ``T`` objects for each displacement.
        These are either ``T_disp - T_eq`` or ``T_disp``, where ``T_eq`` is the value at equilibrium and
        ``T_disp`` is the value after displacement.
    :param oper_cart_rots: shape (nsym,3,3), dtype float.  For each spacegroup operator, its representation
        as a 3x3 rotation/mirror matrix that operates on column vectors containing Cartesian data.
    :param oper_perms: shape (nsym,nsite), dtype int.  For each spacegroup operator, its representation
        as a permutation that operates on site metadata.
    :return:
        Returns a shape ``(natom, 3)`` array of ``T`` where the item at ``(a, k)`` is the derivative of the
        value with respect to cartesian component ``k`` of the displacement of atom ``a``.  (note that if ``T``
        is itself an array-like type, its axes may be merged with the outer array as per the typical behavior
        of arrays; e.g. if ``T`` is a 3x3 array you will likely get an array of shape ``(natom, 3, 3, 3)``)

    ..note::
        This function is serial and requires a process to have access to data for all atoms.

    ..note::
        This function does not require any assumptions of periodicity and should work equally well
        on isolated molecules (even those with spacegroup-incompatible operators like C5).

    ..note::
        For each star of symmetrically equivalent sites, precisely one site must appear in ``disp_atoms``.
        (if more than one site in the star has displacements, some of the input data may be ignored)

    ..note::
        For best results, when the input displacements are expanded by symmetry, it should produce data
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

    # Proper way to apply data permutations to sparse indices
    oper_inv_perms = np.argsort(oper_perms, axis=1) # oper -> atom -> atom'
    permute_index = lambda oper, site: oper_inv_perms[oper, site]

    # For each representative atom that gets displaced, gather all of its displacements.
    representative_disps = defaultdict(list)
    for (disp, representative) in enumerate(disp_atoms):
        representative_disps[representative].append(disp)

    # Compute derivatives with respect to displaced (representative) atoms
    def compute_representative_row(representative):
        # Expand the available data using the site-symmetry operators to ensure
        # we have enough independent equations for pseudoinversion.
        eq_cart_disps = []  # equation -> 3-vec
        eq_rhses = []  # equation -> flattened 3x3 mat
        for oper, cart_rot in enumerate(oper_cart_rots):
            if permute_index(oper, representative) != representative:
                continue  # not site-symmetry oper

            for disp in representative_disps[representative]:
                rotated = callbacks.rotate(disp_values[disp], oper, cart_rot=cart_rot)
                eq_cart_disps.append(cart_rot @ disp_carts[disp])
                eq_rhses.append(callbacks.flatten(rotated))  # flattened tensor

        # Solve for Q in the overconstrained system   eq_cart_disps   Q   = eq_rhses
        #                                                (?x3)      (3x9) =  (?x9)
        #
        # The columns of Q are the cartesian gradients of each tensor component
        # with respect to the representative atom.
        solved = np.linalg.pinv(eq_cart_disps) @ np.array(eq_rhses)
        return np.array([callbacks.restore(x) for x in solved])

    # atom -> 3x3x3 tensor where last two indices index the tensor.
    # I.e. atom,i,j,k -> partial T_jk / partial x_(atom,i)
    site_derivatives = {rep: compute_representative_row(rep) for rep in representative_disps}

    # Compute derivatives with respect to other atoms by symmetry
    for representative in representative_disps:
        # Find which representative is related to this atom, and any symmetry operation that will send it here
        for oper, cart_rot in enumerate(oper_cart_rots):
            site = permute_index(oper, representative)
            if site not in site_derivatives:
                # Apply the rotation to the inner dimensions of the gradient (i.e. rotate each T)
                t_derivs_by_axis = [callbacks.rotate(deriv, oper_inv_perms[oper], cart_rot) for deriv in site_derivatives[representative]]
                # Apply the rotation to the outer axis of the gradient
                array_derivs_by_axis = [callbacks.flatten(t) for t in t_derivs_by_axis]
                array_derivs_by_axis = cart_rot @ array_derivs_by_axis
                t_derivs_by_axis = [callbacks.restore(arr) for arr in array_derivs_by_axis]

                site_derivatives[site] = t_derivs_by_axis

    # site_derivatives is now dense, so change to ndarray
    missing_indices = set(range(len(site_derivatives))) - set(site_derivatives)
    if missing_indices:
        raise RuntimeError(f'no displaced atoms were symmetrically equivalent to index {next(iter(missing_indices))}!')
    return np.array([site_derivatives[i] for i in range(len(site_derivatives))])

def _rotate_rank_2_tensor(tensor, cart_rot):
    assert tensor.shape == (3, 3)
    return cart_rot @ tensor @ cart_rot.T

def _rotate_rank_3_tensor(tensor, cart_rot):
    assert tensor.shape == (3, 3, 3)
    return np.einsum('ia,jb,kc,abc->ijk', cart_rot, cart_rot, cart_rot, tensor)

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

def phonopy_atoms_to_ase(atoms):
    atoms = ase.Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
    )
    return atoms

class Tee :
    def __init__(self, *fds):
        self.fds = fds

    def write(self, text):
        for fd in self.fds:
            fd.write(text)

    def flush(self):
        for fd in self.fds:
            fd.flush()

    def __enter__(self, *args, **kw):
        for fd in self.fds:
            if fd not in [sys.stdout, sys.stderr]:
                fd.__enter__(*args, **kw)

    def __exit__(self, *args, **kw):
        for fd in self.fds:
            if fd not in [sys.stdout, sys.stderr]:
                fd.__exit__(*args, **kw)

if __name__ == '__main__':
    main()

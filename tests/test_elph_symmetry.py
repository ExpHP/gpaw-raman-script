import ase.build
import numpy as np
from ase.parallel import world

import gpaw
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.elph.electronphonon import ElectronPhononCoupling

import os
import pickle
import copy
import typing as tp
import pytest

from script import interop
from script import symmetry
from script import test_utils
from script.interop import AseDisplacement

TESTDIR = os.path.dirname(__file__)

ATOMS_PER_CELL = 2
DISPLACEMENT_DIST = 1e-2
BASE_PARAMS = dict(
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

def test_identity():
    ensure_test_data()
    data_subdir = 'sc-111'
    full = do_elph_symmetry(
        data_subdir = data_subdir,
        params_fd = BASE_PARAMS,
        supercell = (1, 1, 1),
        all_displacements = list(AseDisplacement.iter(ATOMS_PER_CELL)),
        symmetry_type = None,
    )

    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            pluses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))
            minuses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))
            expected_dH = (arrayify_dict(pluses[1]) - arrayify_dict(minuses[1])) / (2*DISPLACEMENT_DIST)
            expected_forces = (arrayify_dict(pluses[2]) - arrayify_dict(minuses[2])) / (2*DISPLACEMENT_DIST)

            np.testing.assert_allclose(arrayify_dict(full[1][atom][axis]), expected_dH, err_msg=f'atom {atom} axis {axis}')
            np.testing.assert_allclose(arrayify_dict(full[2][atom][axis]), expected_forces, err_msg=f'atom {atom} axis {axis}')

@pytest.fixture
def data_symmetry():
    ensure_test_data()
    data_subdir = 'sc-111'
    full = do_elph_symmetry(
        data_subdir = data_subdir,
        params_fd = BASE_PARAMS,
        supercell = (1, 1, 1),
        all_displacements = [
            AseDisplacement(atom=atom, axis=0, sign=sign)
            for atom in [0, 1] for sign in [-1, +1]
        ],
        symmetry_type = 'pointgroup',
    )
    return data_subdir, full

def test_symmetry_dH(data_symmetry):
    data_subdir, full = data_symmetry
    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            pluses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))
            minuses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))
            expected = (arrayify_dict(pluses[1]) - arrayify_dict(minuses[1])) / (2*DISPLACEMENT_DIST)
            actual = arrayify_dict(full[1][atom][axis])

            # FIXME is this error really supposed to be this big?!
            rtol = 1e-8 if (atom, axis) == (0, 0) else 12.0
            check_symmetry_result(actual, expected, rtol=rtol, err_msg=f'dH for atom {atom} axis {axis}')

def test_symmetry_forces(data_symmetry):
    data_subdir, full = data_symmetry
    pickle.dump(full, open('blah.pckl', 'wb'), protocol=2)
    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            pluses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))
            minuses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))
            expected = (pluses[2] - minuses[2]) / (2*DISPLACEMENT_DIST)
            actual = full[2][atom][axis]

            rtol = 1e-8 if axis == 0 else 1e-1
            check_symmetry_result(actual, expected, rtol=rtol, err_msg=f'forces for atom {atom} axis {axis}')

@pytest.fixture
def data_supercell_211():
    data_subdir = 'sc-211'
    full = do_elph_symmetry(
        data_subdir = data_subdir,
        params_fd = BASE_PARAMS,
        supercell = (2, 1, 1),
        all_displacements = list(AseDisplacement.iter(ATOMS_PER_CELL)),
        symmetry_type = None,
    )
    return data_subdir, full

def test_supercell_211_dH(data_supercell_211):
    ensure_test_data()
    data_subdir, full = data_supercell_211

    offset = 2
    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            plus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))[1]
            minus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))[1]
            expected = (arrayify_dict(plus) - arrayify_dict(minus)) / (2*DISPLACEMENT_DIST)
            actual = arrayify_dict(full[1][offset+atom][axis])

            rtol = 1e-8 if (atom, axis) == (0, 0) else 0.0
            check_symmetry_result(actual, expected, rtol=rtol, err_msg=f'dH for atom {atom} axis {axis}')

def test_supercell_211_forces(data_supercell_211):
    ensure_test_data()
    data_subdir, full = data_supercell_211

    offset = 2
    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            plus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))[2]
            minus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))[2]
            expected = (plus - minus) / (2*DISPLACEMENT_DIST)
            actual = full[2][offset+atom][axis]
            
            # In atom 0 axis 0, the elements with the biggest relative error have a relative error of 0.250000000.
            # Yes. To that many digits of precision.
            # This seems......... *very highly suspicious?*  FIXME

            rtol = 1e-1
            check_symmetry_result(actual, expected, rtol=rtol, err_msg=f'forces for atom {atom} axis {axis}')

# ==============================================================================

def check_symmetry_result(symmetric_array, normal_array, expected_nnz_range=None, rtol=1e-7, atol=0, zerotol=1e-10, err_msg=None):
    # GPAW's results are often quite asymmetric, and enabling symmetry can cause matrix elements to become zero
    # when they previously appeared to have a non-negligible magnitude.
    magnitude = np.max(np.abs(symmetric_array))
    zero_mask = np.abs(symmetric_array) < zerotol * magnitude

    nonzero_idx = np.where(np.logical_not(zero_mask))
    if expected_nnz_range:
        nnz = len(nonzero_idx[0])
        assert nnz in expected_nnz_range

    try:
        np.testing.assert_allclose(symmetric_array[nonzero_idx], normal_array[nonzero_idx], rtol=rtol, atol=atol, err_msg=err_msg)
    except:
        if err_msg:
            print(f'in {err_msg}')
        print(f'    # Zero: {zero_mask.sum()}')
        print(f' # NonZero: {len(nonzero_idx[0])}')
        print('Biggest errors:')
        symmetric_elems = symmetric_array[nonzero_idx]
        normal_elems = normal_array[nonzero_idx]
        ratios = np.abs(symmetric_elems/normal_elems)
        ratios = np.maximum(ratios, 1/ratios)
        for i in np.argsort(ratios)[::-1][:10]:
            print('   {:25} {:25}   RELERR {:.3e}'.format(symmetric_elems[i], normal_elems[i], ratios[i] - 1))
        raise

def check_dH(data_subdir, full):
    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            pluses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))
            minuses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))
            expected = (arrayify_dict(pluses[1]) - arrayify_dict(minuses[1])) / (2*DISPLACEMENT_DIST)
            actual = arrayify_dict(full[1][atom][axis])

            np.testing.assert_allclose(actual, expected, err_msg=f'dH for atom {atom} axis {axis}')

def check_Vt(data_subdir, full):
    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            pluses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))
            minuses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))
            expected = (pluses[0] - minuses[0]) / (2*DISPLACEMENT_DIST)
            actual = full[0][atom][axis]

            np.testing.assert_allclose(actual, expected, err_msg=f'Vt for atom {atom} axis {axis}')

def check_forces(data_subdir, full):
    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            pluses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))
            minuses = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))
            expected = (pluses[2] - minuses[2]) / (2*DISPLACEMENT_DIST)
            actual = full[2][atom][axis]

            np.testing.assert_allclose(actual, expected, err_msg=f'forces for atom {atom} axis {axis}')

def read_elph_input(data_subdir: str, displacement: AseDisplacement) -> tp.Tuple[np.ndarray, tp.Dict[int, np.ndarray], np.ndarray]:
    Vt_sG, dH_asp = pickle.load(open(f'{MAIN_DATA_DIR}/{data_subdir}/elph.{displacement}.pckl', 'rb'))
    forces = pickle.load(open(f'{MAIN_DATA_DIR}/{data_subdir}/phonons.{displacement}.pckl', 'rb'))
    return Vt_sG, dH_asp, forces

def arrayify_dict(arraydict: tp.Dict[int, np.ndarray]) -> np.ndarray:
    return np.array([arraydict[i] for i in range(len(arraydict))])

def get_wfs_with_sym(params_fd, symmetry_type, supercell_atoms):
    # Make a supercell exactly like ElectronPhononCoupling makes, but with point_group = True
    params_fd_sym = copy.deepcopy(params_fd)
    if symmetry_type:
        params_fd_sym = dict(params_fd)
        if 'symmetry' not in params_fd_sym:
            params_fd_sym['symmetry'] = dict(GPAW.default_parameters['symmetry'])
        params_fd_sym['symmetry']['point_group'] = True

        if symmetry_type == 'pointgroup':
            params_fd_sym['symmetry']['symmorphic'] = True
        elif symmetry_type == 'spacegroup':
            params_fd_sym['symmetry']['symmorphic'] = False  # enable full spacegroup # FIXME: doesn't work for supercells
        else: assert False, symmetry_type

        params_fd_sym['symmetry']['tolerance'] = 1e-6

    calc_fd_sym = GPAW(**params_fd_sym)
    dummy_supercell_atoms = supercell_atoms.copy()
    dummy_supercell_atoms.calc = calc_fd_sym
    calc_fd_sym._set_atoms(dummy_supercell_atoms)  # FIXME private method
    calc_fd_sym.initialize()
    calc_fd_sym.set_positions(dummy_supercell_atoms)
    return calc_fd_sym.wfs

def to_elph_original_types(wfs_with_sym, data):
    """ Turns an unpickled elph file to use the original types from gpaw. """
    from gpaw.arraydict import ArrayDict

    array_0, arr_dic, forces = data
    atom_partition = wfs_with_sym.atom_partition
    arr_dic = ArrayDict(
        partition=atom_partition,
        shapes_a=[arr_dic[a].shape for a in range(atom_partition.natoms)],
        dtype=arr_dic[0].dtype,
        d={a:arr_dic[a] for a in atom_partition.my_indices},
    )
    arr_dic.redistribute(atom_partition.as_serial())
    return array_0, arr_dic, forces


# ==============================================================================

class SymmetryOutput(tp.NamedTuple):
    Vt_avsG: np.ndarray
    dH_avasp: np.ndarray  # of gpaw.arraydict.ArrayDict
    F_avav: np.ndarray


# ==============================================================================
# Generate test input files  ('elph.*.pckl')

MAIN_DATA_DIR = 'tests/data/elph_symmetry'

@test_utils.run_once
def ensure_test_data():

    def make_output(path, supercell):
        if not os.path.exists(path):
            # NOTE: We MUST change directory here because 'phonons.*.pckl' are always created in the
            #       current directory and there's no way to configure this.
            os.makedirs(path)
            with test_utils.pushd(path):
                gen_test_data('.', BASE_PARAMS, supercell=supercell)

    make_output(path = f'{MAIN_DATA_DIR}/sc-111', supercell=(1,1,1))
    make_output(path = f'{MAIN_DATA_DIR}/sc-211', supercell=(2,1,1))
    # make_output(path = f'{MAIN_DATA_DIR}/sc-333', supercell=(3,3,3))

def gen_test_data(datadir: str, params_fd: dict, supercell):
    from gpaw.elph.electronphonon import ElectronPhononCoupling

    atoms = Cluster(ase.build.bulk('C'))
    calc_fd = GPAW(**params_fd)
    elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True)

    calc_gs = GPAW(**params_fd)
    atoms.calc = calc_gs
    atoms.get_potential_energy()
    atoms.calc.write("gs.gpw", mode="all")

    # NOTE: original elph.py did this but I don't understand it.
    # The real space grid of the two calculators should match.
    params_fd['gpts'] = calc_gs.wfs.gd.N_c * list(supercell)
    if 'h' in params_fd:
        del params_fd['h']

    if world.rank == 0:
        os.makedirs(datadir, exist_ok=True)
    elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True, name=f'{datadir}/elph')
    elph.run()
    calc_gs.wfs.gd.comm.barrier()
    elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell)
    elph.set_lcao_calculator(calc_fd)
    elph.calculate_supercell_matrix(dump=1)

# ==============================================================================

def elph_callbacks(wfs_with_symmetry: gpaw.wavefunctions.base.WaveFunctions):
    Vt_part = symmetry.GeneralArrayCallbacks(['na', 'na', 'na', 'na'])  # FIXME second one shouldn't be na
    dH_part = symmetry.GpawLcaoDHCallbacks(wfs_with_symmetry)
    forces_part = symmetry.GeneralArrayCallbacks(['atom', 'cart'], oper_deperms=wfs_with_symmetry.kd.symmetry.a_sa)
    return symmetry.TupleCallbacks(Vt_part, dH_part, forces_part)

# ==============================================================================

def do_elph_symmetry(
    data_subdir: str,
    params_fd: dict,
    supercell,
    all_displacements: tp.Iterable[AseDisplacement],
    symmetry_type: tp.Optional[str],
):
    atoms = Cluster(ase.build.bulk('C'))

    # a supercell exactly like ElectronPhononCoupling makes
    supercell_atoms = atoms * supercell
    quotient_perms = list(interop.ase_repeat_translational_symmetry_perms(len(atoms), supercell))

    wfs_with_sym = get_wfs_with_sym(params_fd=params_fd, supercell_atoms=supercell_atoms, symmetry_type=symmetry_type)
    calc_fd = GPAW(**params_fd)

    elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True)

    # GPAW displaces the center cell for some reason instead of the first cell
    get_displaced_index = lambda prim_atom: elph.offset * len(atoms) + prim_atom

    all_displacements = list(all_displacements)
    disp_atoms = [get_displaced_index(disp.atom) for disp in all_displacements]
    disp_carts = [disp.cart_displacement(DISPLACEMENT_DIST) for disp in all_displacements]
    disp_values = [to_elph_original_types(wfs_with_sym, read_elph_input(data_subdir, disp)) for disp in all_displacements]

    full_Vt = np.empty((len(supercell_atoms), 3) + disp_values[0][0].shape)
    full_dH = np.empty((len(supercell_atoms), 3), dtype=object)
    full_forces = np.empty((len(supercell_atoms), 3) + disp_values[0][2].shape)

    lattice = supercell_atoms.get_cell()[...]
    oper_cart_rots = np.einsum('ki,slk,jl->sij', lattice, wfs_with_sym.kd.symmetry.op_scc, np.linalg.inv(lattice))
    if world.rank == 0:
        full_values = symmetry.expand_derivs_by_symmetry(
            disp_atoms,       # disp -> atom
            disp_carts,       # disp -> 3-vec
            disp_values,      # disp -> T  (displaced value, optionally minus equilibrium value)
            elph_callbacks(wfs_with_sym),        # how to work with T
            oper_cart_rots,   # oper -> 3x3
            oper_perms=wfs_with_sym.kd.symmetry.a_sa,       # oper -> atom' -> atom
            quotient_perms=quotient_perms,
        )
        for a in range(len(full_values)):
            for c in range(3):
                full_Vt[a][c] = full_values[a][c][0]
                full_dH[a][c] = dict(full_values[a][c][1])
                full_forces[a][c] = full_values[a][c][2]
    else:
        # FIXME
        # the symmetry part is meant to be done in serial but we should return back to
        # our original parallel state after it...
        pass

    return full_Vt, full_dH, full_forces


# if __name__ == '__main__':
#     test_supercell_333_forces(data_supercell_333())
#     # ensure_test_data()

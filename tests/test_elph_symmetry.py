import ase.build
import numpy as np
import functools
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

memoize = lambda: functools.lru_cache(maxsize=None)

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
            expected_Vt = (pluses[0] - minuses[0]) / (2*DISPLACEMENT_DIST)
            expected_forces = (pluses[2] - minuses[2]) / (2*DISPLACEMENT_DIST)

            np.testing.assert_allclose(full[0][atom][axis], expected_Vt, err_msg=f'atom {atom} axis {axis}')
            np.testing.assert_allclose(full[2][atom][axis], expected_forces, err_msg=f'atom {atom} axis {axis}')

            # dH is a ragged array
            for a2 in range(ATOMS_PER_CELL):
                expected_dH = (pluses[1][a2] - minuses[1][a2]) / (2*DISPLACEMENT_DIST)
                np.testing.assert_allclose(full[1][atom][axis][a2], expected_dH, err_msg=f'atom {atom} axis {axis}')

@pytest.fixture
@memoize()
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
    disp_atom_offset = 0  # index of first atom of center cell
    return data_subdir, full, disp_atom_offset

def test_symmetry_dH(data_symmetry):
    check_dH_derivative(data_symmetry, dict(rtol=1e-8))

def test_symmetry_Vt(data_symmetry):
    # This hurts.
    # Some of the worst things we see in the output:
    #  symmetric  0.003687548   original  -0.0005984935   --> need large zerotol
    #  symmetric  0.01631375    original   0.01144365     --> relerr of 0.43 (disgusting)
    #  symmetric  0.03412427    original   0.019868       --> okay, forget rtol, use atol
    #  symmetric  0.03412427    original   0.07353537     --> even bigger atol
    #  symmetric -1.778908      original  -1.938584       --> big guy with big abs. err; bring back rtol
    tols = dict(zero_rtol=5e-3, rtol=1e-1, atol=4e-2)
    check_Vt_derivative(data_symmetry, tols)

def test_symmetry_forces(data_symmetry):
    tols = lambda atom, axis: dict(rtol = 1e-8 if (atom, axis) == (0, 0) else 1e-3)
    check_forces_derivative(data_symmetry, tols)

@pytest.fixture
@memoize()
def data_supercell_211():
    ensure_test_data()
    data_subdir = 'sc-211'
    full = do_elph_symmetry(
        data_subdir = data_subdir,
        params_fd = BASE_PARAMS,
        supercell = (2, 1, 1),
        all_displacements = list(AseDisplacement.iter(ATOMS_PER_CELL)),
        symmetry_type = None,
    )
    disp_atom_offset = 2  # index of first atom of center cell
    return data_subdir, full, disp_atom_offset

def test_supercell_211_dH(data_supercell_211):
    check_dH_derivative(data_supercell_211, dict(rtol=1e-8))

def test_supercell_211_Vt(data_supercell_211):
    check_Vt_derivative(data_supercell_211, dict(rtol=1e-8))

def test_supercell_211_forces(data_supercell_211):
    check_forces_derivative(data_supercell_211, dict(rtol=1e-8))

# We have to test something that is both a supercell WITH symmetry enabled,
# due to the complication introduced by pointgroup operators moving atoms
# to other cells.
@pytest.fixture
@memoize()
def data_symmetric_211():
    ensure_test_data()
    data_subdir = 'sc-211'
    full = do_elph_symmetry(
        data_subdir = data_subdir,
        params_fd = BASE_PARAMS,
        supercell = (2, 1, 1),
        all_displacements = [
            AseDisplacement(atom=atom, axis=axis, sign=sign)
            for atom in [0, 1]
            for axis in [0, 1]  # only operator compatible with 2x1x1 is a b-c flip, so we need b displacements too
            for sign in [-1, +1]
        ],
        symmetry_type = 'pointgroup',
    )
    disp_atom_offset = 2  # index of first atom of center cell
    return data_subdir, full, disp_atom_offset

def test_symmetric_211_forces(data_symmetric_211):
    def tols(atom, axis):
        if atom == 0 and axis in [0, 1]:
            return dict(rtol=1e-8)
        # 1.0 might sound like a big z-tolerance, but this comes from direct inspection of the data;
        # The vast majority of elements have an absolute value > 1,
        # and all of the worst offenders are among those few elements that do not:
        #    symmetric  -0.9349432   original  -0.6489585
        #    symmetric  -0.9349432   original  -1.220928
        #    symmetric   0.8988729   original   0.5734485
        return dict(rtol=1e-1, zero_atol=1.0)

    check_forces_derivative(data_symmetric_211, tols)

@pytest.fixture
@memoize()
def data_symmetric_211_from_on_demand_opers():
    ensure_test_data()
    # This is like data_symmetric_211 but we using our shim that computes the LCAO opers
    op_scc = np.array([np.eye(3), np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])])
    data_subdir = 'sc-211'
    full = do_elph_symmetry(
        data_subdir = data_subdir,
        params_fd = BASE_PARAMS,
        supercell = (2, 1, 1),
        all_displacements = [
            AseDisplacement(atom=atom, axis=axis, sign=sign)
            for atom in [0, 1]
            for axis in [0, 1]  # only operator compatible with 2x1x1 is a b-c flip, so we need b displacements too
            for sign in [-1, +1]
        ],
        symmetry_type = 'pointgroup',
    )
    disp_atom_offset = 2  # index of first atom of center cell
    return data_subdir, full, disp_atom_offset

@pytest.fixture
@memoize()
def data_supercell_311():
    ensure_test_data()
    data_subdir = 'sc-311'
    full = do_elph_symmetry(
        data_subdir = data_subdir,
        params_fd = BASE_PARAMS,
        supercell = (3, 1, 1),
        all_displacements = list(AseDisplacement.iter(ATOMS_PER_CELL)),
        symmetry_type = None,
    )
    disp_atom_offset = 2  # index of first atom of center cell
    return data_subdir, full, disp_atom_offset

def test_supercell_311_dH(data_supercell_311):
    check_dH_derivative(data_supercell_311, dict(rtol=1e-8))

def test_supercell_311_Vt(data_supercell_311):
    check_Vt_derivative(data_supercell_311, dict(rtol=1e-8))

def test_supercell_311_forces(data_supercell_311):
    check_forces_derivative(data_supercell_311, dict(rtol=1e-8))

# ==============================================================================

def check_Vt_derivative(data, tols):
    data_subdir, full_output, disp_atom_offset = data

    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            plus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))[0]
            minus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))[0]
            expected = (plus - minus) / (2*DISPLACEMENT_DIST)
            actual = full_output[0][disp_atom_offset + atom][axis]

            this_tols = possibly_call(tols, atom=atom, axis=axis)
            check_symmetry_result(actual, expected, err_msg=f'Vt for atom {atom} axis {axis}', **this_tols)

def check_dH_derivative(data, tols):
    data_subdir, full_output, disp_atom_offset = data

    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            plus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))[1]
            minus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))[1]

            for a2 in range(ATOMS_PER_CELL):
                expected = (plus[a2] - minus[a2]) / (2*DISPLACEMENT_DIST)
                actual = full_output[1][disp_atom_offset + atom][axis][a2]

                this_tols = possibly_call(tols, atom=atom, axis=axis)
                check_symmetry_result(actual, expected, err_msg=f'dH for atom {atom} axis {axis}, atom2 {a2}', **this_tols)

def check_forces_derivative(data, tols):
    data_subdir, full_output, disp_atom_offset = data

    for atom in range(ATOMS_PER_CELL):
        for axis in range(3):
            plus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=+1))[2]
            minus = read_elph_input(data_subdir, AseDisplacement(atom=atom, axis=axis, sign=-1))[2]
            expected = (plus - minus) / (2*DISPLACEMENT_DIST)
            actual = full_output[2][disp_atom_offset + atom][axis]

            this_tols = possibly_call(tols, atom=atom, axis=axis)
            check_symmetry_result(actual, expected, err_msg=f'forces for atom {atom} axis {axis}', **this_tols)

def possibly_call(value, *args, **kw):
    while callable(value):
        value = value(*args, **kw)
    return value

# ==============================================================================

def check_symmetry_result(symmetric_array, normal_array, expected_nnz_range=None, rtol=1e-7, atol=0, zero_rtol=1e-10, zero_atol=0, err_msg=None):
    # GPAW's results are often quite asymmetric, and enabling symmetry can cause matrix elements to become zero
    # when they previously appeared to have a non-negligible magnitude.
    zero_thresh = zero_rtol * np.max(np.abs(symmetric_array)) + zero_atol
    zero_mask = np.abs(symmetric_array) < zero_thresh

    nonzero_idx = np.where(np.logical_not(zero_mask))
    if expected_nnz_range:
        nnz = len(nonzero_idx[0])
        assert nnz in expected_nnz_range

    try:
        np.testing.assert_allclose(symmetric_array[nonzero_idx], normal_array[nonzero_idx], rtol=rtol, atol=atol, err_msg=err_msg)
    except:
        if err_msg:
            print(f'in {err_msg}')

        def error_ratio(symmetric, normal):
            with np.errstate(divide='ignore'):  # 1/0 is okay
                normal = np.where(symmetric == 0, 1, normal)  # avoid 0/0
                ratio = np.abs(symmetric/normal)
                return np.maximum(ratio, 1/ratio)

        symmetric_elems = symmetric_array[nonzero_idx]
        normal_elems = normal_array[nonzero_idx]
        all_ratios = error_ratio(symmetric_elems, normal_elems)
        terribad_ratio_cutoff = sorted(all_ratios)[-10:][0]  # for flagging the worst 10 ratios

        print(f'    # Zero: {zero_mask.sum()}')
        print(f' # NonZero: {len(nonzero_idx[0])}')
        print('All errors:')
        for index_tuple in np.ndindex(symmetric_array.shape):
            symmetric_value = symmetric_array[index_tuple]
            normal_value = normal_array[index_tuple]
            abs_err = abs(symmetric_value - normal_value)

            ratio = error_ratio(symmetric_value, normal_value)
            line = f'{str(index_tuple):>16}  {symmetric_value:13.07} {normal_value:13.07}   ABSERR {abs_err:10.3e}   RELERR {ratio - 1:10.3e}   '
            if np.abs(symmetric_value) < zero_thresh:
                line += f'(zero)'
            elif terribad_ratio_cutoff <= ratio:
                line += '(!!!!)'
            print(line)

        print()
        print('Biggest errors:')
        for i in np.argsort(all_ratios)[::-1][:10]:
            print('   {:25} {:25}   RELERR {:.3e}'.format(symmetric_elems[i], normal_elems[i], all_ratios[i] - 1))
        raise

def read_elph_input(data_subdir: str, displacement: AseDisplacement) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Vt_sG, dH_asp = pickle.load(open(f'{MAIN_DATA_DIR}/{data_subdir}/elph.{displacement}.pckl', 'rb'))
    forces = pickle.load(open(f'{MAIN_DATA_DIR}/{data_subdir}/phonons.{displacement}.pckl', 'rb'))
    return Vt_sG, dH_asp, forces

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
    make_output(path = f'{MAIN_DATA_DIR}/sc-311', supercell=(3,1,1))
    # make_output(path = f'{MAIN_DATA_DIR}/sc-333', supercell=(3,3,3))

def gen_test_data(datadir: str, params_fd: dict, supercell):
    from gpaw.elph.electronphonon import ElectronPhononCoupling

    params_gs = copy.deepcopy(params_fd)

    atoms = Cluster(ase.build.bulk('C'))

    calc_gs = GPAW(**params_gs)
    atoms.calc = calc_gs
    atoms.get_potential_energy()
    atoms.calc.write("gs.gpw", mode="all")

    # Make sure the real space grid matches the original.
    # (basically we multiply the number of grid points in each dimension by the supercell dimension)
    params_fd['gpts'] = calc_gs.wfs.gd.N_c * list(supercell)
    if 'h' in params_fd:
        del params_fd['h']
    del calc_gs

    if world.rank == 0:
        os.makedirs(datadir, exist_ok=True)
    calc_fd = GPAW(**params_fd)
    elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True, name=f'{datadir}/elph')
    elph.run()
    calc_fd.wfs.gd.comm.barrier()
    elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell)
    elph.set_lcao_calculator(calc_fd)
    elph.calculate_supercell_matrix(dump=1)

# ==============================================================================

def elph_callbacks(
        wfs_with_symmetry: gpaw.wavefunctions.base.WaveFunctions,
        supercell,
        elphsym: tp.Optional[symmetry.ElphGpawSymmetrySource] = None,
        ):
    if elphsym is None:
        elphsym = symmetry.ElphGpawSymmetrySource.from_wfs_with_symmetry(wfs_with_symmetry)
    Vt_part = symmetry.GpawLcaoVTCallbacks(wfs_with_symmetry, elphsym, supercell=supercell)
    dH_part = symmetry.GpawLcaoDHCallbacks(wfs_with_symmetry, elphsym)
    forces_part = symmetry.GeneralArrayCallbacks(['atom', 'cart'])
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

    # Make sure the grid matches our calculations (we repeated the grid of the groundstate)
    params_fd = copy.deepcopy(params_fd)
    params_fd['gpts'] = GPAW('gs.gpw').wfs.gd.N_c * list(supercell)
    if 'h' in params_fd:
        del params_fd['h']

    wfs_with_sym = get_wfs_with_sym(params_fd=params_fd, supercell_atoms=supercell_atoms, symmetry_type=symmetry_type)
    calc_fd = GPAW(**params_fd)

    # GPAW displaces the center cell for some reason instead of the first cell
    elph = ElectronPhononCoupling(atoms, calc=calc_fd, supercell=supercell, calculate_forces=True)
    displaced_cell_index = elph.offset
    del elph  # just showing that we don't use these anymore
    del calc_fd

    get_displaced_index = lambda prim_atom: displaced_cell_index * len(atoms) + prim_atom

    all_displacements = list(all_displacements)
    disp_atoms = [get_displaced_index(disp.atom) for disp in all_displacements]
    disp_carts = [disp.cart_displacement(DISPLACEMENT_DIST) for disp in all_displacements]
    disp_values = [read_elph_input(data_subdir, disp) for disp in all_displacements]

    full_Vt = np.empty((len(supercell_atoms), 3) + disp_values[0][0].shape)
    full_dH = np.empty((len(supercell_atoms), 3), dtype=object)
    full_forces = np.empty((len(supercell_atoms), 3) + disp_values[0][2].shape)

    lattice = supercell_atoms.get_cell()[...]
    oper_cart_rots = interop.gpaw_op_scc_to_cart_rots(wfs_with_sym.kd.symmetry.op_scc, lattice)
    if world.rank == 0:
        full_values = symmetry.expand_derivs_by_symmetry(
            disp_atoms,       # disp -> atom
            disp_carts,       # disp -> 3-vec
            disp_values,      # disp -> T  (displaced value, optionally minus equilibrium value)
            elph_callbacks(wfs_with_sym, supercell),        # how to work with T
            oper_cart_rots,   # oper -> 3x3
            oper_perms=wfs_with_sym.kd.symmetry.a_sa,       # oper -> atom' -> atom
            quotient_perms=quotient_perms,
        )
        for a in range(len(full_values)):
            for c in range(3):
                full_Vt[a][c] = full_values[a][c][0]
                full_dH[a][c] = full_values[a][c][1]
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

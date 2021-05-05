from . import utils

import ase
import numpy as np
import typing as tp
from gpaw.wavefunctions.base import WaveFunctions

import sys

def ase_repeat_translational_symmetry_perms(natoms, repeats):
    """ Get the full quotient group of pure translational symmetries of ``atoms * repeats``,
    represented as permutations of atomic data.

    The exact order of the output is specified: Consider that each of these translations corresponds
    to a unique integer point in the primitive lattice ``(ia, ib, ic)``, with ``0 <= ia < na``,
    ``0 <= ib < nb``, ``0 <= ic < nc``.  The output produces these in lexicographic order,
    beginning with ``(0, 0, 0)``, then ``(0, 0, 1)``, up to ``(0, 0, ic-1)``, then ``(0, 1, 0)``, etc.

    Each permutation is a permutation that can be applied using array indexing to data indexed by atom.
    I.e. ``new_data = data[perm]``.
    """
    if isinstance(repeats, int):
        repeats = (repeats, repeats, repeats)

    # It is unclear whether ASE actually specifies the order of atoms in a supercell anywhere,
    # so be ready for the worst.
    if not np.array_equal(repeats, (1, 1, 1)):
        __check_ase_repeat_convention_hasnt_changed()

    n_a, n_b, n_c = repeats
    atom_perm = np.arange(natoms)  # we never rearrange the atoms within a cell
    for a_perm in all_cyclic_perms_of_len(n_a):
        for b_perm in all_cyclic_perms_of_len(n_b):
            for c_perm in all_cyclic_perms_of_len(n_c):
                # in ASE, fastest index is atoms, then repeats[2], then repeats[1], then repeats[0]
                yield utils.permutation_outer_product(a_perm, b_perm, c_perm, atom_perm)

def all_cyclic_perms_of_len(n):
    """ Produce permutations in the cyclic group of rank n.

    The order of the output is specified: The ``i``th permutation pushes elements ``i`` positions
    to the right. """
    # e.g. for n=4 this gives [[0,1,2,3], [3,0,1,2], [2,3,0,1], [1,2,3,0]]
    return np.add.outer(-np.arange(n), np.arange(n)) % n

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

def get_deperm_from_phonopy_sc_to_ase_sc(natoms, repeats):
    """ Get permutation that maps data for a phonopy supercell into data for an ASE supercell.

    I.e. at index ``ase_index``, the output will hold ``phonopy_index``. """
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    assert np.array(repeats).shape == (3,)

    # Generate a supercell with a bunch of simple integer positions
    unitcell_symbols = ['C'] * natoms
    unitcell_positions = np.outer(np.arange(natoms), [1,1,1])  # [0,0,0], [1,1,1], [2,2,2], ...
    unitcell_lattice = np.eye(3) * natoms
    p_atoms = PhonopyAtoms(symbols=unitcell_symbols, positions=unitcell_positions, cell=unitcell_lattice)
    a_atoms = ase.Atoms(symbols=unitcell_symbols, positions=unitcell_positions, cell=unitcell_lattice)

    # get supercells.  (is_symmetry=False in phonopy is because otherwise it may warn about the reduced symmetry of the supercell)
    a_sc_positions = (a_atoms * repeats).get_positions()
    p_sc_positions = phonopy.Phonopy(p_atoms, supercell_matrix=np.diag(repeats), is_symmetry=False).supercell.get_positions()

    # Positions had better be integers
    a_sc_positions_int = np.rint(a_sc_positions)
    p_sc_positions_int = np.rint(p_sc_positions)
    np.testing.assert_allclose(a_sc_positions, a_sc_positions_int, atol=1e-8)
    np.testing.assert_allclose(p_sc_positions, p_sc_positions_int, atol=1e-8)
    # Make sure nothing funny happened to atoms within the cell
    assert (a_sc_positions_int[0] == [0, 0, 0]).all()
    assert (p_sc_positions_int[0] == [0, 0, 0]).all()

    # Find how each can be converted into a simple intermediate order.  (lexically sorted)
    deperm_ase_to_lexical = _lexsort_rows(a_sc_positions_int)
    deperm_phonopy_to_lexical = _lexsort_rows(p_sc_positions_int)
    assert (a_sc_positions_int[deperm_ase_to_lexical] == p_sc_positions_int[deperm_phonopy_to_lexical]).all()

    # Compose
    deperm_lexical_to_ase = np.argsort(deperm_ase_to_lexical)
    return deperm_phonopy_to_lexical[deperm_lexical_to_ase]

class AseDisplacement(tp.NamedTuple):
    """ Helper for representing the strings in filenames generated by ase's Vibrations. """
    atom: int
    axis: int
    sign: int

    @classmethod
    def iter(cls, natoms: int) -> tp.Iterator['AseDisplacement']:
        for atom in range(natoms):
            for axis in range(3):
                for sign in [+1, -1]:
                    yield cls(atom, axis, sign)

    def cart_displacement(self, magnitude: float) -> np.ndarray:
        out = np.zeros(3)
        out[self.axis] = self.sign * magnitude
        return out

    def __str__(self) -> str:
        axis_str = 'xyz'[self.axis]
        sign_str = '-' if self.sign == -1 else '+'
        return f'{self.atom}{axis_str}{sign_str}'

def gpaw_broadcast_array_dict_to_dicts(arraydict):
    """ Take a GPAW arraydict and build a dict that has data at all atoms on all processes. """
    out_a = {}
    for a in range(arraydict.partition.natoms):
        out_a[a] = np.zeros(arraydict.shapes_a[a])
        if a in arraydict:
            out_a[a][:] = arraydict[a]
        arraydict.partition.comm.sum(out_a[a])
    return out_a

def gpaw_op_scc_to_cart_rots(op_scc: np.ndarray, lattice: np.ndarray):
    assert op_scc.shape[1:] == (3,3)
    assert lattice.shape == (3,3)

    # cartesian column operator M = A^T U^T A^-T = (A^-1 U A)^T
    return np.einsum('ik,skl,lj->sji', np.linalg.inv(lattice), op_scc, lattice)

def cart_rots_to_gpaw_op_scc(cart_rots: np.ndarray, lattice: np.ndarray):
    assert cart_rots.shape[1:] == (3,3)
    assert lattice.shape == (3,3)

    #  M = (A^-1 U A)^T  <===>   A M^T A^-1 = U
    return np.einsum('ik,skl,lj->sij', lattice, cart_rots.transpose(0, 2, 1), np.linalg.inv(lattice))

def gpaw_flat_G_oper_permutations(wfs: WaveFunctions):
    """ Get spacegroup operators as permutations of a flattened 'G' axis in GPAW.

    The order of operators in the output matches GPAW's Symmetry class. """
    return _gpaw_flat_G_permutations(wfs.gd.N_c, wfs.kd.symmetry.op_scc, wfs.kd.symmetry.ft_sc, pbc_c=wfs.gd.pbc)

def gpaw_flat_G_quotient_permutations(N_c, repeats, pbc_c):
    """ Get pure translational symmetry operators as permutations of a flattened 'G' axis in GPAW.

    The order of operators in the output matches ``ase_repeat_translational_symmetry_perms``. """
    N_c = np.array(N_c)
    repeats = np.array(repeats)

    ft_sc = lexically_ordered_integer_gridpoints(repeats) / repeats[None, :]
    op_scc = np.tile(np.eye(3), (len(ft_sc), 1, 1))  # all identity matrices
    return _gpaw_flat_G_permutations(N_c, op_scc, ft_sc, pbc_c)

def _gpaw_flat_G_permutations(N_c, op_scc, ft_sc, pbc_c):
    N_c = np.array(N_c)
    assert N_c.shape == (3,)

    # Go to basis where gridpoints have integer locations.
    #
    # Starting with   f' = U^T f + t    (f = frac vec, U = gpaw integer rotation matrix)
    #     and using   n = diag(N_c) f    (n = integer coords of gridpoint)
    #        we get   n' = diag(N_c) U^T diag(N_c)^-1 n + diag(N_c) f
    # and can define  n' = Q^T n + W,
    #                 Q = diag(N_c)^-1 U diag(N_c),      W = diag(N_c) f.
    intop_scc = np.einsum('Bb,abc,cC->aBC', np.diag(1.0 / N_c), op_scc, np.diag(N_c))
    intft_sc = N_c * ft_sc

    # gridpoints initially in lexicographic order, matching the order you get from flattening data
    # from gpaw that has an axis labeled 'G'
    gridpoints = lexically_ordered_integer_gridpoints(N_c, pbc_c)
    gridshape = gpaw_grid_shape(N_c, pbc_c)
    out = []
    for intop_cc, intft_c, op_cc, ft_c in zip(intop_scc, intft_sc, op_scc, ft_sc):
        gridpoints_after_float = gridpoints @ intop_cc + intft_c  # transform the row vectors.  Q (not Q^T) acts on row vectors
        gridpoints_after = np.rint(gridpoints_after_float)
        try:
            utils.assert_allclose_with_counterexamples(
                gridpoints_after_float.reshape(gridshape + (3,)),
                gridpoints_after.reshape(gridshape + (3,)),
                # grid points might not be EXACTLY compatible with symmetry operations due to poor centering of atoms,
                # but we do want to check that at lest they match up 1-1.
                # If they don't, the errors are typically evenly spaced in the interval -0.5 to 0.5, so there should be
                #  at least one with an absolute error of >= 0.3.
                atol=1e-1,
                err_msg="Symmetries are incompatible with grid!",
            )
        except:
            print(' U: ', op_cc.tolist(), file=sys.stderr)
            print('FT: ', ft_c.tolist(), file=sys.stderr)
            raise
        gridpoints_after %= N_c

        # perm that turns gridpoints_after into original coordinates
        inv_coperm = _lexsort_rows(gridpoints_after)
        # the inverse permutation on coordinates is the permutation on data; just what we want
        deperm = inv_coperm
        out.append(deperm)
    return np.array(out)

def lexically_ordered_integer_gridpoints(dim, pbc=True):
    """ Returns lexically ordered tuples of integers where the ``i``th element satisfies ``0 <= tuple[i] < dim[i]``.

    ``pbc`` is gpaw PBC flags. The element at index 0 is omitted from an axis when PBC = False there. """
    dim = tuple(dim)
    grid_ints = np.mgrid[tuple(slice(1 - int(p), n) for (n, p) in np.broadcast(dim, pbc))]  # shape (len(dim), *dim)
    grid_ints = grid_ints.reshape(len(dim), -1)  # shape (len(dim), product(dim))
    return grid_ints.T  # shape  (product(dim), len(dim))

def _lexsort_rows(arr):
    """ Return the indices that lexically sort the rows of a matrix. """
    assert arr.ndim == 2
    return np.lexsort(arr[:, ::-1].T)  # np.lexsort bizarrely does a colexical sort

def gpaw_grid_shape(N_c, pbc_c):
    return tuple(np.array(N_c) - 1 + pbc_c)

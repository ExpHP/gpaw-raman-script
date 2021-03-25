from . import utils

import ase
import numpy as np
import typing as tp

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
                yield utils.permutation_outer_product(a_perm, b_perm, c_perm, atom_perm)


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


class AseDisplacement(tp.NamedTuple):
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

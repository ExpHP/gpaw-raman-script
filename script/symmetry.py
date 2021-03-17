#!/usr/bin/env python3

from collections import defaultdict
import itertools

import numpy as np
import gpaw
import scipy.linalg

from ruamel.yaml import YAML
yaml = YAML(typ='rt')

from abc import ABC, abstractmethod
import typing as tp
T = tp.TypeVar("T")

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
        assert obj.shape == (3, 3)
        return cart_rot @ obj @ cart_rot.T

    def permute_atoms(self, obj, deperm):
        return obj

# FIXME untested
class GeneralArrayCallbacks(SymmetryCallbacks):
    def __init__(self, axis_labels, oper_deperms=None, quotient_deperms=None):
        super().__init__()
        self.shape = None
        self.axis_labels = list(axis_labels)
        self.rotator = TensorRotator(label == 'cart' for label in self.axis_labels)
        self.oper_deperms = oper_deperms
        self.quotient_deperms = quotient_deperms
        if 'atom' in self.axis_labels:
            if oper_deperms is None:
                raise RuntimeError('need oper_deperms if there are atom axes')
            if quotient_deperms is None:
                self.quotient_deperms = np.array([np.arange(len(oper_deperms[0]))])

        unknown_labels = set(axis_labels) - {'na', 'atom', 'cart'}
        if unknown_labels:
            raise RuntimeError(f'bad axis labels: {sorted(unknown_labels)}')

    def initialize(self, obj):
        self.shape = obj.shape
        assert len(self.shape) == len(self.axis_labels)

    def flatten(self, obj):
        return obj.reshape(-1)

    def restore(self, arr):
        return arr.reshape(self.shape)

    def rotate(self, obj, oper, cart_rot):
        assert obj.shape == self.shape, [self.shape, obj.shape]
        obj = self.rotator.rotate(cart_rot, obj)
        if self.oper_deperms is not None:
            obj = self.permute_atoms(obj, self.oper_deperms[oper])
        return obj

    def permute_atoms(self, obj, deperm):
        obj = obj.copy()
        for axis, label in enumerate(self.axis_labels):
            if label == 'atom':
                # perform integer array indexing on the `axis`th axis
                obj = obj[(slice(None),) * axis + (deperm,)]
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
    callbacks: SymmetryCallbacks[T],        # how to work with T
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
        in ``oper_perms/oper_cart_rots``.

    :return:
        Returns a shape ``(natom, 3)`` array of ``T`` where the item at ``(a, k)`` is the derivative of
        the value with respect to cartesian component ``k`` of the displacement of atom ``a``.
        Note that the output is *always* 2-dimensional with ``dtype=object``, even if ``T`` is an array type.
        (so the output may be an array of arrays).  This is done because numpy's overly eager array detection
        could easily lead to data loss if allowed to run unchecked on ``T``.  If you want to reconstruct a
        single array, try ``np.array(output.tolist())``.

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
        The precise definition of the permutations is as follows: Suppose that you have an array of
        atom coordinates ``carts`` (shape ``(nsite,3)``) and an array of data ``data`` (shape ``(nsite,)``).
        Then, for any given spacegroup operation with rotation ``rot``, translation ``trans``, and permutation ``perm`,
        pairing ``carts @ rot.T + trans`` with ``data`` should produce a scenario equivalent to pairing ``carts``
        with ``data[perm]`` (using `integer array indexing <https://numpy.org/doc/stable/reference/arrays.indexing.html#integer-array-indexing>`).
        In this manner, ``perm`` essentially represents the action of the operator
        on metadata when coordinate data is fixed.

        Equivalently, it is the *inverse* of the permutation that operates on the coordinates.
        This is to say that ``(carts @ rot.T + trans)[perm]`` should be equivalent (under lattice translation)
        to the original ``carts``.
    """

    # FIXME too many local variables visible in this function

    assert len(disp_carts) == len(disp_atoms) == len(disp_values)
    assert len(oper_cart_rots) == len(oper_perms)

    natoms = len(oper_perms[0])

    disp_values = list(disp_values)
    callbacks.initialize(disp_values[0])

    # For each representative atom that gets displaced, gather all of its displacements.
    representative_disps = defaultdict(list)
    for (disp, representative) in enumerate(disp_atoms):   # FIXME: scope of these variables is uncomfortably large
        representative_disps[representative].append(disp)

    if quotient_perms is None:
        # Just the identity.
        quotient_perms = np.array([np.arange(len(oper_perms[0]))])

    sym_info = PrecomputedSymmetryIndexInfo(representative_disps.keys(), oper_perms, quotient_perms)

    def apply_combined_oper(value: T, combined: 'CombinedOperator'):
        oper, quotient = combined
        value = callbacks.rotate(value, oper, cart_rot=oper_cart_rots[oper])
        value = callbacks.permute_atoms(value, quotient_perms[quotient])
        return value

    # Compute derivatives with respect to displaced (representative) atoms
    def compute_representative_row(representative):
        # Expand the available data using the site-symmetry operators to ensure
        # we have enough independent equations for pseudoinversion.
        eq_cart_disps = []  # equation -> 3-vec
        eq_rhses = []  # equation -> flattened T

        # Generate equations by pairing each site symmetry operator with each displacement of this atom
        for combined_op in sym_info.site_symmetry_for_rep(representative):
            cart_rot = oper_cart_rots[combined_op.oper]
            for disp in representative_disps[representative]:
                transformed = apply_combined_oper(disp_values[disp], combined_op)

                eq_cart_disps.append(cart_rot @ disp_carts[disp])
                eq_rhses.append(callbacks.flatten(transformed))

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
    site_derivatives = {rep: compute_representative_row(rep) for rep in representative_disps.keys()}

    # Fill out more rows (i.e. derivatives w.r.t. other atoms) by applying spacegroup symmetry
    for atom in range(natoms):
        if atom in site_derivatives:
            continue

        # We'll just apply the first operator that sends us here
        rep = sym_info.data[atom].rep
        combined_op = sym_info.data[atom].operators[0]

        # Apply the rotation to the inner dimensions of the gradient (i.e. rotate each T)
        t_derivs_by_axis = [apply_combined_oper(deriv, combined_op) for deriv in site_derivatives[rep]]

        # Apply the rotation to the outer axis of the gradient (i.e. for each scalar element of T, rotate its gradient)
        array_derivs_by_axis = [callbacks.flatten(t) for t in t_derivs_by_axis]
        array_derivs_by_axis = oper_cart_rots[combined_op.oper] @ array_derivs_by_axis
        t_derivs_by_axis = [callbacks.restore(arr) for arr in array_derivs_by_axis]

        site_derivatives[atom] = t_derivs_by_axis

    # site_derivatives should now be dense
    assert set(range(natoms)) == set(site_derivatives)

    # Convert to array, in a manner that prevents numpy from detecting the dimensions of T.
    final_out = np.empty((natoms, 3), dtype=object)
    final_out[...] = [site_derivatives[i] for i in range(natoms)]
    return final_out

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
    """
    rep: AtomIndex
    operators: tp.List[CombinedOperator]

class PrecomputedSymmetryIndexInfo:
    """ A class that records how to reach each atom from a predetermined set of symmetry representatives.

    Attributes:
        from_reps   dict. For each atom, a ``FromRepInfo`` describing how to reach that atom.
    """
    data: tp.Dict[AtomIndex, FromRepInfo]

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
                    # Find the site that rep gets sent to
                    site = oper_inv_deperms[oper][rep]
                    site = quotient_inv_deperms[quotient][site]
                    if site not in from_reps:
                        from_reps[site] = FromRepInfo(rep, [])
                    from_reps[site].operators.append(CombinedOperator(oper, quotient))

        if redundant_reps:
            message = ', '.join('{} (~= {})'.format(a, from_reps[a].rep) for a in redundant_reps)
            raise RuntimeError('redundant atoms in representative list:  {}'.format(message))

        natoms = len(oper_deperms[0])
        missing_indices = set(range(natoms)) - set(from_reps)
        if missing_indices:
            raise RuntimeError(f'no representative atoms were symmetrically equivalent to these indices: {sorted(missing_indices)}!')

        self.data = from_reps

    def site_symmetry_for_rep(self, rep: AtomIndex) -> tp.Iterable[CombinedOperator]:
        """ Get operators in the site symmetry of a representative atom. """
        true_rep = self.data[rep].rep
        assert true_rep == rep, "not a representative: {} (image of {})".format(rep, true_rep)

        return self.data[rep].operators

# ==============================================================================

class TensorRotator:
    """ Helper for automating the production of an einsum call that applies a single matrix to many axes of an array.
    
    E.g. could perform something similar to ``np.einsum('Aa,Bb,Dd,abcd->ABcD', rot, rot, rot, array)`` if we wanted
    to rotate axes 0, 1, and 3 of an array. """
    def __init__(self, axis_rotate_flags: tp.Iterator[bool]):
        unused_subscripts = itertools.count(start=0)
        self.array_subscripts = []
        self.rotmat_subscripts = []
        self.out_subscripts = []

        for flag in axis_rotate_flags:
            if flag:
                sum_subscript = next(unused_subscripts)
                out_subscript = next(unused_subscripts)
                self.rotmat_subscripts.append((out_subscript, sum_subscript))
                self.array_subscripts.append(sum_subscript)
                self.out_subscripts.append(out_subscript)
            else:
                subscript = next(unused_subscripts)
                self.array_subscripts.append(subscript)
                self.out_subscripts.append(subscript)

    def rotate(self, rot, array):
        einsum_args = []
        for subscripts in self.rotmat_subscripts:
            einsum_args.append(rot)
            einsum_args.append(subscripts)
        einsum_args += [array, self.array_subscripts, self.out_subscripts]
        return np.einsum(*einsum_args)

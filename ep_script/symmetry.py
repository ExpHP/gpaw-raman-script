#!/usr/bin/env python3

from . import interop

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
U = tp.TypeVar("U")

AtomIndex = int
QuotientIndex = int
OperIndex = int

class SymmetryCallbacks(ABC, tp.Generic[T]):
    """ Class that factors out operations needed by ``expand_derivs_by_symmetry`` to make it
    general over all different sorts of data.

    Instances must not be reused for more than one call to ``expand_derivs_by_symmetry``.
    This restriction enables implementations to record data about the shape of their input if necessary.
    """
    def __init__(self):
        self.__already_init = False
        self.__flat_len = None

    def initialize(self, obj: T):
        """ Record any data needed about the shape of T, if necessary.

        This will always be the first method called, and will be called exactly once on an
        arbitrarily-chosen item from ``disp_values``. """
        if self.__already_init:
            raise RuntimeError('SymmetryCallbacks instances must not be reused')
        self.__already_init = True

    def flatten(self, obj: T) -> np.ndarray:
        arr = self.flatten_impl(obj)
        if self.__flat_len is None:
            self.__flat_len, = arr.shape
        else:
            np.testing.assert_array_equal(arr.shape, (self.__flat_len,))
        return arr

    def flat_len(self):
        if self.__flat_len is None:
            raise ValueError('must call .flatten() at least once first')
        return self.__flat_len

    @abstractmethod
    def flatten_impl(self, obj: T) -> np.ndarray:
        """ Method that should be defined on subclasses to implement ``flatten``. """
        raise NotImplementedError

    @abstractmethod
    def restore(self, arr: np.ndarray) -> T:
        """ Reconstruct an object from an ndarray of ndim 1. """
        raise NotImplementedError

    @abstractmethod
    def apply_oper(self, obj: T, oper: OperIndex, cart_rot, atom_deperm) -> T:
        """ Apply a spacegroup operation.

        The ``cart_rot`` matrix (representing the rotational part of the operation as a 3x3 matrix
        to apply on the left side of a 3-vector) and the ``atom_deperm`` array (representing the operator
        as a permutation of data indexed by atom, where ``transformed_data = data[atom_deperm]``)
        are supplied for convenience.  If an implementation doesn't need them (or it is not enough,
        e.g. it needs to know the fractional translation part), then that's fine; it should store the
        necessary data and look at ``oper`` instead.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_quotient(self, obj: T, quotient: QuotientIndex, atom_deperm: np.ndarray) -> T:
        """ Apply a pure translational symmetry.

        The ``atom_deperm`` array (representing the translation as a permutation of data indexed by
        atom, i.e. ``translated_data = data[atom_deperm]``) is supplied for convenience.
        If an implementation doesn't need it (or it is not enough, e.g. it needs to know how
        the points in a finite grid permute), then that's fine; it should store the necessary data
        and look at ``quotient`` instead.
        """
        raise NotImplementedError

class ArrayCallbacks(SymmetryCallbacks[np.ndarray]):
    """ Superclass that handles the flattening and restoration steps for callbacks on an array. """
    shape: tp.Tuple[int, ...]

    def initialize(self, obj):
        super().initialize(obj)
        self.shape = obj.shape

    def flatten_impl(self, obj):
        np.testing.assert_array_equal(obj.shape, self.shape)
        return obj.reshape(-1)

    def restore(self, arr):
        return arr.reshape(self.shape)

class Tensor2Callbacks(ArrayCallbacks):
    def apply_oper(self, obj, oper, cart_rot, atom_deperm):
        assert obj.shape == (3, 3)
        return cart_rot @ obj @ cart_rot.T

    def apply_quotient(self, obj, quotient, atom_deperm):
        return obj

Label = str
OperType = str  # "oper" | "quotient"
TransformType = str  # "matrix" | "perm"

class GeneralArrayCallbacks(ArrayCallbacks):
    """ Implements SymmetryCallbacks for any ndarray where the action of a symmetry operation factorizes cleanly
    into an independent group action on each individual axis of the array. """

    # the arrays in here are the data arrays mentioned in the documentation of the `label_specs` argument.
    oper_specs: tp.Dict[Label, tp.Tuple[TransformType, np.ndarray]]
    quotient_specs: tp.Dict[Label, tp.Tuple[TransformType, np.ndarray]]

    PLACEHOLDER_CART_ROT = object()
    PLACEHOLDER_ATOM_DEPERM = object()
    ALL_PLACEHOLDERS = [PLACEHOLDER_CART_ROT, PLACEHOLDER_ATOM_DEPERM]

    def __init__(
            self,
            axis_labels: tp.Sequence[Label],
            *label_specs: tp.Tuple[tp.Tuple[Label, OperType], TransformType, np.ndarray],
            ):
        """
        Construct a callback for a general n-dimensional array by specifying the action on each individual axis
        (either as a matrix product or a permutation).

        :param axis_labels: sequence of ``str`` for each array axis.  ``"cart"`` means to rotate this axis
        like a Cartesian vector. ``"atom"`` means it indexes atoms in the supercell. ``"na"`` means to leave this
        axis alone.  The meaning of other labels can be controlled through ``label_specs``.
        Any label is allowed to appear any number of times.

        :param label_specs: Supplies the data necessary for transforming each label.
        First in each spec is a key, composed of two strings.  The first string is an axis label;
        let ``<axislen>`` refer to the length of this axis.
        The second string is one of ``'oper'`` (for spacegroup) or ``'quotient'`` (for supercell translations);
        let ``<nsym>`` refer to the size of the corresponding symmetry group.

        After the key is a string indicating the type of operation to perform, followed by the data for that operation.
        * If the operation is ``'perm'``, then the data is an integer array of shape ``(<nsym>, <axislen>)``.
          Each row is the representation of an operator as a permutation of data indexed by that axis,
          in such a form that ``new_data == data[perm]``.
        * If the operation is ``'matrix'``, then the data is an array of shape ``(<nsym>, <axislen>, <axislen>)``
          that could be applied on the left hand side of a column vector indexed by this axis.

        Any label used in ``axis_labels`` must have at least one spec,
        except for the labels ``"cart"``, ``"atom"`` and ``"na"``.

        Example:  A rank 2 cartesian tensor (such as polarizability) could be transformed using
        ``GeneralArrayCallbacks(['cart', 'cart'])``, which is equivalent to computing rotations as ``R @ tensor @ R.T``.

        Example:  GPAW has an array ``Vt_sG`` which is indexed by ``(spin, na, nb, nc)``, where ``(na, nb, nc)`` are
        the dimensions of a real-space grid.  Unfortunately, the action of a rotation on this doesn't cleanly factor
        out into an independent action on each axis, but if you flatten out the last three axes to get ``(spin, gridpoint)``,
        then you will find that all symmetry operations permute these gridpoints.
        Therefore, after computing the necessary perms, one could use
        ``GeneralArrayCallbacks(['na', 'grid'], (('grid', 'oper'), 'perm', oper_grid_perms), (('grid', 'quotient'), 'perm', quotient_grid_perms))``.
        (See also ``WrappedCallbacks`` for a way to automatically perform the flattening of the gridpoint axes)
        """
        super().__init__()

        self.axis_labels = list(axis_labels)

        seen_keys = set()
        self.oper_specs = {}
        self.quotient_specs = {}
        for key, transform_kind, transform_data in label_specs:
            transform_data = np.array(transform_data)

            label, oper_type = key
            key = (label, oper_type)  # ensure hashable
            if key in seen_keys:
                raise ValueError(f'Key {repr(key)} used multiple times!')
            seen_keys.add(key)

            if label in ['na', 'cart', 'rot']:
                raise ValueError(f"Cannot override label {label}")

            if transform_kind == 'matrix':
                assert transform_data.ndim == 3
                assert transform_data.shape[1] == transform_data.shape[2]
            elif transform_kind == 'perm':
                assert transform_data.ndim == 2
            else:
                raise ValueError(f'Invalid transform type {repr(transform_kind)}')

            if oper_type == 'oper':
                self.oper_specs[label] = (transform_kind, transform_data)
            elif oper_type == 'quotient':
                self.quotient_specs[label] = (transform_kind, transform_data)
            else:
                raise ValueError(f'Invalid symmetry type {repr(oper_type)}')

        self.oper_specs['cart'] = ('matrix', type(self).PLACEHOLDER_CART_ROT)
        self.oper_specs['atom'] = ('perm', type(self).PLACEHOLDER_ATOM_DEPERM)
        self.quotient_specs['atom'] = ('perm', type(self).PLACEHOLDER_ATOM_DEPERM)

        unknown_labels = set(axis_labels) - {'na'} - set(self.oper_specs) - set(self.quotient_specs)
        if unknown_labels:
            raise RuntimeError(f'Some labels are missing symmetry data: {sorted(unknown_labels)}')

    def initialize(self, obj):
        super().initialize(obj)
        np.testing.assert_equal(len(self.shape), len(self.axis_labels), err_msg=str(self.shape))

        # Sanity test: any label that appears multiple times should have the same length for each of its axes
        label_dims = {'cart': 3}
        for dim, label in zip(obj.shape, self.axis_labels):
            if label in label_dims:
                assert label_dims[label] == dim, f'size mismatch for {label}: ({self.axis_labels}, {obj.shape})'
            else:
                label_dims[label] = dim

        # Sanity test: That length should match the data saved for its transformations
        for sym_kind, specs in [('oper', self.oper_specs), ('quotient', self.quotient_specs)]:
            for label, (_, data) in specs.items():
                if label not in label_dims:
                    continue  # could happen for 'atom' if it's not used in the array
                label_dim = label_dims[label]
                if not any(data is x for x in self.ALL_PLACEHOLDERS):  # can't use 'in' operator because ndarray
                    assert data.shape[1] == label_dim, f'dim for {repr(label)} is {label_dim}, but {sym_kind} has data of shape {data.shape}'

    def apply_oper(self, obj, oper, cart_rot, atom_deperm):
        return self._apply_sym(obj, self.oper_specs, oper, cart_rot=cart_rot, atom_deperm=atom_deperm)

    def apply_quotient(self, obj, quotient, atom_deperm):
        return self._apply_sym(obj, self.quotient_specs, quotient, atom_deperm=atom_deperm)

    def _apply_sym(self, obj, label_specs, sym_index, cart_rot=None, atom_deperm=None):
        assert obj.shape == self.shape, [self.shape, obj.shape]
        assert np.array(sym_index).shape == ()

        obj = obj.copy()

        for label, (transform_kind, sym_transform_data) in label_specs.items():
            if sym_transform_data is type(self).PLACEHOLDER_ATOM_DEPERM:
                transform_data = atom_deperm
            elif sym_transform_data is type(self).PLACEHOLDER_CART_ROT:
                transform_data = cart_rot
            else:
                transform_data = sym_transform_data[sym_index]

            if transform_kind == 'matrix':
                rotation_matrix = transform_data
                # perform a tensor contraction over every axis with this label
                rotator = TensorRotator(l == label for l in self.axis_labels)
                obj = rotator.rotate(rotation_matrix, obj)

            elif transform_kind == 'perm':
                permutation = transform_data
                for iter_axis, iter_label in enumerate(self.axis_labels):
                    if iter_label == label:
                        # perform integer array indexing on the `axis`th axis
                        obj = obj[(slice(None),) * iter_axis + (permutation,)]

            else:
                assert False, transform_kind  # unreachable

        return obj

class AtomDictCallbacks(SymmetryCallbacks[tp.Dict[int, T]], tp.Generic[T]):
    """ Callbacks for a dict over atoms. (e.g. for use as a ragged array) """
    def __init__(self):
        super().__init__()
        self.shapes = None

    def initialize(self, obj):
        super().initialize(obj)

        self.shapes = [obj[a].shape for a in range(len(obj))]

    def flatten_impl(self, obj):
        return np.concatenate([obj[a].reshape(-1) for a in range(len(obj))])

    def restore(self, arr):
        sizes = [np.product(shape) for shape in self.shapes]
        splits = np.cumsum(sizes)[:-1]
        arrs_a = np.split(arr, splits)

        return {a: arrs_a[a].reshape(self.shapes[a]) for a in range(len(arrs_a))}

    def apply_oper(self, obj, sym, cart_rot, atom_deperm):
        return self._permute(obj, atom_deperm)

    def apply_quotient(self, obj, quotient, atom_deperm):
        return self._permute(obj, atom_deperm)

    def _permute(self, obj, atom_deperm):
        out_a = {}
        for anew, aold in enumerate(atom_deperm):
            out_a[anew] = obj[aold].copy()
        return out_a

class GpawLcaoDHCallbacks(AtomDictCallbacks[np.ndarray]):
    """ Callbacks for ``calc.hamiltonian.dH_asp`` (as a ragged dict of arrays) in GPAW LCAO mode. """
    def __init__(self, wfs, symmetry: 'ElphGpawSymmetrySource'):
        super().__init__()
        self.nspins = wfs.nspins
        self.symmetry = symmetry

    def apply_oper(self, obj, sym, cart_rot, atom_deperm):
        from gpaw.utilities import pack, unpack2

        a_a = self.symmetry.a_sa[sym]
        assert (a_a == atom_deperm).all(), "mismatched oper order or something?"

        # permute the 'a' axis (atoms in the dict)
        obj = super().apply_oper(obj, sym, cart_rot, atom_deperm)

        # and now the 'p' axis
        dH_asp = obj
        for a in range(len(dH_asp)):
            R_ii = self.symmetry.R_asii[a][sym]
            for s in range(self.nspins):
                dH_p = dH_asp[a][s]
                dH_ii = unpack2(dH_p)
                tmp_ii = R_ii @ dH_ii @ R_ii.T
                tmp_p = pack(tmp_ii)
                dH_asp[a][s][...] = tmp_p

        return dH_asp

def GpawLcaoVTCallbacks(wfs, symmetry: 'ElphGpawSymmetrySource', supercell):
    assert len(supercell) == 3
    return GpawLcaoVTCallbacks__from_parts(
        nspins=wfs.nspins,
        N_c=wfs.gd.N_c,
        op_scc=symmetry.op_scc,
        ft_sc=symmetry.ft_sc,
        supercell=supercell,
        pbc_c=wfs.gd.pbc_c,
    )

def GpawLcaoVTCallbacks__from_parts(nspins, N_c, op_scc, ft_sc, supercell, pbc_c):
    from . import interop

    N_c = tuple(N_c)
    supercell = tuple(supercell)
    grid_oper_deperms = interop._gpaw_flat_G_permutations(N_c, op_scc, ft_sc, pbc_c)
    grid_quotient_deperms = interop.gpaw_flat_G_quotient_permutations(N_c=N_c, repeats=supercell, pbc_c=pbc_c)

    return WrappedCallbacks[np.ndarray, np.ndarray](
        # To apply these permutations we have to flatten the three grid axes.
        convert_into=lambda arr: arr.reshape((nspins, -1)),
        convert_from=lambda arr: arr.reshape((nspins,) + tuple(np.array(N_c) - 1 + pbc_c)),
        wrapped=GeneralArrayCallbacks(
            ['na', 'flatgrid'],
            (('flatgrid', 'oper'), 'perm', grid_oper_deperms),
            (('flatgrid', 'quotient'), 'perm', grid_quotient_deperms),
        ),
    )

class TupleCallbacks(SymmetryCallbacks[tp.Tuple]):
    # Callbacks for each item.
    parts: tp.List[SymmetryCallbacks]

    def __init__(self, *parts: SymmetryCallbacks):
        super().__init__()
        self.parts = list(parts)

    def initialize(self, obj):
        super().initialize(obj)

        assert isinstance(obj, tuple)
        assert len(obj) == len(self.parts)
        for (x, callbacks) in zip(obj, self.parts):
            callbacks.initialize(x)

    def flatten_impl(self, obj):
        return np.concatenate([callbacks.flatten(x) for (x, callbacks) in zip(obj, self.parts)])

    def restore(self, arr):
        splits = np.cumsum([callbacks.flat_len() for callbacks in self.parts])[:-1]
        arrs = np.split(arr, splits)
        return tuple(callbacks.restore(arr) for (arr, callbacks) in zip(arrs, self.parts))

    def apply_oper(self, obj, oper, cart_rot, atom_deperm):
        return tuple(callbacks.apply_oper(x, oper, cart_rot, atom_deperm) for (x, callbacks) in zip(obj, self.parts))

    def apply_quotient(self, obj, quotient, atom_deperm):
        return tuple(callbacks.apply_quotient(x, quotient, atom_deperm) for (x, callbacks) in zip(obj, self.parts))

class WrappedCallbacks(tp.Generic[T, U], SymmetryCallbacks[T]):
    def __init__(self, convert_into: tp.Callable[[T], U], convert_from: tp.Callable[[U], T], wrapped: SymmetryCallbacks[U]):
        super().__init__()
        self.convert_into = convert_into
        self.convert_from = convert_from
        self.wrapped = wrapped

    def initialize(self, obj):
        super().initialize(obj)
        return self.wrapped.initialize(self.convert_into(obj))

    def flatten_impl(self, obj):
        return self.wrapped.flatten_impl(self.convert_into(obj))

    def restore(self, arr):
        return self.convert_from(self.wrapped.restore(arr))

    def apply_oper(self, obj, oper, cart_rot, atom_deperm):
        converted = self.convert_into(obj)
        transformed = self.wrapped.apply_oper(converted, oper, cart_rot, atom_deperm)
        return self.convert_from(transformed)

    def apply_quotient(self, obj, quotient, atom_deperm):
        converted = self.convert_into(obj)
        transformed = self.wrapped.apply_quotient(converted, quotient, atom_deperm)
        return self.convert_from(transformed)


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
        value = callbacks.apply_oper(value, oper, cart_rot=oper_cart_rots[oper], atom_deperm=oper_perms[oper])
        value = callbacks.apply_quotient(value, quotient, atom_deperm=quotient_perms[quotient])
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
            raise RuntimeError(f'no representative atoms were symmetrically equivalent to these indices: {sorted(missing_indices)}!  (num symmetry opers: {len(oper_deperms)})')

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
    def __init__(self, axis_rotate_flags: tp.Iterable[bool]):
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

# ==============================================================================

class ElphGpawSymmetrySource:
    """ Helper for code that needs symmetry data in the format provided by GPAW, but for operators
    not computed by GPAW. """

    op_scc: np.ndarray  # replacement for wfs.kd.symmetry.op_scc
    ft_sc: np.ndarray  # replacement for wfs.kd.symmetry.ft_sc
    a_sa: np.ndarray  # replacement for wfs.kd.symmetry.a_sa
    R_asii: tp.Dict[AtomIndex, np.ndarray]  # replacement for wfs.setups[a].R_sii

    def __init__(self, op_scc, ft_sc, a_sa, R_asii):
        self.op_scc = op_scc
        self.ft_sc = ft_sc
        self.a_sa = a_sa
        self.R_asii = R_asii

    @classmethod
    def from_wfs_with_symmetry(cls, wfs):
        """ Create from GPAW's own Symmetry. """
        return cls(
            op_scc=wfs.kd.symmetry.op_scc,
            ft_sc=wfs.kd.symmetry.ft_sc,
            a_sa=wfs.kd.symmetry.a_sa,
            R_asii={a: setup.R_sii for (a, setup) in enumerate(wfs.setups)},
        )

    @classmethod
    def from_setups_and_ops(cls, setups: 'gpaw.setup.Setups', lattice, oper_cart_rots, oper_cart_trans, oper_deperms):
        """ Create from a custom set of operators. """
        op_scc = interop.cart_rots_to_gpaw_op_scc(oper_cart_rots, lattice)
        ft_sc = oper_cart_trans @ np.linalg.inv(lattice)
        a_sa = oper_deperms

        return cls(
            op_scc=op_scc,
            ft_sc=ft_sc,
            a_sa=a_sa,
            R_asii=cls.compute_R_asii(setups, lattice=lattice, op_scc=op_scc),
        )

    # (copied from Setups.set_symmetry, but changed to return a value instead of set an attribute)
    @staticmethod
    def compute_R_asii(setups: 'gpaw.setup.Setups', lattice, op_scc):
        from gpaw.rotation import rotation

        R_slmm = []
        for op_cc in op_scc:
            op_vv = np.dot(np.linalg.inv(lattice), np.dot(op_cc, lattice))
            R_slmm.append([rotation(l, op_vv) for l in range(4)])

        # Compute LCAO operators for each setup.
        # 'X' is an index label for unique atom type (Z, type, basis).  (the key for setups.setups, and value from id_a)
        R_Xsii = {
            X: ElphGpawSymmetrySource.compute_R_sii(setup, R_slmm)
            for (X, setup) in setups.setups.items()
        }

        # Switch to per-atom.
        R_asii = {a: R_Xsii[X] for (a, X) in enumerate(setups.id_a)}
        return R_asii

    # (copied from BaseSetup.calculate_rotations, but changed to return a value instead of set an attribute)
    @staticmethod
    def compute_R_sii(setup, R_slmm):
        nsym = len(R_slmm)
        R_sii = np.zeros((nsym, setup.ni, setup.ni))
        i1 = 0
        for l in setup.l_j:
            i2 = i1 + 2 * l + 1
            for s, R_lmm in enumerate(R_slmm):
                R_sii[s, i1:i2, i1:i2] = R_lmm[l]
            i1 = i2
        return R_sii

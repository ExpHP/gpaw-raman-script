import typing as tp
import itertools
import contextlib
import os

T = tp.TypeVar('T')
G = tp.TypeVar('G')
H = tp.TypeVar('H')

GeneratorIndex = int
NodeIndex = int

class SemigroupTree(tp.Generic[G]):
    generators: tp.List[G]
    generator_indices: tp.List[NodeIndex]
    members: tp.List[G]
    decomps: tp.List[tp.Union[
        GeneratorIndex,  # leaf node
        tp.Tuple[NodeIndex, NodeIndex],  # binary node
    ]]

    def __init__(
            self,
            generators: tp.Iterable[G],
            func: tp.Callable[[G, G], G],
            make_hashable: tp.Callable[[G], tp.Any] = lambda x: x,
    ):
        self.generators = list(generators)
        self.generator_indices = []
        self.members = []
        self.decomps = []

        if not self.generators:
            return  # empty semigroup

        all_seen: tp.Set = set()
        for gen_gen_index, gen in enumerate(self.generators):
            # Ignore redundant generators.
            if not checked_add_to_set(all_seen, make_hashable(gen)):
                continue

            # Add leaf node for generator.
            self.generator_indices.append(len(self.members))
            self.members.append(gen)
            self.decomps.append(gen_gen_index)

        for gen_node_index, gen in zip(self.generator_indices, self.generators):
            # Try applying the generator on the left of every known member.
            #
            # This is written as a 'while' instead of a 'for' because we also want to iterate over
            # any items that were added to the list *during* the loop.
            rhs_node_index = 0
            while rhs_node_index < len(self.members):
                product = func(gen, self.members[rhs_node_index])

                if checked_add_to_set(all_seen, make_hashable(product)):
                    self.members.append(product)
                    self.decomps.append((gen_node_index, rhs_node_index))
                rhs_node_index += 1

    def compute_homomorphism(
        self,
        get_generator: tp.Callable[[GeneratorIndex, G], H],
        compose: tp.Callable[[H, H], H]
    ) -> tp.List[H]:
        out: tp.List[H] = []
        for decomp in self.decomps:
            if isinstance(decomp, tuple):
                a_index, b_index = decomp
                out.append(compose(out[a_index], out[b_index]))
            else:
                gen_index = decomp
                out.append(get_generator(gen_index, self.generators[gen_index]))

        return out


def cyclic_group(
        generator: G,
        func: tp.Callable[[G, G], G],
        make_hashable: tp.Callable[[G], tp.Any] = lambda x: x,
        ):
    """ Generates the cyclic group of a generator G of finite order, in the sequence G^0, G^1, G^2, ... """
    current = generator
    sequence = []
    seen: tp.Set = set()
    while checked_add_to_set(seen, make_hashable(current)):
        sequence.append(current)
        current = func(current, generator)
    # identity is at the end; move it to the front
    return [sequence.pop()] + sequence


def checked_add_to_set(set: tp.Set[T], item: T) -> bool:
    """ Adds an item to a set and returns a boolean indicating whether the item was newly added. """
    was_new = item not in set
    set.add(item)
    return was_new


_NOT_YET_RUN = object()
def run_once(function):
    value = _NOT_YET_RUN
    def wrapped(*args, **kw):
        nonlocal value
        # NOTE: This could have a race condition if called from multiple threads in parallel, but since gpaw
        #       uses MPI, pretty much all possible hope of that has already been thrown out the window.
        if value is _NOT_YET_RUN:
            value = function(*args, **kw)
        return value
    return wrapped

@contextlib.contextmanager
def pushd(dest):
    """ Context manager for changing directory.  The original directory is restored on exiting the ``with`` block.

    This exists because there are some files that GPAW *always* puts in the current directory.

    This will wreak absolute havoc if used in a multithreaded application.
    """
    old_cwd = os.getcwd()
    try:
        os.chdir(dest)
        yield None
    finally:
        os.chdir(old_cwd)

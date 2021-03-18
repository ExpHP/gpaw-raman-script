import typing as tp
import itertools

T = tp.TypeVar('T')
G = tp.TypeVar('G')
H = tp.TypeVar('H')

def close_group(
    generators: tp.Iterable[T],
    func: tp.Callable[[T, T], T],
    make_hashable: tp.Callable[[T], tp.Any] = lambda x: x,
):
    generators = list(generators)

    yield generators[0]
    all_seen = set(make_hashable(generators[0]))

    for g in generators:
        for old in all_seen:
            new = func(g, old)
            new_hashable = make_hashable(new)
            while new_hashable not in all_seen:
                yield new_hashable
                all_seen.add(new_hashable)

                new = func(g, new)
                new_hashable = make_hashable(new)

def set_update_new_items(
        s: tp.MutableSet[T],
        items: tp.Iterable[T],
        make_hashable: tp.Callable[[T], tp.Any] = lambda x: x,
) -> tp.Iterator[T]:
    """ Adds items to a set (possibly mapping them through a function that makes them hashable)
    and yields the items that are newly added. """
    for item in items:
        hashable = make_hashable(item)
        if hashable not in s:
            yield item
            s.add(hashable)

def func_bfs_groups(
    roots: tp.Iterable[T],
    edge_func: tp.Callable[[T], tp.Iterable[T]],
    make_hashable: tp.Callable[[T], tp.Any] = lambda x: x,
) -> tp.Iterator[tp.List[T]]:
    """ In a directed, unweighted graph implicitly constructed by a given "outedges" function, perform a
    breadth-first traversal beginning from ``roots`` and return groups of nodes by their proximity to any
    of the nodes in ``roots``.
    
    E.g. first returns the roots themselves, then returns any new nodes that can be reached in one edge,
    then any nodes that can be reached in 2 edges... """
    all_seen = set()

    new_items = list(set_update_new_items(all_seen, roots, make_hashable=make_hashable))
    while new_items:
        yield new_items
        next_reachable_items = itertools.chain.from_iterable(edge_func(item) for item in new_items)
        new_items = list(set_update_new_items(all_seen, next_reachable_items, make_hashable=make_hashable))

def close_semigroup(
    generators: tp.Iterable[T],
    func: tp.Callable[[T], tp.Iterable[T]],
    make_hashable: tp.Callable[[T], tp.Any] = lambda x: x,
) -> tp.Iterator[T]:
    """ Compute all items in the closure of a semigroup in a breadth-first manner from a set of generators. """
    generators = list(generators)
    yield from itertools.chain.from_iterable(func_bfs_groups(
        roots=generators,
        edge_func=(lambda node: func(g, node) for g in generators),
        make_hashable=make_hashable,
    ))

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
        """ """
        self.generators = list(generators)
        self.generator_indices = []
        self.members = []
        self.decomps = []

        if not self.generators:
            return  # empty semigroup

        all_seen = set()
        for gen_gen_index, gen in enumerate(self.generators):
            # Ignore redundant generators.
            gen_hashable = make_hashable(gen)
            if gen_hashable in all_seen:
                continue

            # Add leaf node for generator.
            all_seen.add(gen_hashable)
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
                product_hashable = make_hashable(product)

                if product_hashable not in all_seen:
                    all_seen.add(product_hashable)
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

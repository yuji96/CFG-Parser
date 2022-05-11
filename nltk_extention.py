from pathlib import Path as p
from typing import Iterator, List, Tuple

from nltk.corpus import BracketParseCorpusReader
from nltk.tree import Tree


class Node:

    def __init__(self, tree: Tree, parent=None):
        self.pos = tree.label()
        self.word = tree[0] if isinstance(tree[0], str) else None

        self.parent = parent
        self.children: List[Node] = list(self.__init_children(tree))

        self.index = None
        if self.is_root:
            self.index = self.__init_index()

    def __init_children(self, tree):
        for child in tree:
            if isinstance(child, Tree):
                yield Node(child, self)

    def __init_index(self, start=0):
        if self.is_terminal:
            self.index = (start, start + 1)
            return self.index

        indexes = []
        for child in self.children:
            start, end = child.__init_index(start)
            indexes.append((start, end))
            start = end
        (left, _), *_, (_, right) = indexes
        self.index = (left, right)
        return self.index

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_terminal(self):
        return self.word is not None

    @property
    def length(self):
        left, right = self.index
        return right - left

    def get_leaves(self) -> List["Node"]:
        if self.is_terminal:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def __str__(self):
        if self.is_terminal:
            return f"({self.pos} [{self.word}])"
        else:
            return f"({self.pos} {[child.pos for child in self.children]})"

    __repr__ = __str__


def read_parsed_corpus(root,
                       pattern: str = "**/wsj_*.mrg") -> Iterator[Tuple[str, Tree]]:
    reader = BracketParseCorpusReader(root, r"wsj_.*\.mrg", tagset="wsj")
    for path in p(root).glob(pattern):
        sents = reader.parsed_sents(f"{path.parent.name}/{path.name}")
        for tree in sents:
            yield f"{path.parent.name}/{path.name}", tree


if __name__ == "__main__":
    _iter = read_parsed_corpus("treebank_3/parsed/mrg/wsj", "00/wsj_0001.mrg")
    _, tree = next(_iter)
    tree = Node(tree)
    leaves = tree.get_leaves()
    print(leaves)
    print([leaf.index for leaf in leaves])
    print(tree.index)

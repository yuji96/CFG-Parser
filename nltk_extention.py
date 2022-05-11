from pathlib import Path as p
from typing import Iterator, List, Tuple

from nltk.corpus import BracketParseCorpusReader
from nltk.tree import Tree


class Node:

    def __init__(self, tree: Tree, parent=None):
        self.pos = tree.label()
        self.parent = parent
        self.word = tree[0] if isinstance(tree[0], str) else None
        self.length = self.init_length()
        self.children: List[Node] = list(self.__init_children(tree))

    def __init_children(self, tree):
        for child in tree:
            if isinstance(child, Tree):
                yield Node(child, self)

    def init_length(self):
        if self.is_terminal:
            return 1

        length = 0
        for child in self.children:
            if isinstance(child, Node):
                length += child.length
            else:
                raise AssertionError
        return length

    def init_index(self):
        pass

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_terminal(self):
        return self.word is not None

    def get_leaves(self) -> List["Node"]:
        if self.is_terminal:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def __str__(self):
        if self.is_terminal:
            return f"({self.word})"
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
    words = tree.get_leaves()
    print(words)
    print(tree.length)

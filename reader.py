from pathlib import Path as p
from typing import Iterator, Tuple

from nltk.corpus import (BracketParseCorpusReader, ChunkedCorpusReader,
                         RegexpTokenizer, tagged_treebank_para_block_reader)
from nltk.tree import Tree
from tqdm import tqdm


def progress(iter_, verbose, **kwargs):
    if not verbose:
        return iter_
    return tqdm(iter_, **kwargs)


def read_chunked_corpus(root, dir_numbers,
                        verbose=False) -> Iterator[Tuple[str, Tree]]:
    dir_numbers = [str(num).zfill(2) for num in dir_numbers]
    reader = ChunkedCorpusReader(
        root,
        r"wsj_.*\.pos",
        sent_tokenizer=RegexpTokenizer(r"(?<=/\.)\s*(?![^\[]*\])", gaps=True),
        para_block_reader=tagged_treebank_para_block_reader,
        tagset="wsj",
    )
    for childdir in progress(sorted(p(root).iterdir()), verbose):
        if childdir.name not in dir_numbers:
            continue

        for path in progress(sorted(childdir.glob("wsj_*.pos")), verbose,
                             leave=False):
            words = reader.tagged_words(f"{childdir.name}/{path.name}")
            yield path, words


def read_parsed_corpus(root, dir_numbers,
                       verbose=False) -> Iterator[Tuple[str, Tree]]:
    """構文解析済みのファイル（`*.mrg`）を `Tree` 型として返すイテレータ。

    Parameters
    ----------
    root : str
        ツリーバンクのパス。
    dir_numbers : Sequence[int]
        サブディレクトリ名の配列。ex) `range(0, 11)` とすると `root/00/wsj_*.mrg` から
        `root/10/wsj_*.mrg` が読み込まれる。
    verbose : bool, default=False
        `True` にした場合は、プログレスバーが表示される。このとき、`print` を使うと
        表示がバグるので `tqdm.write` を使う。

    Yields
    ------
    path : Path
        読み込んだパス。
    tree : Tree
        解析済みの木構造。

    """
    dir_numbers = [str(num).zfill(2) for num in dir_numbers]
    reader = BracketParseCorpusReader(root, r"wsj_.*\.mrg", tagset="wsj")
    for childdir in progress(sorted(p(root).iterdir()), verbose):
        if childdir.name not in dir_numbers:
            continue

        for path in progress(sorted(childdir.glob("wsj_*.mrg")), verbose,
                             leave=False):
            for tree in reader.parsed_sents(f"{childdir.name}/{path.name}"):
                yield path, tree


if __name__ == "__main__":
    for path, words in read_chunked_corpus("treebank_3/tagged/pos/wsj",
                                           range(0, 5 + 1), verbose=True):
        print(path, words, sep="\n")
        break

    for path, tree in read_parsed_corpus("treebank_3/parsed/mrg/wsj",
                                         range(0, 5 + 1), verbose=True):
        print(path, tree, sep="\n")
        break

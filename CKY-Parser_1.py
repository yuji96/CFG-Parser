from collections import defaultdict
from itertools import product

from lxml import etree

example_sentence = "Time flies like an arrow"

grammar_text = """S->NP VP:1.0
VP->V NP:0.5
VP->V PP:0.5
NP->D N:0.4
NP->N N:0.3
NP->N:0.3
PP->P NP:1.0
N->Time:0.7
N->flies:0.2
N->arrow:0.1
V->flies:0.6
V->like:0.4
P->like:1.0
D->an:1.0"""


def CKY(leaves, lexical_rule, syntax_rule):
    n = len(leaves)
    cell = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i, leaf in enumerate(leaves):
        cell[i][i + 1] = lexical_rule[leaf]

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l
            for k in range(i + 1, j):
                for prob_l, x_l in cell[i][k]:
                    for prob_r, x_r in cell[k][j]:
                        cell[i][j] += syntax_rule.get((x_l, x_r), [])
    return cell


def parse(self, text):
    words = text.split(" ")
    self.length = len(words)
    self._init_cky_array(self.length)

    for i, word in enumerate(words):
        for l, p in self.grammar_dict[word]:
            self.cky_array[i][i].append((l, word, p))

    # 対角成分の1つ右,2つ右,...と処理を回すループ
    for d in range(1, self.length):

        # 斜め下に進んでいくループ
        # i,jでどのセルを処理対象とするか決める
        for i in range(self.length - d):
            j = i + d

            # セルの中身を埋めるループ
            for k in range(i, j):

                # 右辺の可能な組み合わせを列挙してる
                for a, b in product(range(len(self.cky_array[i][k])),
                                    range(len(self.cky_array[k + 1][j]))):

                    # 辞書のキーを作る
                    s = "{0} {1}".format(self.cky_array[i][k][a][0],
                                         self.cky_array[k + 1][j][b][0])

                    # キーに合致する文法をぜんぶ出す
                    for l, p in self.grammar_dict[s]:

                        # セルに中身を入れる
                        self.cky_array[i][j].append(
                            (l, ((i, k, a), (k + 1, j, b)), p))

    # parseの最後
    return self._gen_xml_etree_list()


if __name__ == "__main__":
    from nltk.tree import Tree
    from pathlib import Path as p
    tree: Tree = Tree.fromstring(p("example/tree.mrg").read_text())

    from pprint import pprint
    from counter import rule_as_dict

    lexical_dict = {
        'Time': [(0.5, 'N')],
        'an': [(1.0, 'D')],
        'arrow': [(0.5, 'N')],
        'flies': [(1.0, 'V')],
        'like': [(1.0, 'P')]
    }

    syntax_dict = {
        ('D', 'N'): [(0.5, 'NP')],
        ('N', ): [(0.5, 'NP')],
        ('NP', 'VP'): [(1.0, 'S')],
        ('P', 'NP'): [(1.0, 'PP')],
        ('V', 'PP'): [(1.0, 'VP')]
    }

    pprint(syntax_dict)

    print(*CKY(tree.leaves(), lexical_dict, syntax_dict), sep="\n")

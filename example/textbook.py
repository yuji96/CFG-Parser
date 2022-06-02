import sys
from pathlib import Path

src = Path(__file__).joinpath("../../src").resolve()
sys.path.append(str(src))

from cky import CKY

lexical_dict = {
    'Time': [(0.7, 'N')],
    'an': [(1.0, 'D')],
    'arrow': [(0.1, 'N')],
    'flies': [(0.2, 'N'), (0.6, 'V')],
    'like': [(0.4, 'V'), (1.0, 'P')]
}
syntax_dict = {
    ('NP', 'VP'): [(1.0, 'S')],
    ('V', 'NP'): [(0.5, 'VP')],
    ('V', 'PP'): [(0.5, 'VP')],
    ('D', 'N'): [(0.4, 'NP')],
    ('N', 'N'): [(0.3, 'NP')],
    ('P', 'NP'): [(1.0, 'PP')],
}
unary_dict = {
    'S': [(1, 'TOP')],
    'N': [(0.3, 'NP')],
}

leaves = ["Time", "flies", "like", "an", "arrow"]

tree = CKY(leaves, lexical_dict, syntax_dict, unary_dict, beam=5)
tree.pretty_print()

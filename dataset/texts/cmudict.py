""" from https://github.com/keithito/tacotron """

import re

valid_symbols = [
    'A',
    'A0',
    'A0l',
    'Al',
    'B',
    'B0',
    'B0l',
    'Bl',
    'D',
    'D0',
    'D0l',
    'DZ',
    'DZ0',
    'DZ0l',
    'DZH',
    'DZH0',
    'DZH0l',
    'DZHl',
    'DZl',
    'Dl',
    'E',
    'E0',
    'E0l',
    'El',
    'F',
    'F0',
    'F0l',
    'Fl',
    'G',
    'G0',
    'G0l',
    'GH',
    'GH0',
    'GH0l',
    'GHl',
    'Gl',
    'I',
    'I0',
    'I0l',
    'Il',
    'J0',
    'J0l',
    'K',
    'K0',
    'K0l',
    'KH',
    'KH0',
    'KH0l',
    'KHl',
    'Kl',
    'L',
    'L0',
    'L0l',
    'Ll',
    'M',
    'M0',
    'M0l',
    'Ml',
    'N',
    'N0',
    'N0l',
    'Nl',
    'O',
    'O0',
    'O0l',
    'Ol',
    'P',
    'P0',
    'P0l',
    'Pl',
    'R',
    'R0',
    'R0l',
    'Rl',
    'S',
    'S0',
    'S0l',
    'SH',
    'SH0',
    'SH0l',
    'SHl',
    'Sl',
    'T',
    'T0',
    'T0l',
    'TS',
    'TS0',
    'TS0l',
    'TSH',
    'TSH0',
    'TSH0l',
    'TSHl',
    'TSl',
    'Tl',
    'U',
    'U0',
    'U0l',
    'Ul',
    'V',
    'V0',
    'V0l',
    'Vl',
    'Y',
    'Y0',
    'Y0l',
    'Yl',
    'Z',
    'Z0',
    'Z0l',
    'ZH',
    'ZH0',
    'ZH0l',
    'ZHl',
    'Zl',
    'pau',
    'sil',
    'spn', 
    'sp'
]

_valid_symbol_set = set(valid_symbols)


class CMUDict:
    """Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict"""

    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding="latin-1") as f:
                entries = _parse_cmudict(f)
        else:
            entries = _parse_cmudict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        """Returns list of ARPAbet pronunciations of the given word."""
        return self._entries.get(word.upper())


_alt_re = re.compile(r"\([0-9]+\)")


def _parse_cmudict(file):
    cmudict = {}
    for line in file:
        if len(line) and (line[0] >= "A" and line[0] <= "Z" or line[0] == "'"):
            parts = line.split("  ")
            word = re.sub(_alt_re, "", parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in cmudict:
                    cmudict[word].append(pronunciation)
                else:
                    cmudict[word] = [pronunciation]
    return cmudict


def _get_pronunciation(s):
    parts = s.strip().split(" ")
    for part in parts:
        if part not in _valid_symbol_set:
            return None
    return " ".join(parts)

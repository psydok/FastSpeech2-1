""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from dataset.texts import cmudict

_pad = "_"
_eos = "~"
_bos = "^"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_silences = ["spn", "sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _silences + [_eos]

# For Phonemes

PAD = "#"
EOS = "~"
PHONEME_CODES = "A A0 A0l Al B B0 B0l Bl D D0 D0l DZ DZ0 DZ0l DZH DZH0 DZH0l DZHl DZl Dl E E0 E0l El F F0 F0l Fl G G0 G0l GH GH0 GH0l GHl Gl I I0 I0l Il J0 J0l K K0 K0l KH KH0 KH0l KHl Kl L L0 L0l Ll M M0 M0l Ml N N0 N0l Nl O O0 O0l Ol P P0 P0l Pl R R0 R0l Rl S S0 S0l SH SH0 SH0l SHl Sl T T0 T0l TS TS0 TS0l TSH TSH0 TSH0l TSHl TSl Tl U U0 U0l Ul V V0 V0l Vl Y Y0 Y0l Yl Z Z0 Z0l ZH ZH0 ZH0l ZHl Zl pau".split()
_PHONEME_SEP = " "

phonemes_symbols = [PAD, EOS] + PHONEME_CODES  # PAD should be first to have zero id

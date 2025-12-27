/// A pre-tokenized word with its position in the original text.
///
/// Pre-tokens are the result of splitting text on whitespace and punctuation
/// before WordPiece tokenization.
class PreToken {
  /// The text content of this pre-token.
  final String text;

  /// The starting character position in the original text.
  final int start;

  /// The ending character position in the original text (exclusive).
  final int end;

  /// Creates a pre-token with the given text and positions.
  const PreToken({required this.text, required this.start, required this.end});

  @override
  String toString() => 'PreToken("$text", [$start:$end])';
}

/// BERT-style pre-tokenizer for text normalization and splitting.
///
/// Performs the following operations:
/// - Normalizes Unicode characters and removes control characters
/// - Optionally converts text to lowercase
/// - Optionally strips accents (e.g., `é` -> `e`)
/// - Adds spaces around Chinese characters for character-level tokenization
/// - Splits on whitespace and punctuation
///
/// Example:
/// ```dart
/// final preTokenizer = BertPreTokenizer();
/// final tokens = preTokenizer.preTokenize('Hello, World!');
/// // Returns: [PreToken("hello", 0, 5), PreToken(",", 5, 6), ...]
/// ```
class BertPreTokenizer {
  /// Whether to convert text to lowercase.
  final bool lowercase;

  /// Whether to strip accents from characters.
  final bool stripAccents;

  /// Whether to add spaces around Chinese characters.
  final bool handleChineseChars;

  /// Creates a BERT pre-tokenizer with the specified options.
  const BertPreTokenizer({
    this.lowercase = true,
    this.stripAccents = true,
    this.handleChineseChars = true,
  });

  /// Pre-tokenizes text into a list of [PreToken]s.
  ///
  /// Normalizes the text and splits it on whitespace and punctuation,
  /// preserving character positions for later mapping.
  List<PreToken> preTokenize(String text) {
    final normalized = _normalizeSinglePass(text);
    return _splitOnWhitespaceAndPunctuation(normalized);
  }

  String _normalizeSinglePass(String text) {
    final buffer = StringBuffer();

    for (final rune in text.runes) {
      if (rune == 0 || rune == 0xFFFD) {
        continue;
      }

      if (_isControl(rune)) {
        continue;
      }

      if (_isWhitespace(rune)) {
        buffer.write(' ');
        continue;
      }

      if (handleChineseChars && _isChineseChar(rune)) {
        buffer.write(' ');
        buffer.writeCharCode(rune);
        buffer.write(' ');
        continue;
      }

      var processedRune = rune;

      if (stripAccents) {
        final stripped = _accentMapInt[rune];
        if (stripped != null) {
          if (lowercase) {
            buffer.write(stripped.toLowerCase());
          } else {
            buffer.write(stripped);
          }
          continue;
        }
        if (_isCombiningMark(rune)) {
          continue;
        }
      }

      if (lowercase) {
        processedRune = _toLowerCodePoint(rune);
      }

      buffer.writeCharCode(processedRune);
    }

    return buffer.toString();
  }

  int _toLowerCodePoint(int rune) {
    if (rune >= 0x41 && rune <= 0x5A) {
      return rune + 0x20;
    }
    if (rune < 0x80) {
      return rune;
    }
    final lower = String.fromCharCode(rune).toLowerCase();
    return lower.codeUnitAt(0);
  }

  List<PreToken> _splitOnWhitespaceAndPunctuation(String normalized) {
    final tokens = <PreToken>[];
    var tokenStart = -1;
    var normIndex = 0;

    while (normIndex < normalized.length) {
      final char = normalized[normIndex];
      final rune = normalized.codeUnitAt(normIndex);

      if (_isWhitespace(rune)) {
        if (tokenStart >= 0) {
          tokens.add(
            PreToken(
              text: normalized.substring(tokenStart, normIndex),
              start: tokenStart,
              end: normIndex,
            ),
          );
          tokenStart = -1;
        }
      } else if (_isPunctuation(rune)) {
        if (tokenStart >= 0) {
          tokens.add(
            PreToken(
              text: normalized.substring(tokenStart, normIndex),
              start: tokenStart,
              end: normIndex,
            ),
          );
          tokenStart = -1;
        }

        tokens.add(PreToken(text: char, start: normIndex, end: normIndex + 1));
      } else {
        if (tokenStart < 0) {
          tokenStart = normIndex;
        }
      }

      normIndex++;
    }

    if (tokenStart >= 0) {
      tokens.add(
        PreToken(
          text: normalized.substring(tokenStart),
          start: tokenStart,
          end: normalized.length,
        ),
      );
    }

    return tokens;
  }

  bool _isWhitespace(int rune) {
    if (rune == 0x20 || rune == 0x09 || rune == 0x0A || rune == 0x0D) {
      return true;
    }

    if (rune == 0x00A0 ||
        rune == 0x1680 ||
        (rune >= 0x2000 && rune <= 0x200A) ||
        rune == 0x202F ||
        rune == 0x205F ||
        rune == 0x3000) {
      return true;
    }

    return false;
  }

  bool _isControl(int rune) {
    if (rune == 0x09 || rune == 0x0A || rune == 0x0D) {
      return false;
    }

    if ((rune >= 0x00 && rune <= 0x1F) || (rune >= 0x7F && rune <= 0x9F)) {
      return true;
    }

    return false;
  }

  bool _isPunctuation(int rune) {
    if ((rune >= 0x21 && rune <= 0x2F) ||
        (rune >= 0x3A && rune <= 0x40) ||
        (rune >= 0x5B && rune <= 0x60) ||
        (rune >= 0x7B && rune <= 0x7E)) {
      return true;
    }

    if (rune >= 0x2000 && rune <= 0x206F) {
      return true;
    }

    if (rune >= 0x3000 && rune <= 0x303F) {
      return true;
    }

    return false;
  }

  bool _isChineseChar(int rune) {
    if ((rune >= 0x4E00 && rune <= 0x9FFF) ||
        (rune >= 0x3400 && rune <= 0x4DBF) ||
        (rune >= 0x20000 && rune <= 0x2A6DF) ||
        (rune >= 0x2A700 && rune <= 0x2B73F) ||
        (rune >= 0x2B740 && rune <= 0x2B81F) ||
        (rune >= 0x2B820 && rune <= 0x2CEAF) ||
        (rune >= 0x2CEB0 && rune <= 0x2EBEF) ||
        (rune >= 0xF900 && rune <= 0xFAFF) ||
        (rune >= 0x2F800 && rune <= 0x2FA1F)) {
      return true;
    }

    return false;
  }

  bool _isCombiningMark(int rune) {
    if (rune >= 0x0300 && rune <= 0x036F) {
      return true;
    }

    if (rune >= 0x1AB0 && rune <= 0x1AFF) {
      return true;
    }

    if (rune >= 0x1DC0 && rune <= 0x1DFF) {
      return true;
    }

    if (rune >= 0x20D0 && rune <= 0x20FF) {
      return true;
    }

    if (rune >= 0xFE20 && rune <= 0xFE2F) {
      return true;
    }

    return false;
  }
}

const _accentMapInt = <int, String>{
  0xC0: 'A',
  0xC1: 'A',
  0xC2: 'A',
  0xC3: 'A',
  0xC4: 'A',
  0xC5: 'A',
  0xC7: 'C',
  0xC8: 'E',
  0xC9: 'E',
  0xCA: 'E',
  0xCB: 'E',
  0xCC: 'I',
  0xCD: 'I',
  0xCE: 'I',
  0xCF: 'I',
  0xD0: 'D',
  0xD1: 'N',
  0xD2: 'O',
  0xD3: 'O',
  0xD4: 'O',
  0xD5: 'O',
  0xD6: 'O',
  0xD9: 'U',
  0xDA: 'U',
  0xDB: 'U',
  0xDC: 'U',
  0xDD: 'Y',
  0xDE: 'TH',
  0xE0: 'a',
  0xE1: 'a',
  0xE2: 'a',
  0xE3: 'a',
  0xE4: 'a',
  0xE5: 'a',
  0xE7: 'c',
  0xE8: 'e',
  0xE9: 'e',
  0xEA: 'e',
  0xEB: 'e',
  0xEC: 'i',
  0xED: 'i',
  0xEE: 'i',
  0xEF: 'i',
  0xF0: 'd',
  0xF1: 'n',
  0xF2: 'o',
  0xF3: 'o',
  0xF4: 'o',
  0xF5: 'o',
  0xF6: 'o',
  0xF9: 'u',
  0xFA: 'u',
  0xFB: 'u',
  0xFC: 'u',
  0xFD: 'y',
  0xFF: 'y',
  0xFE: 'th',
  0x100: 'A',
  0x101: 'a',
  0x102: 'A',
  0x103: 'a',
  0x104: 'A',
  0x105: 'a',
  0x106: 'C',
  0x107: 'c',
  0x108: 'C',
  0x109: 'c',
  0x10A: 'C',
  0x10B: 'c',
  0x10C: 'C',
  0x10D: 'c',
  0x10E: 'D',
  0x10F: 'd',
  0x110: 'D',
  0x111: 'd',
  0x112: 'E',
  0x113: 'e',
  0x114: 'E',
  0x115: 'e',
  0x116: 'E',
  0x117: 'e',
  0x118: 'E',
  0x119: 'e',
  0x11A: 'E',
  0x11B: 'e',
  0x11C: 'G',
  0x11D: 'g',
  0x11E: 'G',
  0x11F: 'g',
  0x120: 'G',
  0x121: 'g',
  0x122: 'G',
  0x123: 'g',
  0x124: 'H',
  0x125: 'h',
  0x126: 'H',
  0x127: 'h',
  0x128: 'I',
  0x129: 'i',
  0x12A: 'I',
  0x12B: 'i',
  0x12C: 'I',
  0x12D: 'i',
  0x12E: 'I',
  0x12F: 'i',
  0x130: 'I',
  0x131: 'i',
  0x134: 'J',
  0x135: 'j',
  0x136: 'K',
  0x137: 'k',
  0x138: 'k',
  0x139: 'L',
  0x13A: 'l',
  0x13B: 'L',
  0x13C: 'l',
  0x13D: 'L',
  0x13E: 'l',
  0x13F: 'L',
  0x140: 'l',
  0x141: 'L',
  0x142: 'l',
  0x143: 'N',
  0x144: 'n',
  0x145: 'N',
  0x146: 'n',
  0x147: 'N',
  0x148: 'n',
  0x149: 'n',
  0x14A: 'N',
  0x14B: 'n',
  0x14C: 'O',
  0x14D: 'o',
  0x14E: 'O',
  0x14F: 'o',
  0x150: 'O',
  0x151: 'o',
  0x152: 'OE',
  0x153: 'oe',
  0x154: 'R',
  0x155: 'r',
  0x156: 'R',
  0x157: 'r',
  0x158: 'R',
  0x159: 'r',
  0x15A: 'S',
  0x15B: 's',
  0x15C: 'S',
  0x15D: 's',
  0x15E: 'S',
  0x15F: 's',
  0x160: 'S',
  0x161: 's',
  0x162: 'T',
  0x163: 't',
  0x164: 'T',
  0x165: 't',
  0x166: 'T',
  0x167: 't',
  0x168: 'U',
  0x169: 'u',
  0x16A: 'U',
  0x16B: 'u',
  0x16C: 'U',
  0x16D: 'u',
  0x16E: 'U',
  0x16F: 'u',
  0x170: 'U',
  0x171: 'u',
  0x172: 'U',
  0x173: 'u',
  0x174: 'W',
  0x175: 'w',
  0x176: 'Y',
  0x177: 'y',
  0x178: 'Y',
  0x179: 'Z',
  0x17A: 'z',
  0x17B: 'Z',
  0x17C: 'z',
  0x17D: 'Z',
  0x17E: 'z',
  0x1A0: 'O',
  0x1A1: 'o',
  0x1AF: 'U',
  0x1B0: 'u',
  0x1CD: 'A',
  0x1CE: 'a',
  0x1CF: 'I',
  0x1D0: 'i',
  0x1D1: 'O',
  0x1D2: 'o',
  0x1D3: 'U',
  0x1D4: 'u',
  0x1D5: 'U',
  0x1D6: 'u',
  0x1D7: 'U',
  0x1D8: 'u',
  0x1D9: 'U',
  0x1DA: 'u',
  0x1DB: 'U',
  0x1DC: 'u',
  0x1EA0: 'A',
  0x1EA1: 'a',
  0x1EA2: 'A',
  0x1EA3: 'a',
  0x1EA4: 'A',
  0x1EA5: 'a',
  0x1EA6: 'A',
  0x1EA7: 'a',
  0x1EA8: 'A',
  0x1EA9: 'a',
  0x1EAA: 'A',
  0x1EAB: 'a',
  0x1EAC: 'A',
  0x1EAD: 'a',
  0x1EAE: 'A',
  0x1EAF: 'a',
  0x1EB0: 'A',
  0x1EB1: 'a',
  0x1EB2: 'A',
  0x1EB3: 'a',
  0x1EB4: 'A',
  0x1EB5: 'a',
  0x1EB6: 'A',
  0x1EB7: 'a',
  0x1EB8: 'E',
  0x1EB9: 'e',
  0x1EBA: 'E',
  0x1EBB: 'e',
  0x1EBC: 'E',
  0x1EBD: 'e',
  0x1EBE: 'E',
  0x1EBF: 'e',
  0x1EC0: 'E',
  0x1EC1: 'e',
  0x1EC2: 'E',
  0x1EC3: 'e',
  0x1EC4: 'E',
  0x1EC5: 'e',
  0x1EC6: 'E',
  0x1EC7: 'e',
  0x1EC8: 'I',
  0x1EC9: 'i',
  0x1ECA: 'I',
  0x1ECB: 'i',
  0x1ECC: 'O',
  0x1ECD: 'o',
  0x1ECE: 'O',
  0x1ECF: 'o',
  0x1ED0: 'O',
  0x1ED1: 'o',
  0x1ED2: 'O',
  0x1ED3: 'o',
  0x1ED4: 'O',
  0x1ED5: 'o',
  0x1ED6: 'O',
  0x1ED7: 'o',
  0x1ED8: 'O',
  0x1ED9: 'o',
  0x1EDA: 'O',
  0x1EDB: 'o',
  0x1EDC: 'O',
  0x1EDD: 'o',
  0x1EDE: 'O',
  0x1EDF: 'o',
  0x1EE0: 'O',
  0x1EE1: 'o',
  0x1EE2: 'O',
  0x1EE3: 'o',
  0x1EE4: 'U',
  0x1EE5: 'u',
  0x1EE6: 'U',
  0x1EE7: 'u',
  0x1EE8: 'U',
  0x1EE9: 'u',
  0x1EEA: 'U',
  0x1EEB: 'u',
  0x1EEC: 'U',
  0x1EED: 'u',
  0x1EEE: 'U',
  0x1EEF: 'u',
  0x1EF0: 'U',
  0x1EF1: 'u',
  0x1EF2: 'Y',
  0x1EF3: 'y',
  0x1EF4: 'Y',
  0x1EF5: 'y',
  0x1EF6: 'Y',
  0x1EF7: 'y',
  0x1EF8: 'Y',
  0x1EF9: 'y',
};

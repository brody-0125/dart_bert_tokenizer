import 'unicode_data.dart';

/// A normalized word with its original Unicode code-point positions.
class PreToken {
  /// Normalized token text.
  final String text;

  /// Inclusive position in the original input, in Unicode code points.
  final int start;

  /// Exclusive position in the original input, in Unicode code points.
  final int end;
  final List<(int, int)>? _alignment;

  /// Creates a pre-token whose characters map directly to the original input.
  const PreToken({required this.text, required this.start, required this.end})
    : _alignment = null;

  PreToken._(this.text, List<(int, int)> alignment)
    : _alignment = alignment,
      start = alignment.first.$1,
      end = alignment.last.$2;

  /// Maps a normalized UTF-16 span back to the original code-point span.
  (int, int) originalSpan(int from, int to) => _alignment == null
      ? (start + from, start + to)
      : (_alignment[from].$1, _alignment[to - 1].$2);

  /// An aligned slice whose indices are UTF-16 positions in [text].
  PreToken slice(int from, int to) {
    if (from == 0 && to == text.length) return this;
    final value = text.substring(from, to);
    if (_alignment != null) {
      return PreToken._(value, _alignment.sublist(from, to));
    }
    return PreToken(text: value, start: start + from, end: start + to);
  }

  @override
  String toString() => 'PreToken("$text", [$start:$end])';
}

/// BERT normalization and punctuation splitting with original-text alignment.
class BertPreTokenizer {
  /// Whether to lowercase text.
  final bool lowercase;

  /// Whether to apply canonical decomposition and remove nonspacing marks.
  final bool stripAccents;

  /// Whether to split Chinese characters individually.
  final bool handleChineseChars;

  /// Whether to remove control characters and normalize whitespace.
  final bool cleanText;

  /// Whether to split whitespace and punctuation after normalization.
  final bool split;

  /// Creates a BERT pre-tokenizer.
  const BertPreTokenizer({
    this.lowercase = true,
    this.stripAccents = true,
    this.handleChineseChars = true,
    this.cleanText = true,
    this.split = true,
  });

  static final _punctuation = RegExp(r'\p{P}', unicode: true);
  static final _control = RegExp(r'[\p{Cc}\p{Cf}]', unicode: true);
  static final _separator = RegExp(r'\p{Zs}', unicode: true);
  static final _whitespace = RegExp(r'\p{White_Space}', unicode: true);

  /// Normalizes and splits [text], preserving original code-point offsets.
  List<PreToken> preTokenize(String text) => splitNormalized(normalize(text));

  /// Normalizes text while retaining original code-point alignment.
  PreToken normalize(String text, {int offset = 0}) {
    final normalized = StringBuffer();
    final alignment = <(int, int)>[];
    void append(String value, int position) {
      normalized.write(value);
      alignment.addAll(List.filled(value.length, (position, position + 1)));
    }

    var position = offset;
    for (final rune in text.runes) {
      final char = String.fromCharCode(rune);
      final isSpace =
          rune == 32 ||
          rune == 9 ||
          rune == 10 ||
          rune == 13 ||
          _separator.hasMatch(char);
      if (cleanText &&
          (rune == 0 ||
              rune == 0xfffd ||
              (!isSpace && _control.hasMatch(char)))) {
        position++;
        continue;
      }
      if (cleanText && isSpace) {
        append(' ', position++);
        continue;
      }
      if (handleChineseChars && _isChineseChar(rune)) append(' ', position);
      var value = stripAccents ? _withoutAccents(rune) : char;
      if (lowercase) {
        // Dart uses simple lowercase; Unicode full lowercase expands dotted I.
        value = value.replaceAll('\u0130', 'i\u0307').toLowerCase();
      }
      append(value, position);
      if (handleChineseChars && _isChineseChar(rune)) append(' ', position);
      position++;
    }
    final value = normalized.toString();
    return value.isEmpty
        ? PreToken(text: '', start: offset, end: offset)
        : PreToken._(value, alignment);
  }

  /// Splits already normalized text without normalizing it a second time.
  List<PreToken> splitNormalized(PreToken input) {
    final value = input.text;
    if (value.isEmpty) return [];
    if (!split) return [input];
    final result = <PreToken>[];
    var start = 0;
    var index = 0;
    void emit(int end) {
      if (start < end) {
        result.add(input.slice(start, end));
      }
    }

    for (final rune in value.runes) {
      final char = String.fromCharCode(rune);
      final width = char.length;
      final whitespace = _whitespace.hasMatch(char);
      final punctuation =
          (rune >= 33 && rune <= 47) ||
          (rune >= 58 && rune <= 64) ||
          (rune >= 91 && rune <= 96) ||
          (rune >= 123 && rune <= 126) ||
          _punctuation.hasMatch(char);
      if (whitespace || punctuation) {
        emit(index);
        if (punctuation && !whitespace) {
          result.add(input.slice(index, index + width));
        }
        start = index + width;
      }
      index += width;
    }
    emit(index);
    return result;
  }

  String _withoutAccents(int rune) {
    if (rune >= 0xac00 && rune <= 0xd7a3) {
      final syllable = rune - 0xac00;
      return String.fromCharCodes([
        0x1100 + syllable ~/ 588,
        0x1161 + (syllable % 588) ~/ 28,
        if (syllable % 28 != 0) 0x11a7 + syllable % 28,
      ]);
    }
    return canonicalWithoutAccents[rune] ?? String.fromCharCode(rune);
  }

  bool _isChineseChar(int rune) =>
      (rune >= 0x4e00 && rune <= 0x9fff) ||
      (rune >= 0x3400 && rune <= 0x4dbf) ||
      (rune >= 0x20000 && rune <= 0x2a6df) ||
      (rune >= 0x2a700 && rune <= 0x2b73f) ||
      (rune >= 0x2b740 && rune <= 0x2b81f) ||
      (rune >= 0x2b820 && rune <= 0x2ceaf) ||
      (rune >= 0xf900 && rune <= 0xfaff) ||
      (rune >= 0x2f800 && rune <= 0x2fa1f);
}

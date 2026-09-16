import 'added_token.dart';
import 'pre_tokenizer.dart';
import 'vocabulary.dart';
import 'unicode_word_data.dart';

/// Immutable registration snapshot shared by encoding and isolate workers.
class AddedVocabulary {
  final Vocabulary vocab;
  final Map<int, AddedToken> tokens;
  final Set<int> specialIds;
  final Map<int, String> decoded;
  final Map<String, int> _raw;
  final Map<String, int> _normalized;
  final RegExp? _rawPattern;
  final RegExp? _normalizedPattern;

  AddedVocabulary.empty(this.vocab)
    : tokens = const {},
      specialIds = const {},
      decoded = const {},
      _raw = const {},
      _normalized = const {},
      _rawPattern = null,
      _normalizedPattern = null;

  AddedVocabulary._(
    this.vocab,
    this.tokens,
    this.specialIds,
    this.decoded,
    this._raw,
    this._normalized,
  ) : _rawPattern = _pattern(_raw),
      _normalizedPattern = _pattern(_normalized);

  static RegExp? _pattern(Map<String, int> entries) {
    if (entries.isEmpty) return null;
    final patterns = entries.keys.toList()
      ..sort((a, b) => b.length.compareTo(a.length));
    return RegExp(patterns.map(RegExp.escape).join('|'), unicode: true);
  }

  (AddedVocabulary, int) register(
    List<AddedToken> additions,
    BertPreTokenizer normalizer,
  ) {
    final updated = Map<int, AddedToken>.of(tokens);
    final ids = <String, int>{};
    final specials = Set<int>.of(specialIds);
    var nextId = vocab.size;
    var count = 0;
    for (final token in additions) {
      if (token.content.isEmpty) continue;
      final id =
          ids[token.content] ??
          (vocab.contains(token.content)
              ? vocab.tokenToId(token.content)
              : nextId++);
      if (id > 0x7fffffff) {
        throw ArgumentError('Added token ID exceeds Int32 range');
      }
      if (updated[id] == token) continue;
      ids[token.content] = id;
      updated[id] = token;
      if (token.special) specials.add(id);
      count++;
    }
    if (count == 0) return (this, 0);
    final raw = <String, int>{};
    final normalized = <String, int>{};
    final decoded = <int, String>{};
    // Lowest ID wins normalized-pattern collisions, independently of map order.
    for (final id in updated.keys.toList()..sort()) {
      final token = updated[id]!;
      final content = token.normalized
          ? normalizer.normalize(token.content).text
          : token.content;
      decoded[id] = content;
      if (content.isEmpty) continue;
      (token.normalized ? normalized : raw).putIfAbsent(content, () => id);
    }
    return (
      AddedVocabulary._(
        vocab.withAddedTokens(ids),
        updated,
        specials,
        decoded,
        raw,
        normalized,
      ),
      count,
    );
  }

  static bool _wordAt(String text, int index) {
    if (index < 0 || index >= text.length) return false;
    var rune = text.codeUnitAt(index);
    if (rune >= 0xd800 && rune <= 0xdbff && index + 1 < text.length) {
      final low = text.codeUnitAt(index + 1);
      if (low >= 0xdc00 && low <= 0xdfff) {
        rune = 0x10000 + ((rune - 0xd800) << 10) + low - 0xdc00;
      }
    }
    var left = 0;
    var right = addedTokenWordRanges.length ~/ 2;
    while (left < right) {
      final middle = (left + right) ~/ 2;
      if (rune < addedTokenWordRanges[middle * 2]) {
        right = middle;
      } else if (rune > addedTokenWordRanges[middle * 2 + 1]) {
        left = middle + 1;
      } else {
        return true;
      }
    }
    return false;
  }

  static bool _wordBefore(String text, int index) {
    if (index == 0) return false;
    var previous = index - 1;
    final unit = text.codeUnitAt(previous);
    if (unit >= 0xdc00 && unit <= 0xdfff && previous > 0) previous--;
    return _wordAt(text, previous);
  }

  static final _spaceStart = RegExp(r'^\p{White_Space}*', unicode: true);
  static final _spaceEnd = RegExp(r'\p{White_Space}*$', unicode: true);

  /// Extracts raw tokens, then normalizes and extracts normalized tokens.
  Iterable<(PreToken, int?)> extract(
    String text,
    BertPreTokenizer normalizer,
  ) sync* {
    const identity = BertPreTokenizer(
      lowercase: false,
      stripAccents: false,
      handleChineseChars: false,
      cleanText: false,
      split: false,
    );
    var position = 0;
    var previous = 0;
    for (final (start, end, id) in _matches(text, _rawPattern, _raw)) {
      position = start >= previous
          ? position + text.substring(previous, start).runes.length
          : text.substring(0, start).runes.length;
      final piece = text.substring(start, end);
      if (id != null) {
        yield (identity.normalize(piece, offset: position), id);
      } else {
        yield* _split(
          normalizer.normalize(piece, offset: position),
          _normalizedPattern,
          _normalized,
        );
      }
      position += piece.runes.length;
      previous = end;
    }
  }

  Iterable<(PreToken, int?)> _split(
    PreToken input,
    RegExp? pattern,
    Map<String, int> ids,
  ) sync* {
    for (final (start, end, id) in _matches(input.text, pattern, ids)) {
      yield (input.slice(start, end), id);
    }
  }

  Iterable<(int, int, int?)> _matches(
    String text,
    RegExp? pattern,
    Map<String, int> ids,
  ) sync* {
    if (text.isEmpty) return;
    var consumed = 0;
    for (final match in pattern?.allMatches(text) ?? const <RegExpMatch>[]) {
      var start = match.start;
      var end = match.end;
      final id = ids[match.group(0)]!;
      final token = tokens[id]!;
      if (token.singleWord &&
          (_wordBefore(text, start) || _wordAt(text, end))) {
        continue;
      }
      if (token.lstrip) {
        start = _spaceEnd.firstMatch(text.substring(0, start))!.start;
        if (start < consumed) start = consumed;
      }
      if (token.rstrip) end += _spaceStart.firstMatch(text.substring(end))!.end;
      if (consumed < start) yield (consumed, start, null);
      yield (start, end, id);
      consumed = end;
    }
    if (consumed < text.length) {
      yield (consumed, text.length, null);
    }
  }
}

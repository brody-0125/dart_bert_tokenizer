import 'dart:io';

import 'trie.dart';

/// Constants for BERT special tokens.
///
/// These tokens have special meaning in BERT models:
/// - `[PAD]`: Padding token for batch processing
/// - `[UNK]`: Unknown token for out-of-vocabulary words
/// - `[CLS]`: Classification token at the start of each sequence
/// - `[SEP]`: Separator token between sequences
/// - `[MASK]`: Mask token for masked language modeling
class SpecialTokens {
  /// Padding token used to pad sequences to equal length.
  static const String pad = '[PAD]';

  /// Unknown token used for out-of-vocabulary words.
  static const String unk = '[UNK]';

  /// Classification token added at the start of each sequence.
  static const String cls = '[CLS]';

  /// Separator token used between text pairs.
  static const String sep = '[SEP]';

  /// Mask token used in masked language modeling tasks.
  static const String mask = '[MASK]';

  /// List of all default special tokens.
  static const List<String> defaults = [pad, unk, cls, sep, mask];
}

/// A vocabulary mapping between tokens and their integer IDs.
///
/// The vocabulary is loaded from a file where each line contains a token,
/// with the line number (0-indexed) being the token's ID.
///
/// Internally uses trie data structures for efficient longest-prefix matching
/// during tokenization.
///
/// ## Loading a Vocabulary
///
/// ```dart
/// // From file (async)
/// final vocab = await Vocabulary.fromFile('vocab.txt');
///
/// // From file (sync)
/// final vocab = Vocabulary.fromFileSync('vocab.txt');
///
/// // From string content
/// final vocab = Vocabulary.fromString(vocabContent);
/// ```
class Vocabulary {
  final Map<String, int> _tokenToId = {};
  final List<String> _idToToken = [];
  final Trie _trie = Trie();
  final Trie _subwordTrie = Trie();

  /// The prefix used for subword tokens (default: `##`).
  final String subwordPrefix;

  /// Creates an empty vocabulary with the given subword prefix.
  Vocabulary({this.subwordPrefix = '##'});

  /// The number of tokens in this vocabulary.
  int get size => _idToToken.length;

  /// Trie for looking up whole-word tokens.
  Trie get trie => _trie;

  /// Trie for looking up subword tokens (without prefix).
  Trie get subwordTrie => _subwordTrie;

  /// The ID of the unknown token `[UNK]`.
  int get unkTokenId => _tokenToId[SpecialTokens.unk] ?? 100;

  /// The ID of the classification token `[CLS]`.
  int get clsTokenId => _tokenToId[SpecialTokens.cls] ?? 101;

  /// The ID of the separator token `[SEP]`.
  int get sepTokenId => _tokenToId[SpecialTokens.sep] ?? 102;

  /// The ID of the padding token `[PAD]`.
  int get padTokenId => _tokenToId[SpecialTokens.pad] ?? 0;

  /// The ID of the mask token `[MASK]`.
  int get maskTokenId => _tokenToId[SpecialTokens.mask] ?? 103;

  /// Loads a vocabulary from a file asynchronously.
  ///
  /// The file should contain one token per line, with the line number
  /// (0-indexed) being the token's ID.
  static Future<Vocabulary> fromFile(
    String path, {
    String subwordPrefix = '##',
  }) async {
    final file = File(path);
    final lines = await file.readAsLines();
    return Vocabulary._fromLines(lines, subwordPrefix: subwordPrefix);
  }

  /// Loads a vocabulary from a file synchronously.
  static Vocabulary fromFileSync(String path, {String subwordPrefix = '##'}) {
    final file = File(path);
    final lines = file.readAsLinesSync();
    return Vocabulary._fromLines(lines, subwordPrefix: subwordPrefix);
  }

  /// Creates a vocabulary from a string containing tokens separated by newlines.
  static Vocabulary fromString(String content, {String subwordPrefix = '##'}) {
    final lines = content.split('\n');
    return Vocabulary._fromLines(lines, subwordPrefix: subwordPrefix);
  }

  /// Creates a vocabulary from a list of token strings.
  static Vocabulary fromTokens(
    List<String> tokens, {
    String subwordPrefix = '##',
  }) {
    return Vocabulary._fromLines(tokens, subwordPrefix: subwordPrefix);
  }

  static Vocabulary _fromLines(
    List<String> lines, {
    String subwordPrefix = '##',
  }) {
    final vocab = Vocabulary(subwordPrefix: subwordPrefix);

    for (var i = 0; i < lines.length; i++) {
      final token = lines[i].trim();
      if (token.isEmpty) continue;

      vocab._addToken(token, i);
    }

    return vocab;
  }

  void _addToken(String token, int id) {
    _tokenToId[token] = id;

    while (_idToToken.length <= id) {
      _idToToken.add('');
    }
    _idToToken[id] = token;

    if (token.startsWith(subwordPrefix)) {
      final subword = token.substring(subwordPrefix.length);
      _subwordTrie.insert(subword, id);
    } else if (!token.startsWith('[') || !token.endsWith(']')) {
      _trie.insert(token, id);
    }
  }

  /// Converts a token string to its ID.
  ///
  /// Returns the unknown token ID if the token is not in the vocabulary.
  int tokenToId(String token) {
    return _tokenToId[token] ?? unkTokenId;
  }

  /// Converts a token ID to its string representation.
  ///
  /// Returns `[UNK]` if the ID is out of range.
  String idToToken(int id) {
    if (id < 0 || id >= _idToToken.length) {
      return SpecialTokens.unk;
    }
    final token = _idToToken[id];
    return token.isEmpty ? SpecialTokens.unk : token;
  }

  /// Returns whether the vocabulary contains the given token.
  bool contains(String token) => _tokenToId.containsKey(token);

  /// Returns whether the token is a special token (e.g., `[CLS]`, `[SEP]`).
  bool isSpecialToken(String token) {
    return token.startsWith('[') && token.endsWith(']');
  }

  /// Returns an unmodifiable view of the token-to-ID mapping.
  Map<String, int> get vocabularyMap => Map.unmodifiable(_tokenToId);

  /// Returns an unmodifiable list of all tokens in ID order.
  List<String> get tokens => List.unmodifiable(_idToToken);

  /// Finds the longest matching token starting at the given position.
  ///
  /// - [text]: The text to search in.
  /// - [startIndex]: Position to start matching from.
  /// - [isSubword]: If true, searches the subword trie instead.
  ///
  /// Returns the match, or `null` if no match is found.
  TrieMatch? findLongestMatch(
    String text, {
    int startIndex = 0,
    bool isSubword = false,
  }) {
    final targetTrie = isSubword ? _subwordTrie : _trie;
    return targetTrie.findLongestPrefix(text, startIndex);
  }

  @override
  String toString() => 'Vocabulary(size: $size)';
}

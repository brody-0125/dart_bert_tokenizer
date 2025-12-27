/// A node in a [Trie] data structure.
///
/// Each node represents a character in the trie and can optionally mark
/// the end of a token.
class TrieNode {
  /// Child nodes keyed by Unicode code point.
  final Map<int, TrieNode> children = {};

  /// The token ID if this node marks the end of a token, or `null`.
  int? tokenId;

  /// The token string if this node marks the end of a token, or `null`.
  String? token;

  /// Whether this node marks the end of a valid token.
  bool get isEndOfToken => tokenId != null;
}

/// A trie (prefix tree) data structure for efficient token lookup.
///
/// Used internally by [Vocabulary] for fast longest-prefix matching during
/// tokenization. Supports Unicode characters through code point indexing.
///
/// Example:
/// ```dart
/// final trie = Trie();
/// trie.insert('hello', 1);
/// trie.insert('help', 2);
///
/// final match = trie.findLongestPrefix('hello world');
/// print(match?.token); // 'hello'
/// ```
class Trie {
  final TrieNode _root = TrieNode();
  int _size = 0;

  /// The number of tokens in this trie.
  int get size => _size;

  /// Inserts a token into the trie.
  ///
  /// - [token]: The token string to insert.
  /// - [tokenId]: The ID to associate with this token.
  void insert(String token, int tokenId) {
    var node = _root;

    for (final codePoint in token.runes) {
      node = node.children.putIfAbsent(codePoint, () => TrieNode());
    }

    if (node.tokenId == null) {
      _size++;
    }
    node.tokenId = tokenId;
    node.token = token;
  }

  /// Looks up a token and returns its ID, or `null` if not found.
  int? lookup(String token) {
    var node = _root;

    for (final codePoint in token.runes) {
      final child = node.children[codePoint];
      if (child == null) {
        return null;
      }
      node = child;
    }

    return node.tokenId;
  }

  /// Returns whether the trie contains the given token.
  bool contains(String token) => lookup(token) != null;

  /// Finds the longest matching token starting at the given position.
  ///
  /// - [text]: The text to search in.
  /// - [startIndex]: Position to start matching from (default: 0).
  ///
  /// Returns a [TrieMatch] with the longest match, or `null` if no match.
  TrieMatch? findLongestPrefix(String text, [int startIndex = 0]) {
    var node = _root;
    TrieMatch? lastMatch;

    for (var i = startIndex; i < text.length; i++) {
      final codeUnit = text.codeUnitAt(i);

      int codePoint;
      if (codeUnit >= 0xD800 && codeUnit <= 0xDBFF && i + 1 < text.length) {
        final low = text.codeUnitAt(i + 1);
        if (low >= 0xDC00 && low <= 0xDFFF) {
          codePoint = 0x10000 + ((codeUnit - 0xD800) << 10) + (low - 0xDC00);
          i++;
        } else {
          codePoint = codeUnit;
        }
      } else {
        codePoint = codeUnit;
      }

      final child = node.children[codePoint];
      if (child == null) {
        break;
      }

      node = child;

      if (node.isEndOfToken) {
        lastMatch = TrieMatch(
          token: node.token!,
          tokenId: node.tokenId!,
          start: startIndex,
          end: i + 1,
        );
      }
    }

    return lastMatch;
  }

  /// Finds all matching tokens starting at the given position.
  ///
  /// Returns matches in order of increasing length.
  List<TrieMatch> findAllPrefixes(String text, [int startIndex = 0]) {
    final matches = <TrieMatch>[];
    var node = _root;

    for (var i = startIndex; i < text.length; i++) {
      final codeUnit = text.codeUnitAt(i);

      int codePoint;
      if (codeUnit >= 0xD800 && codeUnit <= 0xDBFF && i + 1 < text.length) {
        final low = text.codeUnitAt(i + 1);
        if (low >= 0xDC00 && low <= 0xDFFF) {
          codePoint = 0x10000 + ((codeUnit - 0xD800) << 10) + (low - 0xDC00);
          i++;
        } else {
          codePoint = codeUnit;
        }
      } else {
        codePoint = codeUnit;
      }

      final child = node.children[codePoint];
      if (child == null) {
        break;
      }

      node = child;

      if (node.isEndOfToken) {
        matches.add(
          TrieMatch(
            token: node.token!,
            tokenId: node.tokenId!,
            start: startIndex,
            end: i + 1,
          ),
        );
      }
    }

    return matches;
  }

  /// Removes all tokens from this trie.
  void clear() {
    _root.children.clear();
    _size = 0;
  }
}

/// Represents a successful token match from a [Trie] lookup.
///
/// Contains the matched token, its ID, and the character range in the
/// source text.
class TrieMatch {
  /// The matched token string.
  final String token;

  /// The vocabulary ID of the matched token.
  final int tokenId;

  /// The starting character position of the match.
  final int start;

  /// The ending character position of the match (exclusive).
  final int end;

  /// Creates a trie match result.
  const TrieMatch({
    required this.token,
    required this.tokenId,
    required this.start,
    required this.end,
  });

  /// The length of the match in characters.
  int get length => end - start;

  @override
  String toString() => 'TrieMatch(token: $token, id: $tokenId, [$start:$end])';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is TrieMatch &&
          token == other.token &&
          tokenId == other.tokenId &&
          start == other.start &&
          end == other.end;

  @override
  int get hashCode => Object.hash(token, tokenId, start, end);
}

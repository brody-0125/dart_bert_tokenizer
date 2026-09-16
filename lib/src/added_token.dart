/// A token extracted before WordPiece segmentation.
///
/// Normal tokens match normalized input by default; special tokens match raw
/// input unless [normalized] is explicitly set. Adding tokens does not resize
/// a model's embedding matrix.
class AddedToken {
  /// Original token spelling used for vocabulary lookup.
  final String content;

  /// Require Unicode word boundaries on both sides of the match.
  final bool singleWord;

  /// Include adjacent whitespace to the left in the match and its offsets.
  final bool lstrip;

  /// Include adjacent whitespace to the right in the match and its offsets.
  final bool rstrip;

  /// Match after applying the tokenizer's normalizer to input and content.
  final bool normalized;

  /// Omit this token when decoding with skipSpecialTokens enabled.
  final bool special;

  /// Creates an immutable token definition. Empty dynamic tokens are ignored.
  const AddedToken(
    this.content, {
    this.singleWord = false,
    this.lstrip = false,
    this.rstrip = false,
    bool? normalized,
    this.special = false,
  }) : normalized = normalized ?? !special;

  @override
  bool operator ==(Object other) =>
      other is AddedToken &&
      content == other.content &&
      singleWord == other.singleWord &&
      lstrip == other.lstrip &&
      rstrip == other.rstrip &&
      normalized == other.normalized &&
      special == other.special;

  @override
  int get hashCode =>
      Object.hash(content, singleWord, lstrip, rstrip, normalized, special);
}

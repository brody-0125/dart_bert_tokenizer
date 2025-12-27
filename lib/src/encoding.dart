import 'dart:typed_data';

/// Strategy for truncating text pairs when they exceed the maximum length.
///
/// Used with [WordPieceTokenizer.encodePair] and [TruncationConfig] to
/// control how text pairs are truncated.
enum TruncationStrategy {
  /// Removes tokens from the longer sequence first, alternating if equal.
  ///
  /// This is the default strategy and typically produces the best results
  /// for most use cases.
  longestFirst,

  /// Only truncates the first sequence.
  ///
  /// Useful when the second sequence (e.g., context) should be preserved.
  onlyFirst,

  /// Only truncates the second sequence.
  ///
  /// Useful when the first sequence (e.g., question) should be preserved.
  onlySecond,

  /// Does not truncate; returns the full sequences.
  ///
  /// Use with caution as this may produce sequences longer than the model
  /// can handle.
  doNotTruncate,
}

/// The result of tokenizing text with a [WordPieceTokenizer].
///
/// Contains all the information needed to feed text into a BERT model:
/// token IDs, attention masks, type IDs (for distinguishing text pairs),
/// and mappings between tokens and original text positions.
///
/// ## Properties
///
/// - [ids]: Token IDs for model input
/// - [tokens]: Token strings for debugging
/// - [attentionMask]: 1 for real tokens, 0 for padding
/// - [typeIds]: Segment IDs (0 for first sequence, 1 for second)
/// - [specialTokensMask]: 1 for special tokens, 0 for regular tokens
///
/// ## Token-Character Mapping
///
/// The encoding provides methods to map between tokens and characters:
///
/// ```dart
/// // Find which token contains character at position 5
/// final tokenIdx = encoding.charToToken(5);
///
/// // Find character span for token at index 2
/// final (start, end) = encoding.tokenToChars(2)!;
/// ```
class Encoding {
  /// The token strings in this encoding.
  final List<String> tokens;

  /// The token IDs for model input.
  ///
  /// These are the integer indices into the vocabulary that the model uses.
  final Int32List ids;

  /// Segment/type IDs distinguishing text pairs.
  ///
  /// For single text: all 0s.
  /// For text pairs: 0 for first sequence, 1 for second sequence.
  final Uint8List typeIds;

  /// Attention mask indicating which tokens are real vs padding.
  ///
  /// 1 for real tokens, 0 for padding tokens.
  final Uint8List attentionMask;

  /// Mask indicating which tokens are special tokens.
  ///
  /// 1 for special tokens (`[CLS]`, `[SEP]`, `[PAD]`), 0 for regular tokens.
  final Uint8List specialTokensMask;

  /// Character offsets for each token as `(start, end)` pairs.
  ///
  /// Maps each token back to its position in the original text.
  /// Special tokens have offset `(0, 0)`.
  final List<(int, int)> offsets;

  /// Word indices for each token.
  ///
  /// Multiple tokens from the same word share the same word ID.
  /// Special tokens have `null` word ID.
  final List<int?> wordIds;

  final List<int?>? _sequenceIds;

  /// Creates an encoding with the specified data.
  ///
  /// This constructor is typically not called directly; use
  /// [WordPieceTokenizer.encode] or [EncodingBuilder] instead.
  Encoding({
    required this.tokens,
    required List<int> ids,
    required List<int> typeIds,
    required List<int> attentionMask,
    required List<int> specialTokensMask,
    required this.offsets,
    required this.wordIds,
    List<int?>? sequenceIds,
  }) : ids = ids is Int32List ? ids : Int32List.fromList(ids),
       typeIds = typeIds is Uint8List ? typeIds : Uint8List.fromList(typeIds),
       attentionMask = attentionMask is Uint8List
           ? attentionMask
           : Uint8List.fromList(attentionMask),
       specialTokensMask = specialTokensMask is Uint8List
           ? specialTokensMask
           : Uint8List.fromList(specialTokensMask),
       _sequenceIds = sequenceIds;

  Encoding._typed({
    required this.tokens,
    required this.ids,
    required this.typeIds,
    required this.attentionMask,
    required this.specialTokensMask,
    required this.offsets,
    required this.wordIds,
    List<int?>? sequenceIds,
  }) : _sequenceIds = sequenceIds;

  /// The number of tokens in this encoding.
  int get length => tokens.length;

  /// Whether this encoding contains no tokens.
  bool get isEmpty => tokens.isEmpty;

  /// Whether this encoding contains at least one token.
  bool get isNotEmpty => tokens.isNotEmpty;

  /// Sequence IDs indicating which sequence each token belongs to.
  ///
  /// Returns 0 for first sequence tokens, 1 for second sequence tokens,
  /// and `null` for special tokens.
  List<int?> get sequenceIds {
    if (_sequenceIds != null) return _sequenceIds;

    return List.generate(length, (i) {
      if (specialTokensMask[i] == 1) return null;
      return typeIds[i];
    });
  }

  /// Returns the number of sequences in this encoding.
  ///
  /// Returns 0 if empty, 1 for single text, 2 for text pairs.
  int get nSequences {
    final seqIds = sequenceIds;
    if (seqIds.any((id) => id == 1)) return 2;
    if (seqIds.any((id) => id == 0)) return 1;
    return 0;
  }

  /// Finds the token index containing the given character position.
  ///
  /// - [charPos]: Character position in the original text.
  /// - [sequenceIndex]: Which sequence to search (0 or 1 for pairs).
  ///
  /// Returns the token index, or `null` if no token contains this position.
  int? charToToken(int charPos, {int sequenceIndex = 0}) {
    final seqIds = sequenceIds;
    for (var i = 0; i < length; i++) {
      if (seqIds[i] != sequenceIndex) continue;
      final (start, end) = offsets[i];
      if (charPos >= start && charPos < end) {
        return i;
      }
    }
    return null;
  }

  /// Finds the word index containing the given character position.
  ///
  /// - [charPos]: Character position in the original text.
  /// - [sequenceIndex]: Which sequence to search (0 or 1 for pairs).
  int? charToWord(int charPos, {int sequenceIndex = 0}) {
    final tokenIdx = charToToken(charPos, sequenceIndex: sequenceIndex);
    if (tokenIdx == null) return null;
    return wordIds[tokenIdx];
  }

  /// Returns the character span for a token at the given index.
  ///
  /// Returns `(start, end)` character positions, or `null` for special tokens.
  (int, int)? tokenToChars(int tokenIndex) {
    if (tokenIndex < 0 || tokenIndex >= length) return null;
    final offset = offsets[tokenIndex];
    if (offset == (0, 0) && specialTokensMask[tokenIndex] == 1) return null;
    return offset;
  }

  /// Returns the word index for a token at the given index.
  int? tokenToWord(int tokenIndex) {
    if (tokenIndex < 0 || tokenIndex >= length) return null;
    return wordIds[tokenIndex];
  }

  /// Returns which sequence a token belongs to (0, 1, or null for special).
  int? tokenToSequence(int tokenIndex) {
    if (tokenIndex < 0 || tokenIndex >= length) return null;
    return sequenceIds[tokenIndex];
  }

  /// Returns the character span for an entire word.
  ///
  /// - [wordIndex]: The word index to look up.
  /// - [sequenceIndex]: Which sequence to search.
  (int, int)? wordToChars(int wordIndex, {int sequenceIndex = 0}) {
    final seqIds = sequenceIds;
    int? start;
    int? end;

    for (var i = 0; i < length; i++) {
      if (seqIds[i] != sequenceIndex) continue;
      if (wordIds[i] != wordIndex) continue;

      final (tStart, tEnd) = offsets[i];
      if (start == null || tStart < start) start = tStart;
      if (end == null || tEnd > end) end = tEnd;
    }

    if (start == null || end == null) return null;
    return (start, end);
  }

  /// Returns the token span for an entire word.
  ///
  /// Returns `(startTokenIdx, endTokenIdx)` where end is exclusive.
  (int, int)? wordToTokens(int wordIndex, {int sequenceIndex = 0}) {
    final seqIds = sequenceIds;
    int? start;
    int? end;

    for (var i = 0; i < length; i++) {
      if (seqIds[i] != sequenceIndex) continue;
      if (wordIds[i] != wordIndex) continue;

      start ??= i;
      end = i + 1;
    }

    if (start == null || end == null) return null;
    return (start, end);
  }

  /// Creates an empty encoding with no tokens.
  factory Encoding.empty() => Encoding._typed(
    tokens: const [],
    ids: Int32List(0),
    typeIds: Uint8List(0),
    attentionMask: Uint8List(0),
    specialTokensMask: Uint8List(0),
    offsets: const [],
    wordIds: const [],
    sequenceIds: const [],
  );

  /// Merges multiple encodings into a single encoding.
  ///
  /// - [encodings]: List of encodings to merge.
  /// - [growingOffsets]: If true, offsets accumulate across encodings.
  static Encoding merge(
    List<Encoding> encodings, {
    bool growingOffsets = true,
  }) {
    if (encodings.isEmpty) return Encoding.empty();
    if (encodings.length == 1) return encodings.first;

    var totalLength = 0;
    for (final enc in encodings) {
      totalLength += enc.length;
    }

    final tokens = <String>[];
    final ids = Int32List(totalLength);
    final typeIds = Uint8List(totalLength);
    final attentionMask = Uint8List(totalLength);
    final specialTokensMask = Uint8List(totalLength);
    final offsets = <(int, int)>[];
    final wordIds = <int?>[];
    final sequenceIds = <int?>[];

    var offsetShift = 0;
    var wordIdShift = 0;
    var destIndex = 0;

    for (var encIdx = 0; encIdx < encodings.length; encIdx++) {
      final enc = encodings[encIdx];

      tokens.addAll(enc.tokens);

      ids.setRange(destIndex, destIndex + enc.length, enc.ids);
      typeIds.setRange(destIndex, destIndex + enc.length, enc.typeIds);
      attentionMask.setRange(
        destIndex,
        destIndex + enc.length,
        enc.attentionMask,
      );
      specialTokensMask.setRange(
        destIndex,
        destIndex + enc.length,
        enc.specialTokensMask,
      );

      if (growingOffsets) {
        for (final (start, end) in enc.offsets) {
          if (start == 0 && end == 0) {
            offsets.add((0, 0));
          } else {
            offsets.add((start + offsetShift, end + offsetShift));
          }
        }
        final maxEnd = enc.offsets
            .where((o) => o != (0, 0))
            .fold<int>(0, (max, o) => o.$2 > max ? o.$2 : max);
        offsetShift += maxEnd;
      } else {
        offsets.addAll(enc.offsets);
      }

      for (final wid in enc.wordIds) {
        if (wid == null) {
          wordIds.add(null);
        } else {
          wordIds.add(wid + wordIdShift);
        }
      }
      final maxWordId = enc.wordIds
          .where((w) => w != null)
          .fold<int>(0, (max, w) => w! > max ? w : max);
      wordIdShift += maxWordId + 1;

      sequenceIds.addAll(enc.sequenceIds);
      destIndex += enc.length;
    }

    return Encoding._typed(
      tokens: tokens,
      ids: ids,
      typeIds: typeIds,
      attentionMask: attentionMask,
      specialTokensMask: specialTokensMask,
      offsets: offsets,
      wordIds: wordIds,
      sequenceIds: sequenceIds,
    );
  }

  /// Returns a new encoding padded to the target length.
  ///
  /// - [targetLength]: The desired total length.
  /// - [padTokenId]: The token ID to use for padding.
  /// - [padToken]: The token string for padding (default: `[PAD]`).
  /// - [padOnRight]: Whether to pad on the right side (default: true).
  ///
  /// Returns `this` if already at or above target length.
  Encoding withPadding({
    required int targetLength,
    required int padTokenId,
    String padToken = '[PAD]',
    bool padOnRight = true,
  }) {
    if (length >= targetLength) {
      return this;
    }

    final srcLen = length;
    final dstOffset = padOnRight ? 0 : targetLength - srcLen;

    final paddedTokens = List<String>.filled(targetLength, padToken);
    paddedTokens.setRange(dstOffset, dstOffset + srcLen, tokens);

    final paddedIds = Int32List(targetLength);
    for (var i = 0; i < targetLength; i++) {
      paddedIds[i] = padTokenId;
    }
    paddedIds.setRange(dstOffset, dstOffset + srcLen, ids);

    final paddedTypeIds = Uint8List(targetLength);
    paddedTypeIds.setRange(dstOffset, dstOffset + srcLen, typeIds);

    final paddedAttentionMask = Uint8List(targetLength);
    paddedAttentionMask.setRange(dstOffset, dstOffset + srcLen, attentionMask);

    final paddedSpecialTokensMask = Uint8List(targetLength);
    for (var i = 0; i < targetLength; i++) {
      paddedSpecialTokensMask[i] = 1;
    }
    paddedSpecialTokensMask.setRange(
      dstOffset,
      dstOffset + srcLen,
      specialTokensMask,
    );

    final paddedOffsets = List<(int, int)>.filled(targetLength, (0, 0));
    paddedOffsets.setRange(dstOffset, dstOffset + srcLen, offsets);

    final paddedWordIds = List<int?>.filled(targetLength, null);
    paddedWordIds.setRange(dstOffset, dstOffset + srcLen, wordIds);

    final paddedSequenceIds = List<int?>.filled(targetLength, null);
    paddedSequenceIds.setRange(dstOffset, dstOffset + srcLen, sequenceIds);

    return Encoding._typed(
      tokens: paddedTokens,
      ids: paddedIds,
      typeIds: paddedTypeIds,
      attentionMask: paddedAttentionMask,
      specialTokensMask: paddedSpecialTokensMask,
      offsets: paddedOffsets,
      wordIds: paddedWordIds,
      sequenceIds: paddedSequenceIds,
    );
  }

  /// Returns a new encoding padded to a multiple of the given value.
  ///
  /// Useful for hardware optimization where tensor dimensions should be
  /// multiples of certain values.
  Encoding withPaddingToMultipleOf({
    required int multiple,
    required int padTokenId,
    String padToken = '[PAD]',
    bool padOnRight = true,
  }) {
    if (multiple <= 0) return this;

    final remainder = length % multiple;
    if (remainder == 0) return this;

    final targetLength = length + (multiple - remainder);
    return withPadding(
      targetLength: targetLength,
      padTokenId: padTokenId,
      padToken: padToken,
      padOnRight: padOnRight,
    );
  }

  /// Returns a new encoding truncated to the maximum length.
  ///
  /// - [maxLength]: The maximum number of tokens.
  /// - [truncateFromEnd]: If true, removes tokens from the end; otherwise
  ///   removes from the beginning.
  ///
  /// Returns `this` if already at or below max length.
  Encoding withTruncation({
    required int maxLength,
    bool truncateFromEnd = true,
  }) {
    if (length <= maxLength) {
      return this;
    }

    final seqIds = sequenceIds;

    if (truncateFromEnd) {
      return Encoding._typed(
        tokens: tokens.sublist(0, maxLength),
        ids: ids.sublist(0, maxLength),
        typeIds: typeIds.sublist(0, maxLength),
        attentionMask: attentionMask.sublist(0, maxLength),
        specialTokensMask: specialTokensMask.sublist(0, maxLength),
        offsets: offsets.sublist(0, maxLength),
        wordIds: wordIds.sublist(0, maxLength),
        sequenceIds: seqIds.sublist(0, maxLength),
      );
    } else {
      final start = length - maxLength;
      return Encoding._typed(
        tokens: tokens.sublist(start),
        ids: ids.sublist(start),
        typeIds: typeIds.sublist(start),
        attentionMask: attentionMask.sublist(start),
        specialTokensMask: specialTokensMask.sublist(start),
        offsets: offsets.sublist(start),
        wordIds: wordIds.sublist(start),
        sequenceIds: seqIds.sublist(start),
      );
    }
  }

  /// Converts this encoding to a Map representation.
  ///
  /// Useful for serialization or debugging.
  Map<String, dynamic> toMap() => {
    'tokens': tokens,
    'ids': ids,
    'type_ids': typeIds,
    'attention_mask': attentionMask,
    'special_tokens_mask': specialTokensMask,
    'offsets': offsets.map((e) => [e.$1, e.$2]).toList(),
    'word_ids': wordIds,
    'sequence_ids': sequenceIds,
  };

  @override
  String toString() =>
      'Encoding(tokens: $tokens, ids: $ids, nSequences: $nSequences)';

  /// Truncates a pair of encodings to fit within the maximum length.
  ///
  /// Returns a tuple of truncated encodings based on the specified strategy.
  ///
  /// - [encodingA]: The first encoding.
  /// - [encodingB]: The second encoding.
  /// - [maxLength]: Maximum combined length including special tokens.
  /// - [strategy]: How to distribute truncation between sequences.
  /// - [numSpecialTokens]: Number of special tokens that will be added.
  static (Encoding, Encoding) truncatePair({
    required Encoding encodingA,
    required Encoding encodingB,
    required int maxLength,
    TruncationStrategy strategy = TruncationStrategy.longestFirst,
    int numSpecialTokens = 3,
  }) {
    final availableLength = maxLength - numSpecialTokens;
    if (availableLength <= 0) {
      return (Encoding.empty(), Encoding.empty());
    }

    final totalLength = encodingA.length + encodingB.length;
    if (totalLength <= availableLength) {
      return (encodingA, encodingB);
    }

    final tokensToRemove = totalLength - availableLength;

    switch (strategy) {
      case TruncationStrategy.longestFirst:
        return _truncateLongestFirst(encodingA, encodingB, tokensToRemove);

      case TruncationStrategy.onlyFirst:
        final newLengthA = encodingA.length - tokensToRemove;
        if (newLengthA <= 0) {
          return (Encoding.empty(), encodingB);
        }
        return (encodingA.withTruncation(maxLength: newLengthA), encodingB);

      case TruncationStrategy.onlySecond:
        final newLengthB = encodingB.length - tokensToRemove;
        if (newLengthB <= 0) {
          return (encodingA, Encoding.empty());
        }
        return (encodingA, encodingB.withTruncation(maxLength: newLengthB));

      case TruncationStrategy.doNotTruncate:
        return (encodingA, encodingB);
    }
  }

  static (Encoding, Encoding) _truncateLongestFirst(
    Encoding encodingA,
    Encoding encodingB,
    int tokensToRemove,
  ) {
    var lengthA = encodingA.length;
    var lengthB = encodingB.length;

    for (var i = 0; i < tokensToRemove; i++) {
      if (lengthA > lengthB) {
        lengthA--;
      } else {
        lengthB--;
      }
    }

    final truncatedA = lengthA < encodingA.length
        ? encodingA.withTruncation(maxLength: lengthA)
        : encodingA;
    final truncatedB = lengthB < encodingB.length
        ? encodingB.withTruncation(maxLength: lengthB)
        : encodingB;

    return (truncatedA, truncatedB);
  }
}

/// A builder for constructing [Encoding] objects incrementally.
///
/// Useful for building encodings token by token during the tokenization
/// process.
///
/// Example:
/// ```dart
/// final builder = EncodingBuilder();
/// builder.addSpecialToken(token: '[CLS]', id: 101, typeId: 0);
/// builder.addToken(token: 'hello', id: 7592, typeId: 0, offset: (0, 5));
/// final encoding = builder.build();
/// ```
class EncodingBuilder {
  final List<String> _tokens = [];
  final List<int> _ids = [];
  final List<int> _typeIds = [];
  final List<int> _attentionMask = [];
  final List<int> _specialTokensMask = [];
  final List<(int, int)> _offsets = [];
  final List<int?> _wordIds = [];
  final List<int?> _sequenceIds = [];

  /// Adds a regular token to the encoding.
  ///
  /// - [token]: The token string.
  /// - [id]: The token's vocabulary ID.
  /// - [typeId]: Segment ID (0 or 1).
  /// - [offset]: Character span in original text as `(start, end)`.
  /// - [wordId]: Optional word index.
  /// - [sequenceId]: Optional sequence ID.
  void addToken({
    required String token,
    required int id,
    required int typeId,
    required (int, int) offset,
    int? wordId,
    int? sequenceId,
  }) {
    _tokens.add(token);
    _ids.add(id);
    _typeIds.add(typeId);
    _attentionMask.add(1);
    _specialTokensMask.add(0);
    _offsets.add(offset);
    _wordIds.add(wordId);
    _sequenceIds.add(sequenceId ?? typeId);
  }

  /// Adds a special token (e.g., `[CLS]`, `[SEP]`, `[PAD]`) to the encoding.
  ///
  /// Special tokens have no character offset and null word/sequence IDs.
  void addSpecialToken({
    required String token,
    required int id,
    required int typeId,
  }) {
    _tokens.add(token);
    _ids.add(id);
    _typeIds.add(typeId);
    _attentionMask.add(1);
    _specialTokensMask.add(1);
    _offsets.add((0, 0));
    _wordIds.add(null);
    _sequenceIds.add(null);
  }

  /// Builds and returns the constructed [Encoding].
  ///
  /// The builder can be reused after calling [clear].
  Encoding build() => Encoding._typed(
    tokens: List.unmodifiable(_tokens),
    ids: Int32List.fromList(_ids),
    typeIds: Uint8List.fromList(_typeIds),
    attentionMask: Uint8List.fromList(_attentionMask),
    specialTokensMask: Uint8List.fromList(_specialTokensMask),
    offsets: List.unmodifiable(_offsets),
    wordIds: List.unmodifiable(_wordIds),
    sequenceIds: List.unmodifiable(_sequenceIds),
  );

  /// Clears the builder for reuse.
  void clear() {
    _tokens.clear();
    _ids.clear();
    _typeIds.clear();
    _attentionMask.clear();
    _specialTokensMask.clear();
    _offsets.clear();
    _wordIds.clear();
    _sequenceIds.clear();
  }
}

import 'dart:async';
import 'dart:isolate';

import 'encoding.dart';
import 'pre_tokenizer.dart';
import 'vocabulary.dart';

const _kMinBatchSizeForParallel = 8;

/// Specifies the direction for padding tokens.
///
/// Used with [PaddingConfig] to control where padding tokens are added.
enum PaddingDirection {
  /// Adds padding tokens to the right (end) of the sequence.
  right,

  /// Adds padding tokens to the left (beginning) of the sequence.
  left,
}

/// Configuration for sequence padding.
///
/// Controls how sequences are padded to achieve uniform length in batch
/// processing. Padding can be applied to a fixed length or to a multiple
/// of a specified value.
///
/// Example:
/// ```dart
/// // Pad all sequences to length 128
/// tokenizer.enablePadding(length: 128);
///
/// // Pad to the nearest multiple of 8
/// tokenizer.enablePadding(padToMultipleOf: 8);
/// ```
class PaddingConfig {
  /// The direction in which padding tokens are added.
  ///
  /// Defaults to [PaddingDirection.right].
  final PaddingDirection direction;

  /// The target length to pad sequences to.
  ///
  /// If `null`, sequences are padded to the length of the longest sequence
  /// in the batch.
  final int? length;

  /// Pads sequences to a multiple of this value.
  ///
  /// Useful for hardware optimization where tensor dimensions should be
  /// multiples of certain values (e.g., 8 or 16).
  final int? padToMultipleOf;

  /// Creates a padding configuration.
  ///
  /// - [direction]: Where to add padding tokens (default: right).
  /// - [length]: Fixed target length, or `null` for dynamic padding.
  /// - [padToMultipleOf]: Pad to nearest multiple of this value.
  const PaddingConfig({
    this.direction = PaddingDirection.right,
    this.length,
    this.padToMultipleOf,
  });
}

/// Specifies the direction for sequence truncation.
///
/// Used with [TruncationConfig] to control which end of the sequence is
/// truncated when it exceeds the maximum length.
enum TruncationDirection {
  /// Truncates tokens from the right (end) of the sequence.
  right,

  /// Truncates tokens from the left (beginning) of the sequence.
  left,
}

/// Configuration for sequence truncation.
///
/// Controls how sequences are truncated when they exceed a maximum length.
/// This is essential for BERT models which have a fixed maximum sequence
/// length (typically 512 tokens).
///
/// Example:
/// ```dart
/// // Truncate sequences longer than 512 tokens
/// tokenizer.enableTruncation(maxLength: 512);
///
/// // Truncate from the left for tasks where the end is more important
/// tokenizer.enableTruncation(
///   maxLength: 128,
///   direction: TruncationDirection.left,
/// );
/// ```
class TruncationConfig {
  /// The maximum length of the encoded sequence.
  ///
  /// Sequences longer than this will be truncated.
  final int maxLength;

  /// The direction from which to truncate.
  ///
  /// Defaults to [TruncationDirection.right] (truncate from end).
  final TruncationDirection direction;

  /// The strategy for truncating text pairs.
  ///
  /// Only applicable when encoding text pairs. See [TruncationStrategy]
  /// for available options.
  final TruncationStrategy strategy;

  /// Creates a truncation configuration.
  ///
  /// - [maxLength]: Required. The maximum sequence length.
  /// - [direction]: Where to truncate from (default: right).
  /// - [strategy]: How to truncate text pairs (default: longest first).
  const TruncationConfig({
    required this.maxLength,
    this.direction = TruncationDirection.right,
    this.strategy = TruncationStrategy.longestFirst,
  });
}

/// Configuration options for the WordPiece tokenizer.
///
/// Controls the behavior of text normalization, special token handling,
/// and subword tokenization.
///
/// The default configuration matches the standard BERT uncased tokenizer:
/// - Lowercase text
/// - Strip accents
/// - Add special handling for Chinese characters
/// - Use `##` as the subword prefix
/// - Add `[CLS]` and `[SEP]` tokens automatically
///
/// Example:
/// ```dart
/// // Standard uncased configuration (default)
/// final uncasedConfig = WordPieceConfig();
///
/// // Cased configuration (preserves case)
/// final casedConfig = WordPieceConfig(
///   lowercase: false,
///   stripAccents: false,
/// );
/// ```
class WordPieceConfig {
  /// Whether to convert text to lowercase during normalization.
  ///
  /// Set to `false` for cased models like `bert-base-cased`.
  final bool lowercase;

  /// Whether to remove accents from characters during normalization.
  ///
  /// For example, `é` becomes `e`.
  final bool stripAccents;

  /// Whether to add spaces around Chinese characters.
  ///
  /// This ensures each Chinese character is treated as a separate token,
  /// which is the standard behavior for BERT models.
  final bool handleChineseChars;

  /// The prefix added to subword tokens (continuation tokens).
  ///
  /// Standard BERT uses `##` (e.g., "playing" -> ["play", "##ing"]).
  final String subwordPrefix;

  /// Maximum length of a word before it's replaced with `[UNK]`.
  ///
  /// Words longer than this are considered out-of-vocabulary.
  final int maxWordLength;

  /// Whether to automatically add `[CLS]` token at the start of encodings.
  final bool addClsToken;

  /// Whether to automatically add `[SEP]` token at the end of encodings.
  final bool addSepToken;

  /// Creates a WordPiece tokenizer configuration.
  ///
  /// All parameters have sensible defaults matching the standard BERT
  /// uncased tokenizer.
  const WordPieceConfig({
    this.lowercase = true,
    this.stripAccents = true,
    this.handleChineseChars = true,
    this.subwordPrefix = '##',
    this.maxWordLength = 200,
    this.addClsToken = true,
    this.addSepToken = true,
  });
}

/// A WordPiece tokenizer compatible with BERT and HuggingFace tokenizers.
///
/// This tokenizer implements the WordPiece algorithm used in BERT models.
/// It provides methods for encoding text to token IDs and decoding IDs back
/// to text, with support for single texts, text pairs, and batch processing.
///
/// ## Creating a Tokenizer
///
/// ```dart
/// // From a vocabulary file (async)
/// final tokenizer = await WordPieceTokenizer.fromVocabFile('vocab.txt');
///
/// // From a vocabulary file (sync)
/// final tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt');
///
/// // From a Vocabulary object
/// final tokenizer = WordPieceTokenizer(vocab: vocabulary);
/// ```
///
/// ## Encoding Text
///
/// ```dart
/// // Single text
/// final encoding = tokenizer.encode('Hello, world!');
///
/// // Text pair (for question answering, etc.)
/// final encoding = tokenizer.encodePair('What is AI?', 'AI is...');
///
/// // Batch encoding
/// final encodings = tokenizer.encodeBatch(['Text 1', 'Text 2']);
/// ```
///
/// ## Padding and Truncation
///
/// ```dart
/// tokenizer
///   .enablePadding(length: 128)
///   .enableTruncation(maxLength: 128);
/// ```
///
/// See also:
/// - [Encoding] for the result of tokenization
/// - [WordPieceConfig] for tokenizer configuration options
class WordPieceTokenizer {
  /// The vocabulary used for token lookup.
  final Vocabulary vocab;

  /// The configuration for this tokenizer.
  final WordPieceConfig config;

  late final BertPreTokenizer _preTokenizer;
  PaddingConfig? _paddingConfig;
  TruncationConfig? _truncationConfig;

  /// Creates a WordPiece tokenizer with the given vocabulary.
  ///
  /// - [vocab]: The vocabulary containing token-to-ID mappings.
  /// - [config]: Optional configuration (defaults to standard BERT uncased).
  WordPieceTokenizer({
    required this.vocab,
    this.config = const WordPieceConfig(),
  }) {
    _preTokenizer = BertPreTokenizer(
      lowercase: config.lowercase,
      stripAccents: config.stripAccents,
      handleChineseChars: config.handleChineseChars,
    );
  }

  /// Returns the current padding configuration, or `null` if disabled.
  PaddingConfig? get padding => _paddingConfig;

  /// Returns the current truncation configuration, or `null` if disabled.
  TruncationConfig? get truncation => _truncationConfig;

  /// Enables padding for encoded sequences.
  ///
  /// Returns `this` for method chaining.
  ///
  /// - [direction]: Where to add padding (default: right).
  /// - [length]: Fixed target length, or `null` to pad to longest in batch.
  /// - [padToMultipleOf]: Pad to nearest multiple of this value.
  ///
  /// Example:
  /// ```dart
  /// tokenizer.enablePadding(length: 128, direction: PaddingDirection.right);
  /// ```
  WordPieceTokenizer enablePadding({
    PaddingDirection direction = PaddingDirection.right,
    int? length,
    int? padToMultipleOf,
  }) {
    _paddingConfig = PaddingConfig(
      direction: direction,
      length: length,
      padToMultipleOf: padToMultipleOf,
    );
    return this;
  }

  /// Disables padding.
  ///
  /// Returns `this` for method chaining.
  WordPieceTokenizer noPadding() {
    _paddingConfig = null;
    return this;
  }

  /// Enables truncation for encoded sequences.
  ///
  /// Returns `this` for method chaining.
  ///
  /// - [maxLength]: Required. Maximum sequence length (including special tokens).
  /// - [direction]: Where to truncate from (default: right).
  /// - [strategy]: How to truncate text pairs (default: longest first).
  ///
  /// Example:
  /// ```dart
  /// tokenizer.enableTruncation(maxLength: 512);
  /// ```
  WordPieceTokenizer enableTruncation({
    required int maxLength,
    TruncationDirection direction = TruncationDirection.right,
    TruncationStrategy strategy = TruncationStrategy.longestFirst,
  }) {
    _truncationConfig = TruncationConfig(
      maxLength: maxLength,
      direction: direction,
      strategy: strategy,
    );
    return this;
  }

  /// Disables truncation.
  ///
  /// Returns `this` for method chaining.
  WordPieceTokenizer noTruncation() {
    _truncationConfig = null;
    return this;
  }

  Encoding _applyPostProcessing(Encoding encoding) {
    var result = encoding;

    if (_truncationConfig != null) {
      result = result.withTruncation(
        maxLength: _truncationConfig!.maxLength,
        truncateFromEnd:
            _truncationConfig!.direction == TruncationDirection.right,
      );
    }

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      if (_paddingConfig!.length != null) {
        result = result.withPadding(
          targetLength: _paddingConfig!.length!,
          padTokenId: vocab.padTokenId,
          padOnRight: padOnRight,
        );
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        result = result.withPaddingToMultipleOf(
          multiple: _paddingConfig!.padToMultipleOf!,
          padTokenId: vocab.padTokenId,
          padOnRight: padOnRight,
        );
      }
    }

    return result;
  }

  List<Encoding> _applyBatchPostProcessing(List<Encoding> encodings) {
    if (encodings.isEmpty) return encodings;

    var results = encodings;
    if (_truncationConfig != null) {
      results = results
          .map(
            (e) => e.withTruncation(
              maxLength: _truncationConfig!.maxLength,
              truncateFromEnd:
                  _truncationConfig!.direction == TruncationDirection.right,
            ),
          )
          .toList();
    }

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      int targetLength;
      if (_paddingConfig!.length != null) {
        targetLength = _paddingConfig!.length!;
      } else {
        targetLength = results
            .map((e) => e.length)
            .reduce((a, b) => a > b ? a : b);
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        final multiple = _paddingConfig!.padToMultipleOf!;
        final remainder = targetLength % multiple;
        if (remainder != 0) {
          targetLength += multiple - remainder;
        }
      }

      results = results
          .map(
            (e) => e.withPadding(
              targetLength: targetLength,
              padTokenId: vocab.padTokenId,
              padOnRight: padOnRight,
            ),
          )
          .toList();
    }

    return results;
  }

  /// Creates a tokenizer from a vocabulary file asynchronously.
  ///
  /// The vocabulary file should contain one token per line, with the line
  /// number (0-indexed) being the token's ID.
  ///
  /// - [path]: Path to the vocabulary file.
  /// - [config]: Optional tokenizer configuration.
  ///
  /// Example:
  /// ```dart
  /// final tokenizer = await WordPieceTokenizer.fromVocabFile(
  ///   'assets/vocab.txt',
  ///   config: WordPieceConfig(lowercase: false),
  /// );
  /// ```
  static Future<WordPieceTokenizer> fromVocabFile(
    String path, {
    WordPieceConfig config = const WordPieceConfig(),
  }) async {
    final vocab = await Vocabulary.fromFile(
      path,
      subwordPrefix: config.subwordPrefix,
    );
    return WordPieceTokenizer(vocab: vocab, config: config);
  }

  /// Creates a tokenizer from a vocabulary file synchronously.
  ///
  /// See [fromVocabFile] for details.
  static WordPieceTokenizer fromVocabFileSync(
    String path, {
    WordPieceConfig config = const WordPieceConfig(),
  }) {
    final vocab = Vocabulary.fromFileSync(
      path,
      subwordPrefix: config.subwordPrefix,
    );
    return WordPieceTokenizer(vocab: vocab, config: config);
  }

  /// Returns the number of special tokens that will be added during encoding.
  ///
  /// - [isPair]: Whether encoding a text pair (adds extra `[SEP]` token).
  int numSpecialTokensToAdd({bool isPair = false}) {
    var count = 0;
    if (config.addClsToken) count++;
    if (config.addSepToken) count++;
    if (isPair && config.addSepToken) count++;
    return count;
  }

  /// Encodes a single text string into an [Encoding].
  ///
  /// The text is normalized, tokenized using WordPiece, and special tokens
  /// are added based on the configuration.
  ///
  /// - [text]: The text to encode.
  /// - [addSpecialTokens]: Override whether to add `[CLS]`/`[SEP]` tokens.
  ///
  /// Returns an [Encoding] containing token IDs, attention mask, and other
  /// information needed for model input.
  ///
  /// Example:
  /// ```dart
  /// final encoding = tokenizer.encode('Hello, world!');
  /// print(encoding.tokens); // ['[CLS]', 'hello', ',', 'world', '!', '[SEP]']
  /// print(encoding.ids);    // [101, 7592, 1010, 2088, 999, 102]
  /// ```
  Encoding encode(String text, {bool? addSpecialTokens}) {
    final shouldAddCls = addSpecialTokens ?? config.addClsToken;
    final shouldAddSep = addSpecialTokens ?? config.addSepToken;

    final builder = EncodingBuilder();

    if (shouldAddCls) {
      builder.addSpecialToken(
        token: SpecialTokens.cls,
        id: vocab.clsTokenId,
        typeId: 0,
      );
    }

    final preTokens = _preTokenizer.preTokenize(text);

    for (var wordIdx = 0; wordIdx < preTokens.length; wordIdx++) {
      final preToken = preTokens[wordIdx];
      final wordTokens = _tokenizeWord(preToken.text);

      for (final tokenInfo in wordTokens) {
        builder.addToken(
          token: tokenInfo.token,
          id: tokenInfo.id,
          typeId: 0,
          offset: (
            preToken.start + tokenInfo.startOffset,
            preToken.start + tokenInfo.endOffset,
          ),
          wordId: wordIdx,
        );
      }
    }

    if (shouldAddSep) {
      builder.addSpecialToken(
        token: SpecialTokens.sep,
        id: vocab.sepTokenId,
        typeId: 0,
      );
    }

    return _applyPostProcessing(builder.build());
  }

  /// Encodes a pair of text strings into a single [Encoding].
  ///
  /// Used for tasks like question answering, sentence similarity, or
  /// natural language inference where two text sequences are needed.
  ///
  /// The output format is: `[CLS] textA [SEP] textB [SEP]`
  ///
  /// - [textA]: The first text (e.g., question, premise).
  /// - [textB]: The second text (e.g., context, hypothesis).
  /// - [addSpecialTokens]: Override whether to add special tokens.
  /// - [maxLength]: Maximum total length (overrides [enableTruncation]).
  /// - [truncationStrategy]: How to truncate if sequences are too long.
  ///
  /// Example:
  /// ```dart
  /// final encoding = tokenizer.encodePair(
  ///   'What is the capital of France?',
  ///   'Paris is the capital of France.',
  /// );
  /// print(encoding.typeIds); // [0, 0, ..., 0, 1, 1, ..., 1]
  /// ```
  Encoding encodePair(
    String textA,
    String textB, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy truncationStrategy = TruncationStrategy.longestFirst,
  }) {
    final shouldAddCls = addSpecialTokens ?? config.addClsToken;
    final shouldAddSep = addSpecialTokens ?? config.addSepToken;

    final encodingA = encode(textA, addSpecialTokens: false);
    final encodingB = encode(textB, addSpecialTokens: false);

    final effectiveMaxLength = maxLength ?? _truncationConfig?.maxLength;
    final effectiveStrategy = _truncationConfig?.strategy ?? truncationStrategy;

    final (truncatedA, truncatedB) = effectiveMaxLength != null
        ? Encoding.truncatePair(
            encodingA: encodingA,
            encodingB: encodingB,
            maxLength: effectiveMaxLength,
            strategy: effectiveStrategy,
            numSpecialTokens: numSpecialTokensToAdd(isPair: true),
          )
        : (encodingA, encodingB);

    final builder = EncodingBuilder();

    if (shouldAddCls) {
      builder.addSpecialToken(
        token: SpecialTokens.cls,
        id: vocab.clsTokenId,
        typeId: 0,
      );
    }

    for (var i = 0; i < truncatedA.length; i++) {
      builder.addToken(
        token: truncatedA.tokens[i],
        id: truncatedA.ids[i],
        typeId: 0,
        offset: truncatedA.offsets[i],
        wordId: truncatedA.wordIds[i],
      );
    }

    if (shouldAddSep) {
      builder.addSpecialToken(
        token: SpecialTokens.sep,
        id: vocab.sepTokenId,
        typeId: 0,
      );
    }

    final wordIdOffset = truncatedA.wordIds.where((id) => id != null).length;
    for (var i = 0; i < truncatedB.length; i++) {
      final originalWordId = truncatedB.wordIds[i];
      builder.addToken(
        token: truncatedB.tokens[i],
        id: truncatedB.ids[i],
        typeId: 1,
        offset: truncatedB.offsets[i],
        wordId: originalWordId != null ? wordIdOffset + originalWordId : null,
      );
    }

    if (shouldAddSep) {
      builder.addSpecialToken(
        token: SpecialTokens.sep,
        id: vocab.sepTokenId,
        typeId: 1,
      );
    }

    var result = builder.build();
    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      if (_paddingConfig!.length != null) {
        result = result.withPadding(
          targetLength: _paddingConfig!.length!,
          padTokenId: vocab.padTokenId,
          padOnRight: padOnRight,
        );
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        result = result.withPaddingToMultipleOf(
          multiple: _paddingConfig!.padToMultipleOf!,
          padTokenId: vocab.padTokenId,
          padOnRight: padOnRight,
        );
      }
    }

    return result;
  }

  /// Encodes multiple texts in a batch.
  ///
  /// This method applies consistent padding across all encodings in the batch.
  /// When padding is enabled, all sequences are padded to the same length.
  ///
  /// - [texts]: List of texts to encode.
  /// - [addSpecialTokens]: Override whether to add special tokens.
  ///
  /// Returns a list of [Encoding] objects with consistent lengths (if padding
  /// is enabled).
  List<Encoding> encodeBatch(List<String> texts, {bool? addSpecialTokens}) {
    final savedPadding = _paddingConfig;
    final savedTruncation = _truncationConfig;
    _paddingConfig = null;
    _truncationConfig = null;

    final encodings = texts
        .map((text) => encode(text, addSpecialTokens: addSpecialTokens))
        .toList();

    _paddingConfig = savedPadding;
    _truncationConfig = savedTruncation;

    return _applyBatchPostProcessing(encodings);
  }

  /// Encodes multiple texts in parallel using isolates.
  ///
  /// For large batches, this method distributes work across multiple isolates
  /// for improved performance. Falls back to [encodeBatch] for small batches.
  ///
  /// - [texts]: List of texts to encode.
  /// - [addSpecialTokens]: Override whether to add special tokens.
  /// - [numWorkers]: Number of isolates to use (auto-calculated if not set).
  ///
  /// Example:
  /// ```dart
  /// final encodings = await tokenizer.encodeBatchParallel(
  ///   largeTextList,
  ///   numWorkers: 4,
  /// );
  /// ```
  Future<List<Encoding>> encodeBatchParallel(
    List<String> texts, {
    bool? addSpecialTokens,
    int? numWorkers,
  }) async {
    if (texts.length < _kMinBatchSizeForParallel) {
      return encodeBatch(texts, addSpecialTokens: addSpecialTokens);
    }

    final workerCount = numWorkers ?? _getOptimalWorkerCount(texts.length);
    final chunkSize = (texts.length / workerCount).ceil();

    final vocabTokens = vocab.tokens;
    final futures = <Future<List<_EncodingData>>>[];

    for (var i = 0; i < workerCount; i++) {
      final start = i * chunkSize;
      if (start >= texts.length) break;

      final end = (start + chunkSize).clamp(0, texts.length);
      final chunk = texts.sublist(start, end);

      futures.add(
        Isolate.run(
          () => _encodeChunkInIsolate(
            chunk,
            vocabTokens,
            config,
            addSpecialTokens,
          ),
        ),
      );
    }

    final results = await Future.wait(futures);

    final encodings = <Encoding>[];
    for (final chunkResults in results) {
      for (final data in chunkResults) {
        encodings.add(data.toEncoding());
      }
    }

    return _applyBatchPostProcessing(encodings);
  }

  Future<List<Encoding>> encodePairBatchParallel(
    List<(String, String)> pairs, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy truncationStrategy = TruncationStrategy.longestFirst,
    int? numWorkers,
  }) async {
    if (pairs.length < _kMinBatchSizeForParallel) {
      return encodePairBatch(
        pairs,
        addSpecialTokens: addSpecialTokens,
        maxLength: maxLength,
        truncationStrategy: truncationStrategy,
      );
    }

    final workerCount = numWorkers ?? _getOptimalWorkerCount(pairs.length);
    final chunkSize = (pairs.length / workerCount).ceil();

    final vocabTokens = vocab.tokens;
    final effectiveMaxLength = maxLength ?? _truncationConfig?.maxLength;
    final futures = <Future<List<_EncodingData>>>[];

    for (var i = 0; i < workerCount; i++) {
      final start = i * chunkSize;
      if (start >= pairs.length) break;

      final end = (start + chunkSize).clamp(0, pairs.length);
      final chunk = pairs.sublist(start, end);
      final chunkData = chunk.map((p) => [p.$1, p.$2]).toList();
      futures.add(
        Isolate.run(
          () => _encodePairChunkInIsolate(
            chunkData,
            vocabTokens,
            config,
            addSpecialTokens,
            effectiveMaxLength,
            truncationStrategy,
          ),
        ),
      );
    }

    final results = await Future.wait(futures);
    final encodings = <Encoding>[];
    for (final chunkResults in results) {
      for (final data in chunkResults) {
        encodings.add(data.toEncoding());
      }
    }

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      int targetLength;
      if (_paddingConfig!.length != null) {
        targetLength = _paddingConfig!.length!;
      } else {
        targetLength = encodings
            .map((e) => e.length)
            .reduce((a, b) => a > b ? a : b);
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        final multiple = _paddingConfig!.padToMultipleOf!;
        final remainder = targetLength % multiple;
        if (remainder != 0) {
          targetLength += multiple - remainder;
        }
      }

      return encodings
          .map(
            (e) => e.withPadding(
              targetLength: targetLength,
              padTokenId: vocab.padTokenId,
              padOnRight: padOnRight,
            ),
          )
          .toList();
    }

    return encodings;
  }

  int _getOptimalWorkerCount(int batchSize) {
    const maxWorkers = 4;
    const minItemsPerWorker = 4;

    final workersByItems = (batchSize / minItemsPerWorker).floor();
    return workersByItems.clamp(1, maxWorkers);
  }

  /// Encodes multiple text pairs in a batch.
  ///
  /// - [pairs]: List of text pairs as records `(String, String)`.
  /// - [addSpecialTokens]: Override whether to add special tokens.
  /// - [maxLength]: Maximum total length per encoding.
  /// - [truncationStrategy]: How to truncate long pairs.
  List<Encoding> encodePairBatch(
    List<(String, String)> pairs, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy truncationStrategy = TruncationStrategy.longestFirst,
  }) {
    final savedPadding = _paddingConfig;
    _paddingConfig = null;

    final encodings = pairs
        .map(
          (pair) => encodePair(
            pair.$1,
            pair.$2,
            addSpecialTokens: addSpecialTokens,
            maxLength: maxLength,
            truncationStrategy: truncationStrategy,
          ),
        )
        .toList();

    _paddingConfig = savedPadding;

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      int targetLength;
      if (_paddingConfig!.length != null) {
        targetLength = _paddingConfig!.length!;
      } else {
        targetLength = encodings
            .map((e) => e.length)
            .reduce((a, b) => a > b ? a : b);
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        final multiple = _paddingConfig!.padToMultipleOf!;
        final remainder = targetLength % multiple;
        if (remainder != 0) {
          targetLength += multiple - remainder;
        }
      }

      return encodings
          .map(
            (e) => e.withPadding(
              targetLength: targetLength,
              padTokenId: vocab.padTokenId,
              padOnRight: padOnRight,
            ),
          )
          .toList();
    }

    return encodings;
  }

  List<_TokenInfo> _tokenizeWord(String word) {
    if (word.isEmpty) {
      return [];
    }

    if (word.length > config.maxWordLength) {
      return [
        _TokenInfo(
          token: SpecialTokens.unk,
          id: vocab.unkTokenId,
          startOffset: 0,
          endOffset: word.length,
        ),
      ];
    }

    final tokens = <_TokenInfo>[];
    var start = 0;
    var isFirstSubword = true;

    while (start < word.length) {
      final match = _findLongestMatchAt(
        word,
        start,
        isSubword: !isFirstSubword,
      );

      if (match == null) {
        return [
          _TokenInfo(
            token: SpecialTokens.unk,
            id: vocab.unkTokenId,
            startOffset: 0,
            endOffset: word.length,
          ),
        ];
      }

      final matchedText = word.substring(start, match.endIndex);
      final tokenStr = isFirstSubword
          ? matchedText
          : '${config.subwordPrefix}$matchedText';

      tokens.add(
        _TokenInfo(
          token: tokenStr,
          id: match.tokenId,
          startOffset: start,
          endOffset: match.endIndex,
        ),
      );

      start = match.endIndex;
      isFirstSubword = false;
    }

    return tokens;
  }

  _TrieMatchResult? _findLongestMatchAt(
    String word,
    int startIndex, {
    required bool isSubword,
  }) {
    final trie = isSubword ? vocab.subwordTrie : vocab.trie;
    final match = trie.findLongestPrefix(word, startIndex);
    if (match != null) {
      return _TrieMatchResult(endIndex: match.end, tokenId: match.tokenId);
    }

    if (startIndex < word.length) {
      final charCode = word.codeUnitAt(startIndex);
      final firstChar = String.fromCharCode(charCode);
      final tokenToCheck = isSubword
          ? '${config.subwordPrefix}$firstChar'
          : firstChar;

      if (vocab.contains(tokenToCheck)) {
        return _TrieMatchResult(
          endIndex: startIndex + 1,
          tokenId: vocab.tokenToId(tokenToCheck),
        );
      }
    }

    return null;
  }

  /// Decodes a list of token IDs back to a text string.
  ///
  /// Reconstructs the original text by joining tokens and removing
  /// subword prefixes.
  ///
  /// - [ids]: List of token IDs to decode.
  /// - [skipSpecialTokens]: Whether to exclude special tokens like `[CLS]`,
  ///   `[SEP]`, `[PAD]` from the output (default: `true`).
  ///
  /// Example:
  /// ```dart
  /// final text = tokenizer.decode([101, 7592, 1010, 2088, 999, 102]);
  /// print(text); // 'hello , world !'
  /// ```
  String decode(List<int> ids, {bool skipSpecialTokens = true}) {
    final buffer = StringBuffer();
    var isFirst = true;

    for (final id in ids) {
      final token = vocab.idToToken(id);

      if (skipSpecialTokens && vocab.isSpecialToken(token)) {
        continue;
      }

      if (token.startsWith(config.subwordPrefix)) {
        buffer.write(token.substring(config.subwordPrefix.length));
      } else {
        if (!isFirst) {
          buffer.write(' ');
        }
        buffer.write(token);
        isFirst = false;
      }
    }

    return buffer.toString();
  }

  /// Decodes multiple sequences of token IDs in a batch.
  ///
  /// - [idsBatch]: List of token ID lists to decode.
  /// - [skipSpecialTokens]: Whether to exclude special tokens.
  List<String> decodeBatch(
    List<List<int>> idsBatch, {
    bool skipSpecialTokens = true,
  }) {
    return idsBatch
        .map((ids) => decode(ids, skipSpecialTokens: skipSpecialTokens))
        .toList();
  }

  /// Converts a list of token strings to their corresponding IDs.
  ///
  /// Unknown tokens are mapped to the `[UNK]` token ID.
  List<int> convertTokensToIds(List<String> tokens) {
    return tokens.map(vocab.tokenToId).toList();
  }

  /// Converts a list of token IDs to their corresponding token strings.
  ///
  /// Invalid IDs are mapped to the `[UNK]` token.
  List<String> convertIdsToTokens(List<int> ids) {
    return ids.map(vocab.idToToken).toList();
  }
}

class _TokenInfo {
  final String token;
  final int id;
  final int startOffset;
  final int endOffset;

  const _TokenInfo({
    required this.token,
    required this.id,
    required this.startOffset,
    required this.endOffset,
  });
}

class _TrieMatchResult {
  final int endIndex;
  final int tokenId;

  const _TrieMatchResult({required this.endIndex, required this.tokenId});
}

class _EncodingData {
  final List<String> tokens;
  final List<int> ids;
  final List<int> typeIds;
  final List<int> attentionMask;
  final List<int> specialTokensMask;
  final List<List<int>> offsets;
  final List<int?> wordIds;
  final List<int?> sequenceIds;

  const _EncodingData({
    required this.tokens,
    required this.ids,
    required this.typeIds,
    required this.attentionMask,
    required this.specialTokensMask,
    required this.offsets,
    required this.wordIds,
    required this.sequenceIds,
  });

  factory _EncodingData.fromEncoding(Encoding encoding) {
    return _EncodingData(
      tokens: encoding.tokens.toList(),
      ids: encoding.ids.toList(),
      typeIds: encoding.typeIds.toList(),
      attentionMask: encoding.attentionMask.toList(),
      specialTokensMask: encoding.specialTokensMask.toList(),
      offsets: encoding.offsets.map((o) => [o.$1, o.$2]).toList(),
      wordIds: encoding.wordIds.toList(),
      sequenceIds: encoding.sequenceIds.toList(),
    );
  }

  Encoding toEncoding() {
    return Encoding(
      tokens: tokens,
      ids: ids,
      typeIds: typeIds,
      attentionMask: attentionMask,
      specialTokensMask: specialTokensMask,
      offsets: offsets.map((o) => (o[0], o[1])).toList(),
      wordIds: wordIds,
      sequenceIds: sequenceIds,
    );
  }
}

List<_EncodingData> _encodeChunkInIsolate(
  List<String> texts,
  List<String> vocabTokens,
  WordPieceConfig config,
  bool? addSpecialTokens,
) {
  final vocab = Vocabulary.fromTokens(
    vocabTokens,
    subwordPrefix: config.subwordPrefix,
  );
  final tokenizer = WordPieceTokenizer(vocab: vocab, config: config);
  final results = <_EncodingData>[];
  for (final text in texts) {
    final encoding = tokenizer.encode(text, addSpecialTokens: addSpecialTokens);
    results.add(_EncodingData.fromEncoding(encoding));
  }

  return results;
}

List<_EncodingData> _encodePairChunkInIsolate(
  List<List<String>> pairs,
  List<String> vocabTokens,
  WordPieceConfig config,
  bool? addSpecialTokens,
  int? maxLength,
  TruncationStrategy truncationStrategy,
) {
  final vocab = Vocabulary.fromTokens(
    vocabTokens,
    subwordPrefix: config.subwordPrefix,
  );
  final tokenizer = WordPieceTokenizer(vocab: vocab, config: config);
  final results = <_EncodingData>[];
  for (final pair in pairs) {
    final encoding = tokenizer.encodePair(
      pair[0],
      pair[1],
      addSpecialTokens: addSpecialTokens,
      maxLength: maxLength,
      truncationStrategy: truncationStrategy,
    );
    results.add(_EncodingData.fromEncoding(encoding));
  }

  return results;
}

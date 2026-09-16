import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'encoding.dart';
import 'pre_tokenizer.dart';
import 'tokenizer_json_parser.dart';
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
  int _getOptimalWorkerCount(int count) => (count ~/ 4).clamp(1, 4);

  /// The vocabulary used for token lookup.
  final Vocabulary vocab;

  /// The configuration for this tokenizer.
  final WordPieceConfig config;

  late BertPreTokenizer _preTokenizer;
  Map<String, dynamic>? _json;
  bool _overrideTemplate = false;
  String get _unknownToken =>
      (_json?['model'] as Map?)?['unk_token'] as String? ?? SpecialTokens.unk;
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
    if (length != null && length < 0) {
      throw ArgumentError.value(length, 'length');
    }
    if (padToMultipleOf != null && padToMultipleOf <= 0) {
      throw ArgumentError.value(padToMultipleOf, 'padToMultipleOf');
    }
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
    if (maxLength < 0) throw ArgumentError.value(maxLength, 'maxLength');
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

    if (_paddingConfig != null) {
      final padOnRight = _paddingConfig!.direction == PaddingDirection.right;

      if (_paddingConfig!.length != null) {
        result = result.withPadding(
          targetLength: _paddingConfig!.length!,
          padTokenId:
              (_json?['padding'] as Map?)?['pad_id'] as int? ??
              vocab.padTokenId,
          padToken:
              (_json?['padding'] as Map?)?['pad_token'] as String? ??
              SpecialTokens.pad,
          padTypeId: (_json?['padding'] as Map?)?['pad_type_id'] as int? ?? 0,
          padOnRight: padOnRight,
        );
      }

      if (_paddingConfig!.padToMultipleOf != null) {
        result = result.withPaddingToMultipleOf(
          multiple: _paddingConfig!.padToMultipleOf!,
          padTokenId:
              (_json?['padding'] as Map?)?['pad_id'] as int? ??
              vocab.padTokenId,
          padToken:
              (_json?['padding'] as Map?)?['pad_token'] as String? ??
              SpecialTokens.pad,
          padTypeId: (_json?['padding'] as Map?)?['pad_type_id'] as int? ?? 0,
          padOnRight: padOnRight,
        );
      }
    }

    return result;
  }

  List<Encoding> _applyBatchPostProcessing(List<Encoding> encodings) {
    if (encodings.isEmpty) return encodings;

    var results = encodings;
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
              padTokenId:
                  (_json?['padding'] as Map?)?['pad_id'] as int? ??
                  vocab.padTokenId,
              padToken:
                  (_json?['padding'] as Map?)?['pad_token'] as String? ??
                  SpecialTokens.pad,
              padTypeId:
                  (_json?['padding'] as Map?)?['pad_type_id'] as int? ?? 0,
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

  /// Creates a tokenizer from a HuggingFace `tokenizer.json` file
  /// asynchronously.
  ///
  /// The JSON file must contain a WordPiece model. Normalizer settings,
  /// post-processor configuration, and added tokens are automatically
  /// extracted from the JSON.
  ///
  /// - [path]: Path to the `tokenizer.json` file.
  /// - [configOverride]: Replaces the exposed WordPiece settings as a whole,
  ///   including defaults for omitted fields, and selects the legacy CLS/SEP
  ///   template. This is not a partial merge.
  ///
  /// Example:
  /// ```dart
  /// final tokenizer = await WordPieceTokenizer.fromTokenizerJson(
  ///   'assets/tokenizer.json',
  /// );
  /// ```
  static Future<WordPieceTokenizer> fromTokenizerJson(
    String path, {
    WordPieceConfig? configOverride,
  }) async {
    final content = await File(path).readAsString();
    return _fromParsedTokenizerJson(
      _decodeJson(content),
      configOverride: configOverride,
    );
  }

  /// Creates a tokenizer from a HuggingFace `tokenizer.json` file
  /// synchronously.
  ///
  /// See [fromTokenizerJson] for details.
  static WordPieceTokenizer fromTokenizerJsonSync(
    String path, {
    WordPieceConfig? configOverride,
  }) {
    final content = File(path).readAsStringSync();
    return _fromParsedTokenizerJson(
      _decodeJson(content),
      configOverride: configOverride,
    );
  }

  /// Creates a tokenizer from a JSON string in the HuggingFace
  /// `tokenizer.json` format.
  ///
  /// See [fromTokenizerJson] for details.
  static WordPieceTokenizer fromTokenizerJsonString(
    String jsonString, {
    WordPieceConfig? configOverride,
  }) {
    return _fromParsedTokenizerJson(
      _decodeJson(jsonString),
      configOverride: configOverride,
    );
  }

  static Map<String, dynamic> _decodeJson(String content) {
    final value = jsonDecode(content);
    if (value is! Map<String, dynamic>) {
      throw const FormatException('tokenizer.json must be an object');
    }
    return value;
  }

  static WordPieceTokenizer _fromParsedTokenizerJson(
    Map<String, dynamic> json, {
    WordPieceConfig? configOverride,
  }) {
    final parsed = parseTokenizerJson(json);

    final config =
        configOverride ??
        WordPieceConfig(
          lowercase: parsed.lowercase,
          stripAccents: parsed.stripAccents,
          handleChineseChars: parsed.handleChineseChars,
          subwordPrefix: parsed.subwordPrefix,
          maxWordLength: parsed.maxWordLength,
          addClsToken: parsed.addClsToken,
          addSepToken: parsed.addSepToken,
        );

    final vocab = Vocabulary.fromMap(
      parsed.vocab,
      subwordPrefix: config.subwordPrefix,
    );

    final tokenizer = WordPieceTokenizer(vocab: vocab, config: config)
      .._json = json
      .._overrideTemplate = configOverride != null;
    final normalizer = json['normalizer'] as Map<String, dynamic>?;
    tokenizer._preTokenizer = BertPreTokenizer(
      lowercase: config.lowercase,
      stripAccents: config.stripAccents,
      handleChineseChars: config.handleChineseChars,
      cleanText: normalizer?['clean_text'] as bool? ?? (normalizer != null),
      split: json['pre_tokenizer'] != null,
    );
    final padding = json['padding'] as Map<String, dynamic>?;
    if (padding != null) {
      final strategy = padding['strategy'];
      tokenizer.enablePadding(
        length: strategy is Map ? strategy['Fixed'] as int? : null,
        direction: padding['direction'] == 'Left'
            ? PaddingDirection.left
            : PaddingDirection.right,
        padToMultipleOf: padding['pad_to_multiple_of'] as int?,
      );
    }
    final truncation = json['truncation'] as Map<String, dynamic>?;
    if (truncation != null) {
      tokenizer.enableTruncation(
        maxLength: truncation['max_length'] as int,
        direction: truncation['direction'] == 'Left'
            ? TruncationDirection.left
            : TruncationDirection.right,
        strategy: switch (truncation['strategy']) {
          'OnlyFirst' => TruncationStrategy.onlyFirst,
          'OnlySecond' => TruncationStrategy.onlySecond,
          _ => TruncationStrategy.longestFirst,
        },
      );
    }
    return tokenizer;
  }

  /// Returns the number of special tokens that will be added during encoding.
  ///
  /// - [isPair]: Whether encoding a text pair (adds extra `[SEP]` token).
  int numSpecialTokensToAdd({bool isPair = false}) {
    if (_json != null && !_overrideTemplate) {
      final processor = _json!['post_processor'] as Map<String, dynamic>?;
      if (processor == null) return 0;
      if (processor['type'] == 'BertProcessing') return isPair ? 3 : 2;
      final template = processor[isPair ? 'pair' : 'single'] as List<dynamic>?;
      if (template == null) {
        throw const FormatException('Missing post-processor template');
      }
      return template.where((e) => e['SpecialToken'] != null).length;
    }
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
  Encoding encode(String text, {bool? addSpecialTokens}) =>
      _encode(text, addSpecialTokens: addSpecialTokens);

  Encoding _content(String text) {
    final builder = EncodingBuilder();
    var wordId = 0;
    var position = 0;
    final added =
        (_json?['added_tokens'] as List<dynamic>? ??
                SpecialTokens.defaults
                    .where(vocab.contains)
                    .map(
                      (token) => <String, dynamic>{
                        'content': token,
                        'id': vocab.tokenToId(token),
                      },
                    )
                    .toList())
            .cast<Map<String, dynamic>>();
    final contents = added.map((e) => e['content'] as String).toList()
      ..sort((a, b) => b.length.compareTo(a.length));
    final pattern = contents.isEmpty
        ? null
        : RegExp(contents.map(RegExp.escape).join('|'));
    void appendText(String part) {
      for (final preToken in _preTokenizer.preTokenize(part)) {
        for (final token in _tokenizeWord(preToken.text)) {
          final span = preToken.originalSpan(
            token.startOffset,
            token.endOffset,
          );
          builder.addToken(
            token: token.token,
            id: token.id,
            typeId: 0,
            offset: (position + span.$1, position + span.$2),
            wordId: wordId,
          );
        }
        wordId++;
      }
      position += part.runes.length;
    }

    var start = 0;
    for (final match in pattern?.allMatches(text) ?? const <RegExpMatch>[]) {
      appendText(text.substring(start, match.start));
      final token = match.group(0)!;
      final end = position + token.runes.length;
      builder.addToken(
        token: token,
        id: vocab.tokenToId(token),
        typeId: 0,
        offset: (position, end),
        wordId: wordId++,
      );
      position = end;
      start = match.end;
    }
    appendText(text.substring(start));
    return builder.build();
  }

  Encoding _encode(
    String text, {
    String? pair,
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy? truncationStrategy,
  }) {
    var first = _content(text);
    var second = pair == null ? null : _content(pair);
    final addCls = addSpecialTokens ?? config.addClsToken;
    final addSep = addSpecialTokens ?? config.addSepToken;
    var reserved = (addCls ? 1 : 0) + (addSep ? (pair == null ? 1 : 2) : 0);
    final processor = _json?['post_processor'] as Map<String, dynamic>?;
    List<dynamic>? template;
    if (_json != null && !_overrideTemplate) {
      if (processor == null) {
        template = [
          {
            'Sequence': {'id': 'A', 'type_id': 0},
          },
          if (pair != null)
            {
              'Sequence': {'id': 'B', 'type_id': 1},
            },
        ];
      } else if (processor['type'] == 'TemplateProcessing') {
        template =
            processor[pair == null ? 'single' : 'pair'] as List<dynamic>?;
        if (template == null) {
          throw const FormatException('Missing post-processor template');
        }
      } else if (processor['type'] == 'BertProcessing') {
        template = [
          {
            'SpecialToken': {'id': processor['cls'][0], 'type_id': 0},
          },
          {
            'Sequence': {'id': 'A', 'type_id': 0},
          },
          {
            'SpecialToken': {'id': processor['sep'][0], 'type_id': 0},
          },
          if (pair != null) ...[
            {
              'Sequence': {'id': 'B', 'type_id': 1},
            },
            {
              'SpecialToken': {'id': processor['sep'][0], 'type_id': 1},
            },
          ],
        ];
      }
      if (template != null) {
        if (addSpecialTokens == false) {
          template = template.where((e) => e['Sequence'] != null).toList();
        }
        reserved = template.where((e) => e['SpecialToken'] != null).length;
      }
    }
    final limit = maxLength ?? _truncationConfig?.maxLength;
    if (limit != null) {
      if (limit < reserved) {
        throw ArgumentError.value(
          limit,
          'maxLength',
          'Too small for special tokens',
        );
      }
      final right = _truncationConfig?.direction != TruncationDirection.left;
      if (second == null) {
        first = first.withTruncation(
          maxLength: limit - reserved,
          truncateFromEnd: right,
        );
      } else {
        final truncated = Encoding.truncatePair(
          encodingA: first,
          encodingB: second,
          maxLength: limit,
          numSpecialTokens: reserved,
          truncateFromEnd: right,
          strategy:
              truncationStrategy ??
              _truncationConfig?.strategy ??
              TruncationStrategy.longestFirst,
        );
        first = truncated.$1;
        second = truncated.$2;
      }
    }
    final builder = EncodingBuilder();
    void append(Encoding encoding, int sequence, [int? typeId]) {
      for (var i = 0; i < encoding.length; i++) {
        builder.addToken(
          token: encoding.tokens[i],
          id: encoding.ids[i],
          typeId: typeId ?? sequence,
          offset: encoding.offsets[i],
          wordId: encoding.wordIds[i],
          sequenceId: sequence,
        );
      }
    }

    if (template != null) {
      for (final entry in template) {
        final sequence = entry['Sequence'];
        final special = entry['SpecialToken'];
        if (sequence != null) {
          final index = sequence['id'] == 'A' ? 0 : 1;
          append(
            index == 0 ? first : second!,
            index,
            sequence['type_id'] as int,
          );
        } else if (special != null) {
          final token = special['id'] as String;
          builder.addSpecialToken(
            token: token,
            id: vocab.tokenToId(token),
            typeId: special['type_id'] as int,
          );
        }
      }
      return _applyPostProcessing(builder.build());
    }
    if (addCls) {
      builder.addSpecialToken(
        token: SpecialTokens.cls,
        id: vocab.clsTokenId,
        typeId: 0,
      );
    }
    append(first, 0);
    if (addSep) {
      builder.addSpecialToken(
        token: SpecialTokens.sep,
        id: vocab.sepTokenId,
        typeId: 0,
      );
    }
    if (second != null) {
      append(second, 1);
      if (addSep) {
        builder.addSpecialToken(
          token: SpecialTokens.sep,
          id: vocab.sepTokenId,
          typeId: 1,
        );
      }
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
  }) => _encode(
    textA,
    pair: textB,
    addSpecialTokens: addSpecialTokens,
    maxLength: maxLength,
    truncationStrategy: _truncationConfig?.strategy ?? truncationStrategy,
  );

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
    return _applyBatchPostProcessing(
      texts
          .map((text) => encode(text, addSpecialTokens: addSpecialTokens))
          .toList(),
    );
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
    if (numWorkers != null && numWorkers <= 0) {
      throw ArgumentError.value(numWorkers, 'numWorkers');
    }
    if (texts.length < _kMinBatchSizeForParallel) {
      return encodeBatch(texts, addSpecialTokens: addSpecialTokens);
    }

    final workerCount = numWorkers ?? _getOptimalWorkerCount(texts.length);
    final chunkSize = (texts.length / workerCount).ceil();

    final futures = <Future<List<Encoding>>>[];
    for (var start = 0; start < texts.length; start += chunkSize) {
      final chunk = texts.sublist(
        start,
        (start + chunkSize).clamp(0, texts.length),
      );
      futures.add(
        Isolate.run(
          () => encodeBatch(chunk, addSpecialTokens: addSpecialTokens),
        ),
      );
    }
    return _applyBatchPostProcessing(
      (await Future.wait(futures)).expand((e) => e).toList(),
    );
  }

  Future<List<Encoding>> encodePairBatchParallel(
    List<(String, String)> pairs, {
    bool? addSpecialTokens,
    int? maxLength,
    TruncationStrategy truncationStrategy = TruncationStrategy.longestFirst,
    int? numWorkers,
  }) async {
    if (numWorkers != null && numWorkers <= 0) {
      throw ArgumentError.value(numWorkers, 'numWorkers');
    }
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

    final futures = <Future<List<Encoding>>>[];
    for (var start = 0; start < pairs.length; start += chunkSize) {
      final chunk = pairs.sublist(
        start,
        (start + chunkSize).clamp(0, pairs.length),
      );
      futures.add(
        Isolate.run(
          () => encodePairBatch(
            chunk,
            addSpecialTokens: addSpecialTokens,
            maxLength: maxLength,
            truncationStrategy: truncationStrategy,
          ),
        ),
      );
    }
    return _applyBatchPostProcessing(
      (await Future.wait(futures)).expand((e) => e).toList(),
    );
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
    return _applyBatchPostProcessing(
      pairs
          .map(
            (pair) => encodePair(
              pair.$1,
              pair.$2,
              addSpecialTokens: addSpecialTokens,
              maxLength: maxLength,
              truncationStrategy: truncationStrategy,
            ),
          )
          .toList(),
    );
  }

  List<_TokenInfo> _tokenizeWord(String word) {
    if (word.isEmpty) {
      return [];
    }

    if (word.runes.length > config.maxWordLength) {
      return [
        _TokenInfo(
          token: _unknownToken,
          id: vocab.tokenToId(_unknownToken),
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
            token: _unknownToken,
            id: vocab.tokenToId(_unknownToken),
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
    final tokens = ids
        .map(vocab.idToToken)
        .where(
          (token) =>
              !skipSpecialTokens ||
              !(_json == null
                  ? vocab.isSpecialToken(token)
                  : (_json!['added_tokens'] as List<dynamic>? ?? []).any(
                      (e) => e['content'] == token && e['special'] == true,
                    )),
        )
        .toList();
    final decoder = _json?['decoder'] as Map<String, dynamic>?;
    if (_json != null && decoder == null) return tokens.join(' ');
    final prefix = decoder?['prefix'] as String? ?? config.subwordPrefix;
    final buffer = StringBuffer();
    for (var i = 0; i < tokens.length; i++) {
      var token = tokens[i];
      if (token.startsWith(prefix) && (i > 0 || _json == null)) {
        token = token.substring(prefix.length);
      } else if (i > 0) {
        token = ' $token';
      }
      if (decoder?['cleanup'] == true) {
        for (final entry in const {
          ' .': '.',
          ' ?': '?',
          ' !': '!',
          ' ,': ',',
          " ' ": "'",
          " n't": "n't",
          " 'm": "'m",
          " 's": "'s",
          " 've": "'ve",
          " 're": "'re",
        }.entries) {
          token = token.replaceAll(entry.key, entry.value);
        }
      }
      buffer.write(token);
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

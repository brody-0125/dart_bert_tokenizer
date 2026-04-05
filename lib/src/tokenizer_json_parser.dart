/// Holds the parsed configuration from a `tokenizer.json` file.
class TokenizerJsonConfig {
  /// The vocabulary mapping from token strings to integer IDs.
  final Map<String, int> vocab;

  /// The prefix used for continuation subword tokens (e.g., `##`).
  final String subwordPrefix;

  /// Maximum number of characters in a single word before it becomes `[UNK]`.
  final int maxWordLength;

  /// Whether to convert text to lowercase during normalization.
  final bool lowercase;

  /// Whether to strip accent marks during normalization.
  final bool stripAccents;

  /// Whether to add spaces around Chinese characters.
  final bool handleChineseChars;

  /// Whether to add a `[CLS]` token at the start of sequences.
  final bool addClsToken;

  /// Whether to add a `[SEP]` token at the end of sequences.
  final bool addSepToken;

  const TokenizerJsonConfig({
    required this.vocab,
    this.subwordPrefix = '##',
    this.maxWordLength = 200,
    this.lowercase = true,
    this.stripAccents = true,
    this.handleChineseChars = true,
    this.addClsToken = true,
    this.addSepToken = true,
  });
}

/// Parses a decoded `tokenizer.json` map into a [TokenizerJsonConfig].
///
/// Throws [FormatException] if the JSON structure is invalid or
/// the model type is not `WordPiece`.
TokenizerJsonConfig parseTokenizerJson(Map<String, dynamic> json) {
  // --- model (required) ---
  final model = json['model'];
  if (model == null || model is! Map<String, dynamic>) {
    throw const FormatException(
      'tokenizer.json must contain a "model" object.',
    );
  }

  final modelType = model['type'] as String?;
  // Some HuggingFace tokenizer.json files omit the "type" field.
  // Accept null (inferred as WordPiece if vocab + continuing_subword_prefix
  // are present) or explicit "WordPiece".
  if (modelType != null && modelType != 'WordPiece') {
    throw FormatException(
      'Unsupported model type: "$modelType". Only "WordPiece" is supported.',
    );
  }

  final rawVocab = model['vocab'];
  if (rawVocab == null || rawVocab is! Map<String, dynamic>) {
    throw const FormatException(
      'tokenizer.json model must contain a "vocab" object.',
    );
  }
  final vocab = rawVocab.map<String, int>(
    (key, value) => MapEntry(key, (value as num).toInt()),
  );

  final subwordPrefix =
      (model['continuing_subword_prefix'] as String?) ?? '##';
  final maxWordLength =
      (model['max_input_chars_per_word'] as num?)?.toInt() ?? 200;

  // --- normalizer (optional) ---
  var lowercase = true;
  var stripAccents = true;
  var handleChineseChars = true;

  final normalizer = json['normalizer'];
  if (normalizer is Map<String, dynamic>) {
    lowercase = (normalizer['lowercase'] as bool?) ?? true;
    handleChineseChars =
        (normalizer['handle_chinese_chars'] as bool?) ?? true;

    // strip_accents can be null in HuggingFace tokenizers.
    // When null, infer from lowercase (matches Python tokenizers behavior).
    final rawStripAccents = normalizer['strip_accents'];
    if (rawStripAccents is bool) {
      stripAccents = rawStripAccents;
    } else {
      stripAccents = lowercase;
    }
  }

  // --- post_processor (optional) ---
  var addClsToken = true;
  var addSepToken = true;

  final postProcessor = json['post_processor'];
  if (postProcessor is Map<String, dynamic>) {
    final single = postProcessor['single'];
    if (single is List) {
      addClsToken = _hasSpecialTokenInTemplate(single, '[CLS]');
      addSepToken = _hasSpecialTokenInTemplate(single, '[SEP]');
    }
  }

  // --- added_tokens (optional) ---
  final addedTokens = json['added_tokens'];
  if (addedTokens is List) {
    for (final tokenEntry in addedTokens) {
      if (tokenEntry is Map<String, dynamic>) {
        final content = tokenEntry['content'] as String?;
        final id = (tokenEntry['id'] as num?)?.toInt();
        if (content != null && id != null && !vocab.containsKey(content)) {
          vocab[content] = id;
        }
      }
    }
  }

  return TokenizerJsonConfig(
    vocab: vocab,
    subwordPrefix: subwordPrefix,
    maxWordLength: maxWordLength,
    lowercase: lowercase,
    stripAccents: stripAccents,
    handleChineseChars: handleChineseChars,
    addClsToken: addClsToken,
    addSepToken: addSepToken,
  );
}

/// Checks if a post-processor template list contains a specific special token.
bool _hasSpecialTokenInTemplate(List<dynamic> template, String tokenId) {
  for (final entry in template) {
    if (entry is Map<String, dynamic>) {
      final specialToken = entry['SpecialToken'];
      if (specialToken is Map<String, dynamic>) {
        if (specialToken['id'] == tokenId) return true;
      }
    }
  }
  return false;
}

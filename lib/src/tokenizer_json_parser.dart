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
  try {
    return _parseTokenizerJson(json);
  } on TypeError catch (error) {
    throw FormatException('Invalid tokenizer.json field type: $error');
  }
}

TokenizerJsonConfig _parseTokenizerJson(Map<String, dynamic> json) {
  _validateComponents(json);
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
  final vocab = rawVocab.map<String, int>((key, value) {
    if (value is! int || value < 0 || value > 0x7fffffff) {
      throw FormatException('Invalid vocabulary ID for "$key": $value');
    }
    return MapEntry(key, value);
  });

  if (vocab.values.toSet().length != vocab.length) {
    throw const FormatException('Duplicate vocabulary IDs');
  }

  final subwordPrefix = (model['continuing_subword_prefix'] as String?) ?? '##';
  final maxWordLength =
      (model['max_input_chars_per_word'] as num?)?.toInt() ?? 100;

  if (maxWordLength <= 0) {
    throw const FormatException('max_input_chars_per_word must be positive');
  }

  // --- normalizer (optional) ---
  var lowercase = false;
  var stripAccents = false;
  var handleChineseChars = false;

  final normalizer = json['normalizer'];
  if (normalizer is Map<String, dynamic>) {
    lowercase = (normalizer['lowercase'] as bool?) ?? true;
    handleChineseChars = (normalizer['handle_chinese_chars'] as bool?) ?? true;

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
  var addClsToken = false;
  var addSepToken = false;

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
    var nextId = vocab.length;
    for (final tokenEntry in addedTokens) {
      if (tokenEntry is Map<String, dynamic>) {
        final content = tokenEntry['content'] as String?;
        if (content != null) {
          final id = vocab[content] ?? nextId;
          vocab[content] = id;
          if (id >= nextId) nextId = id + 1;
        }
      }
    }
  }

  final unknown = model['unk_token'] as String? ?? '[UNK]';
  if (!vocab.containsKey(unknown)) {
    throw FormatException('Unknown token "$unknown" is absent from vocabulary');
  }
  _validatePostProcessor(
    json['post_processor'] as Map<String, dynamic>?,
    vocab,
  );
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

void _validateComponents(Map<String, dynamic> json) {
  const supported = {
    'normalizer': ['BertNormalizer'],
    'pre_tokenizer': ['BertPreTokenizer'],
    'post_processor': ['TemplateProcessing', 'BertProcessing'],
    'decoder': ['WordPiece'],
  };
  for (final entry in supported.entries) {
    final component = json[entry.key];
    if (component != null &&
        (component is! Map<String, dynamic> ||
            !entry.value.contains(component['type']))) {
      throw FormatException('Unsupported ${entry.key}: $component');
    }
  }
  for (final key in ['padding', 'truncation']) {
    if (json[key] != null && json[key] is! Map<String, dynamic>) {
      throw FormatException('$key must be an object or null');
    }
  }
  final truncation = json['truncation'];
  if (truncation is Map && (truncation['stride'] ?? 0) != 0) {
    throw const FormatException('Truncation stride/overflow is not supported');
  }
  final model = json['model'];
  if (model is Map) {
    final max = model['max_input_chars_per_word'];
    if (max != null && (max is! int || max <= 0)) {
      throw const FormatException('Invalid max_input_chars_per_word');
    }
  }
  if (truncation is Map) {
    if (truncation['max_length'] is! int ||
        truncation['max_length'] < 0 ||
        ![null, 'Left', 'Right'].contains(truncation['direction']) ||
        ![
          'LongestFirst',
          'OnlyFirst',
          'OnlySecond',
        ].contains(truncation['strategy'])) {
      throw const FormatException('Invalid truncation configuration');
    }
  }
  final padding = json['padding'];
  if (padding is Map) {
    final strategy = padding['strategy'];
    if (strategy != 'BatchLongest' &&
        !(strategy is Map &&
            strategy.length == 1 &&
            strategy['Fixed'] is int &&
            strategy['Fixed'] >= 0)) {
      throw const FormatException('Unsupported padding strategy');
    }
    final multiple = padding['pad_to_multiple_of'];
    if (![null, 'Left', 'Right'].contains(padding['direction']) ||
        padding['pad_id'] is! int ||
        padding['pad_id'] < 0 ||
        padding['pad_id'] > 0x7fffffff ||
        padding['pad_token'] is! String ||
        padding['pad_type_id'] is! int ||
        padding['pad_type_id'] < 0 ||
        padding['pad_type_id'] > 255 ||
        (multiple != null && (multiple is! int || multiple <= 0))) {
      throw const FormatException('Invalid padding configuration');
    }
  }
  final added = json['added_tokens'];
  if (added != null && added is! List) {
    throw const FormatException('added_tokens must be a list');
  }
  for (final token in (added as List? ?? const [])) {
    if (token is! Map ||
        token['content'] is! String ||
        (token['content'] as String).isEmpty ||
        token['id'] is! int ||
        (token['id'] as int) < 0) {
      throw const FormatException('Invalid added token');
    }
    for (final flag in ['single_word', 'lstrip', 'rstrip', 'normalized']) {
      if (token[flag] == true) {
        throw FormatException('AddedToken.$flag is not supported');
      }
    }
  }
}

void _validatePostProcessor(
  Map<String, dynamic>? processor,
  Map<String, int> vocab,
) {
  if (processor == null) return;
  if (processor['type'] == 'BertProcessing') {
    for (final key in ['cls', 'sep']) {
      final token = processor[key];
      if (token is! List ||
          token.length != 2 ||
          token[0] is! String ||
          token[1] != vocab[token[0]]) {
        throw FormatException('Invalid BertProcessing.$key');
      }
    }
    return;
  }
  for (final key in ['single', 'pair']) {
    final template = processor[key];
    if (template == null && key == 'pair') continue;
    if (template is! List) {
      throw FormatException('TemplateProcessing.$key must be a list');
    }
    final seen = <String>{};
    for (final entry in template) {
      if (entry is! Map || entry.length != 1) {
        throw const FormatException('Invalid template entry');
      }
      final seq = entry['Sequence'];
      final token = entry['SpecialToken'];
      final part = seq ?? token;
      if (part is! Map ||
          part['type_id'] is! int ||
          (part['type_id'] as int) < 0 ||
          (part['type_id'] as int) > 255) {
        throw const FormatException(
          'Template type_id must be between 0 and 255',
        );
      }
      if (seq != null) {
        final id = seq['id'];
        if ((id != 'A' && id != 'B') ||
            (key == 'single' && id == 'B') ||
            !seen.add(id as String)) {
          throw const FormatException(
            'Each input sequence must appear once in the template',
          );
        }
      } else if (token != null) {
        final name = token['id'];
        if (name is! String || !vocab.containsKey(name)) {
          throw const FormatException('Unknown template token');
        }
        final definition = (processor['special_tokens'] as Map?)?[name];
        if (definition != null &&
            (definition['ids'] is! List ||
                (definition['ids'] as List).length != 1 ||
                definition['ids'][0] != vocab[name] ||
                definition['tokens'] is! List ||
                (definition['tokens'] as List).length != 1 ||
                definition['tokens'][0] != name)) {
          throw const FormatException(
            'Only single vocabulary-token template entries are supported',
          );
        }
      } else {
        throw const FormatException('Unknown template entry');
      }
    }
    if (!seen.contains('A') || (key == 'pair' && !seen.contains('B'))) {
      throw const FormatException('Template is missing an input sequence');
    }
  }
}

import 'dart:convert';

import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
import 'package:test/test.dart';

/// Builds a minimal tokenizer.json map for testing.
Map<String, dynamic> _buildMinimalTokenizerJson({
  Map<String, int>? vocab,
  Map<String, dynamic>? normalizer,
  Map<String, dynamic>? postProcessor,
  List<Map<String, dynamic>>? addedTokens,
  String? subwordPrefix,
  int? maxInputCharsPerWord,
}) {
  final defaultVocab = {
    '[PAD]': 0,
    '[UNK]': 100,
    '[CLS]': 101,
    '[SEP]': 102,
    '[MASK]': 103,
    'hello': 7592,
    'world': 2088,
    'the': 1996,
    '##s': 2015,
    '##ing': 2075,
    ',': 1010,
    '.': 1012,
    'test': 3231,
    '!': 999,
  };

  return {
    'version': '1.0',
    'truncation': null,
    'padding': null,
    'added_tokens': addedTokens ??
        [
          {
            'id': 0,
            'special': true,
            'content': '[PAD]',
            'single_word': false,
            'lstrip': false,
            'rstrip': false,
            'normalized': false,
          },
          {
            'id': 100,
            'special': true,
            'content': '[UNK]',
            'single_word': false,
            'lstrip': false,
            'rstrip': false,
            'normalized': false,
          },
          {
            'id': 101,
            'special': true,
            'content': '[CLS]',
            'single_word': false,
            'lstrip': false,
            'rstrip': false,
            'normalized': false,
          },
          {
            'id': 102,
            'special': true,
            'content': '[SEP]',
            'single_word': false,
            'lstrip': false,
            'rstrip': false,
            'normalized': false,
          },
          {
            'id': 103,
            'special': true,
            'content': '[MASK]',
            'single_word': false,
            'lstrip': false,
            'rstrip': false,
            'normalized': false,
          },
        ],
    'normalizer': normalizer ??
        {
          'type': 'BertNormalizer',
          'clean_text': true,
          'handle_chinese_chars': true,
          'strip_accents': null,
          'lowercase': true,
        },
    'pre_tokenizer': {'type': 'BertPreTokenizer'},
    'post_processor': postProcessor ??
        {
          'type': 'TemplateProcessing',
          'single': [
            {
              'SpecialToken': {'id': '[CLS]', 'type_id': 0},
            },
            {
              'Sequence': {'id': 'A', 'type_id': 0},
            },
            {
              'SpecialToken': {'id': '[SEP]', 'type_id': 0},
            },
          ],
          'pair': [
            {
              'SpecialToken': {'id': '[CLS]', 'type_id': 0},
            },
            {
              'Sequence': {'id': 'A', 'type_id': 0},
            },
            {
              'SpecialToken': {'id': '[SEP]', 'type_id': 0},
            },
            {
              'Sequence': {'id': 'B', 'type_id': 1},
            },
            {
              'SpecialToken': {'id': '[SEP]', 'type_id': 1},
            },
          ],
          'special_tokens': {
            '[CLS]': {
              'id': '[CLS]',
              'ids': [101],
              'tokens': ['[CLS]'],
            },
            '[SEP]': {
              'id': '[SEP]',
              'ids': [102],
              'tokens': ['[SEP]'],
            },
          },
        },
    'decoder': {'type': 'WordPiece', 'prefix': '##', 'cleanup': true},
    'model': {
      'type': 'WordPiece',
      'unk_token': '[UNK]',
      'continuing_subword_prefix': subwordPrefix ?? '##',
      'max_input_chars_per_word': maxInputCharsPerWord ?? 100,
      'vocab': vocab ?? defaultVocab,
    },
  };
}

void main() {
  group('fromTokenizerJsonString', () {
    test('creates a tokenizer from minimal JSON', () {
      final json = _buildMinimalTokenizerJson();
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      final encoding = tokenizer.encode('hello world');
      expect(encoding.tokens, equals(['[CLS]', 'hello', 'world', '[SEP]']));
      expect(encoding.ids, equals([101, 7592, 2088, 102]));
    });

    test('handles subword tokenization', () {
      final json = _buildMinimalTokenizerJson();
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      final encoding = tokenizer.encode('testing');
      expect(encoding.tokens, equals(['[CLS]', 'test', '##ing', '[SEP]']));
    });

    test('handles unknown tokens', () {
      final json = _buildMinimalTokenizerJson();
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      final encoding = tokenizer.encode('xyz');
      expect(encoding.tokens, equals(['[CLS]', '[UNK]', '[SEP]']));
    });
  });

  group('parseTokenizerJson normalizer extraction', () {
    test('extracts lowercase=true, infers stripAccents from null', () {
      final json = _buildMinimalTokenizerJson(
        normalizer: {
          'type': 'BertNormalizer',
          'clean_text': true,
          'handle_chinese_chars': true,
          'strip_accents': null,
          'lowercase': true,
        },
      );
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      expect(tokenizer.config.lowercase, isTrue);
      expect(tokenizer.config.stripAccents, isTrue);
      expect(tokenizer.config.handleChineseChars, isTrue);
    });

    test('extracts lowercase=false, infers stripAccents=false from null', () {
      final json = _buildMinimalTokenizerJson(
        normalizer: {
          'type': 'BertNormalizer',
          'clean_text': true,
          'handle_chinese_chars': false,
          'strip_accents': null,
          'lowercase': false,
        },
      );
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      expect(tokenizer.config.lowercase, isFalse);
      expect(tokenizer.config.stripAccents, isFalse);
      expect(tokenizer.config.handleChineseChars, isFalse);
    });

    test('respects explicit strip_accents=true even with lowercase=false', () {
      final json = _buildMinimalTokenizerJson(
        normalizer: {
          'type': 'BertNormalizer',
          'clean_text': true,
          'handle_chinese_chars': true,
          'strip_accents': true,
          'lowercase': false,
        },
      );
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      expect(tokenizer.config.lowercase, isFalse);
      expect(tokenizer.config.stripAccents, isTrue);
    });

    test('defaults when normalizer is null', () {
      final json = _buildMinimalTokenizerJson(normalizer: null);
      // Need to set normalizer to null explicitly in the map
      json['normalizer'] = null;
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      expect(tokenizer.config.lowercase, isTrue);
      expect(tokenizer.config.stripAccents, isTrue);
      expect(tokenizer.config.handleChineseChars, isTrue);
    });
  });

  group('post_processor extraction', () {
    test('extracts addClsToken and addSepToken from template', () {
      final json = _buildMinimalTokenizerJson();
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      expect(tokenizer.config.addClsToken, isTrue);
      expect(tokenizer.config.addSepToken, isTrue);
    });

    test('detects missing CLS in template', () {
      final json = _buildMinimalTokenizerJson(
        postProcessor: {
          'type': 'TemplateProcessing',
          'single': [
            {
              'Sequence': {'id': 'A', 'type_id': 0},
            },
            {
              'SpecialToken': {'id': '[SEP]', 'type_id': 0},
            },
          ],
        },
      );
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      expect(tokenizer.config.addClsToken, isFalse);
      expect(tokenizer.config.addSepToken, isTrue);
    });

    test('defaults when post_processor is null', () {
      final json = _buildMinimalTokenizerJson(postProcessor: null);
      json['post_processor'] = null;
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      expect(tokenizer.config.addClsToken, isTrue);
      expect(tokenizer.config.addSepToken, isTrue);
    });
  });

  group('configOverride', () {
    test('overrides JSON-extracted config', () {
      final json = _buildMinimalTokenizerJson();
      final tokenizer = WordPieceTokenizer.fromTokenizerJsonString(
        jsonEncode(json),
        configOverride: const WordPieceConfig(
          lowercase: false,
          stripAccents: false,
          addClsToken: false,
          addSepToken: false,
        ),
      );

      expect(tokenizer.config.lowercase, isFalse);
      expect(tokenizer.config.stripAccents, isFalse);
      expect(tokenizer.config.addClsToken, isFalse);
      expect(tokenizer.config.addSepToken, isFalse);

      // Without CLS/SEP
      final encoding = tokenizer.encode('Hello');
      expect(encoding.tokens.contains('[CLS]'), isFalse);
      expect(encoding.tokens.contains('[SEP]'), isFalse);
    });
  });

  group('error handling', () {
    test('throws FormatException for non-WordPiece model', () {
      final json = _buildMinimalTokenizerJson();
      (json['model'] as Map<String, dynamic>)['type'] = 'BPE';

      expect(
        () => WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json)),
        throwsA(isA<FormatException>().having(
          (e) => e.message,
          'message',
          contains('BPE'),
        )),
      );
    });

    test('throws FormatException for missing model key', () {
      final json = _buildMinimalTokenizerJson();
      json.remove('model');

      expect(
        () => WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json)),
        throwsA(isA<FormatException>()),
      );
    });

    test('throws FormatException for missing vocab', () {
      final json = _buildMinimalTokenizerJson();
      (json['model'] as Map<String, dynamic>).remove('vocab');

      expect(
        () => WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json)),
        throwsA(isA<FormatException>()),
      );
    });
  });

  group('added_tokens merging', () {
    test('merges added_tokens not present in vocab', () {
      final json = _buildMinimalTokenizerJson(
        addedTokens: [
          {
            'id': 999999,
            'special': true,
            'content': '[CUSTOM]',
            'single_word': false,
            'lstrip': false,
            'rstrip': false,
            'normalized': false,
          },
        ],
      );
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      expect(tokenizer.convertTokensToIds(['[CUSTOM]']), equals([999999]));
    });

    test('does not override vocab entries with added_tokens', () {
      final json = _buildMinimalTokenizerJson(
        addedTokens: [
          {
            'id': 9999,
            'special': true,
            'content': '[PAD]',
            'single_word': false,
            'lstrip': false,
            'rstrip': false,
            'normalized': false,
          },
        ],
      );
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));

      // [PAD] should keep its original ID 0 from vocab, not 9999
      expect(tokenizer.convertTokensToIds(['[PAD]']), equals([0]));
    });
  });

  group('model config extraction', () {
    test('extracts max_input_chars_per_word', () {
      final json = _buildMinimalTokenizerJson(maxInputCharsPerWord: 50);
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));
      expect(tokenizer.config.maxWordLength, equals(50));
    });

    test('extracts continuing_subword_prefix', () {
      final json = _buildMinimalTokenizerJson(subwordPrefix: '@@');
      final tokenizer =
          WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json));
      expect(tokenizer.config.subwordPrefix, equals('@@'));
    });
  });

  group('file-based loading', () {
    test('fromTokenizerJsonSync loads from file', () {
      final tokenizer = WordPieceTokenizer.fromTokenizerJsonSync(
        'test/fixtures/tokenizer.json',
      );

      final encoding = tokenizer.encode('Hello, world!');
      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));
      expect(encoding.tokens.length, greaterThan(3));
    });

    test('fromTokenizerJson loads from file asynchronously', () async {
      final tokenizer = await WordPieceTokenizer.fromTokenizerJson(
        'test/fixtures/tokenizer.json',
      );

      final encoding = tokenizer.encode('Hello, world!');
      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));
      expect(encoding.tokens.length, greaterThan(3));
    });
  });

  group('vocab.txt vs tokenizer.json equivalence', () {
    late WordPieceTokenizer vocabTokenizer;
    late WordPieceTokenizer jsonTokenizer;

    setUpAll(() {
      vocabTokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt');
      jsonTokenizer = WordPieceTokenizer.fromTokenizerJsonSync(
        'test/fixtures/tokenizer.json',
      );
    });

    test('produces identical encodings for simple text', () {
      final texts = [
        'Hello, world!',
        'The quick brown fox jumps over the lazy dog.',
        'BERT tokenization test.',
        'I love machine learning!',
      ];

      for (final text in texts) {
        final vocabEncoding = vocabTokenizer.encode(text);
        final jsonEncoding = jsonTokenizer.encode(text);

        expect(jsonEncoding.ids, equals(vocabEncoding.ids),
            reason: 'IDs mismatch for "$text"');
        expect(jsonEncoding.tokens, equals(vocabEncoding.tokens),
            reason: 'Tokens mismatch for "$text"');
      }
    });

    test('produces identical encodings for text pairs', () {
      final vocabEncoding = vocabTokenizer.encodePair(
        'What is Dart?',
        'Dart is a programming language.',
      );
      final jsonEncoding = jsonTokenizer.encodePair(
        'What is Dart?',
        'Dart is a programming language.',
      );

      expect(jsonEncoding.ids, equals(vocabEncoding.ids));
      expect(jsonEncoding.tokens, equals(vocabEncoding.tokens));
      expect(jsonEncoding.typeIds, equals(vocabEncoding.typeIds));
    });

    test('produces identical encodings for subword-heavy text', () {
      final text = 'Tokenization preprocessing unrecognizable';
      final vocabEncoding = vocabTokenizer.encode(text);
      final jsonEncoding = jsonTokenizer.encode(text);

      expect(jsonEncoding.ids, equals(vocabEncoding.ids));
      expect(jsonEncoding.tokens, equals(vocabEncoding.tokens));
    });

    test('produces identical encodings with special characters', () {
      final text = 'café résumé naïve';
      final vocabEncoding = vocabTokenizer.encode(text);
      final jsonEncoding = jsonTokenizer.encode(text);

      expect(jsonEncoding.ids, equals(vocabEncoding.ids));
      expect(jsonEncoding.tokens, equals(vocabEncoding.tokens));
    });

    test('decode round-trip matches', () {
      final text = 'Hello world, this is a test!';
      final vocabEncoding = vocabTokenizer.encode(text);
      final jsonEncoding = jsonTokenizer.encode(text);

      final vocabDecoded = vocabTokenizer.decode(vocabEncoding.ids.toList());
      final jsonDecoded = jsonTokenizer.decode(jsonEncoding.ids.toList());

      expect(jsonDecoded, equals(vocabDecoded));
    });
  });
}

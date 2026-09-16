import 'dart:convert';
import 'dart:io';

import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
import 'package:test/test.dart';

Map<String, dynamic> pipeline() =>
    jsonDecode(
          File(
            'test/fixtures/huggingface/bert-base-uncased.reduced.json',
          ).readAsStringSync(),
        )
        as Map<String, dynamic>;

void main() {
  final invalid = <String, void Function(Map<String, dynamic>)>{
    'unknown normalizer': (j) => j['normalizer'] = {'type': 'NFKC'},
    'unknown pre-tokenizer': (j) => j['pre_tokenizer'] = {'type': 'ByteLevel'},
    'unknown processor': (j) =>
        j['post_processor'] = {'type': 'RobertaProcessing'},
    'unknown decoder': (j) => j['decoder'] = {'type': 'ByteLevel'},
    'negative vocab ID': (j) => j['model']['vocab']['hello'] = -1,
    'fractional vocab ID': (j) => j['model']['vocab']['hello'] = 1.5,
    'duplicate vocab ID': (j) => j['model']['vocab']['hello'] = 0,
    'missing unknown': (j) => j['model']['unk_token'] = 'absent',
    'stride': (j) => j['truncation'] = {'max_length': 10, 'stride': 1},
    'invalid added tokens': (j) => j['added_tokens'] = {},
    'empty added token': (j) => j['added_tokens'][0]['content'] = '',
    'added-token normalized matching': (j) =>
        j['added_tokens'][0]['normalized'] = 'true',
    'added-token single word': (j) => j['added_tokens'][0]['single_word'] = 1,
    'added-token lstrip': (j) => j['added_tokens'][0]['lstrip'] = 'yes',
    'added-token rstrip': (j) => j['added_tokens'][0]['rstrip'] = [],
    'out-of-range template type': (j) =>
        j['post_processor']['single'][1]['Sequence']['type_id'] = 256,
    'missing template sequence': (j) => j['post_processor']['single'] = [],
  };
  for (final entry in invalid.entries) {
    test('rejects ${entry.key}', () {
      final json = pipeline();
      entry.value(json);
      expect(
        () => WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(json)),
        throwsFormatException,
      );
    });
  }
  test('rejects non-object JSON', () {
    expect(
      () => WordPieceTokenizer.fromTokenizerJsonString('[]'),
      throwsFormatException,
    );
  });
  test('rejects invalid public arguments', () async {
    final t = WordPieceTokenizer.fromTokenizerJsonString(
      jsonEncode(pipeline()),
    );
    expect(() => t.enablePadding(length: -1), throwsArgumentError);
    expect(() => t.enablePadding(padToMultipleOf: 0), throwsArgumentError);
    expect(() => t.enableTruncation(maxLength: -1), throwsArgumentError);
    await expectLater(
      t.encodeBatchParallel(['hello'], numWorkers: 0),
      throwsArgumentError,
    );
    await expectLater(
      t.encodePairBatchParallel([('hello', 'world')], numWorkers: 0),
      throwsArgumentError,
    );
    t.enableTruncation(maxLength: 1);
    expect(() => t.encode('hello'), throwsArgumentError);
  });
  test('impossible only-first and only-second truncation fail', () {
    final t = WordPieceTokenizer.fromTokenizerJsonString(
      jsonEncode(pipeline()),
    );
    expect(
      () => t.encodePair(
        'hello',
        'hello world',
        maxLength: 5,
        truncationStrategy: TruncationStrategy.onlyFirst,
      ),
      throwsArgumentError,
    );
    expect(
      () => t.encodePair(
        'hello world',
        'hello',
        maxLength: 5,
        truncationStrategy: TruncationStrategy.onlySecond,
      ),
      throwsArgumentError,
    );
  });
  test('pair word lookup uses sequence-local word IDs', () {
    final t = WordPieceTokenizer.fromTokenizerJsonString(
      jsonEncode(pipeline()),
    );
    final e = t.encodePair('hello world', 'hello');
    expect(e.wordToChars(0, sequenceIndex: 1), (0, 5));
    expect(e.wordToTokens(0, sequenceIndex: 1), (4, 5));
    expect(e.charToWord(1, sequenceIndex: 1), 0);
  });
}

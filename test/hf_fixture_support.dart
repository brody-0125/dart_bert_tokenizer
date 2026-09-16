import 'dart:convert';
import 'dart:io';

import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
import 'package:test/test.dart';

/// Register identical assertions for reduced offline and full network models.
/// Read [raw] inside each test because network setup populates it later.
void registerHfModelTests(Map<String, dynamic> model, String Function() raw) {
  if (model['expected_error'] case final String error) {
    test('rejects unsupported pipeline', () {
      expect(
        () => WordPieceTokenizer.fromTokenizerJsonString(raw()),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains(error),
          ),
        ),
      );
    });
    return;
  }
  final golden =
      jsonDecode(
            File(
              'test/fixtures/huggingface/${model['name']}.golden.json',
            ).readAsStringSync(),
          )
          as Map<String, dynamic>;
  for (final item in golden['cases'] as List<dynamic>) {
    final fixture = item as Map<String, dynamic>;
    test(fixture['name'] as String, () => runHfCase(raw(), fixture));
  }
}

void expectHfEncoding(
  WordPieceTokenizer tokenizer,
  Encoding actual,
  Map<String, dynamic> expected,
) {
  final fields = actual.toMap();
  for (final key in fields.keys) {
    expect(fields[key], expected[key], reason: key);
  }
  expect(tokenizer.decode(actual.ids), expected['decoded'], reason: 'decode');
  expect(
    tokenizer.decode(actual.ids, skipSpecialTokens: false),
    expected['decoded_with_special_tokens'],
    reason: 'decode with specials',
  );
}

Future<void> runHfCase(String raw, Map<String, dynamic> fixture) async {
  final tokenizer = WordPieceTokenizer.fromTokenizerJsonString(raw);
  final padding = fixture['padding'] as Map<String, dynamic>?;
  if (padding != null) {
    tokenizer.enablePadding(
      length: padding['length'] as int?,
      padToMultipleOf: padding['pad_to_multiple_of'] as int?,
      direction: padding['direction'] == 'left'
          ? PaddingDirection.left
          : PaddingDirection.right,
    );
  }
  final truncation = fixture['truncation'] as Map<String, dynamic>?;
  if (truncation != null) {
    tokenizer.enableTruncation(
      maxLength: truncation['max_length'] as int,
      direction: truncation['direction'] == 'left'
          ? TruncationDirection.left
          : TruncationDirection.right,
      strategy: switch (truncation['strategy']) {
        'only_first' => TruncationStrategy.onlyFirst,
        'only_second' => TruncationStrategy.onlySecond,
        _ => TruncationStrategy.longestFirst,
      },
    );
  }
  final List<Encoding> sequential;
  final List<Encoding> parallel;
  if (fixture['pair_batch'] case final List<dynamic> batch) {
    final pairs = batch.map((p) => (p[0] as String, p[1] as String)).toList();
    sequential = tokenizer.encodePairBatch(pairs);
    parallel = await tokenizer.encodePairBatchParallel(pairs, numWorkers: 2);
  } else if (fixture['batch'] case final List<dynamic> batch) {
    final texts = batch.cast<String>();
    sequential = tokenizer.encodeBatch(texts);
    parallel = await tokenizer.encodeBatchParallel(texts, numWorkers: 2);
  } else {
    final special = fixture['add_special_tokens'] as bool?;
    final input = fixture['input'] as String;
    final pair = fixture['pair'] as String?;
    final encoding = pair != null
        ? tokenizer.encodePair(input, pair, addSpecialTokens: special)
        : tokenizer.encode(input, addSpecialTokens: special);
    expectHfEncoding(
      tokenizer,
      encoding,
      fixture['expected'] as Map<String, dynamic>,
    );
    return;
  }
  final expected = fixture['expected'] as List<dynamic>;
  for (final encodings in [sequential, parallel]) {
    expect(encodings.length, expected.length);
    for (var i = 0; i < expected.length; i++) {
      expectHfEncoding(
        tokenizer,
        encodings[i],
        expected[i] as Map<String, dynamic>,
      );
    }
  }
}

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
  for (final row in expected['alignment'] as List<dynamic>? ?? []) {
    final seq = row[0] as int;
    final word = row[1] as int;
    (int, int)? span(dynamic value) =>
        value == null ? null : (value[0] as int, value[1] as int);
    expect(actual.wordToTokens(word, sequenceIndex: seq), span(row[2]));
    expect(actual.wordToChars(word, sequenceIndex: seq), span(row[3]));
    for (var pos = 0; pos < 12; pos++) {
      int? match;
      for (var i = 0; i < actual.length; i++) {
        if (expected['sequence_ids'][i] == seq &&
            expected['word_ids'][i] == word &&
            expected['offsets'][i][0] <= pos &&
            pos < expected['offsets'][i][1]) {
          match = i;
          break;
        }
      }
      expect(
        actual.charToToken(pos, sequenceIndex: seq, wordIndex: word),
        match,
      );
      expect(
        actual.charToWord(pos, sequenceIndex: seq, wordIndex: word),
        match == null ? null : word,
      );
    }
  }
}

Future<void> runHfCase(String raw, Map<String, dynamic> fixture) =>
    runHfTokenizerCase(
      WordPieceTokenizer.fromTokenizerJsonString(raw),
      fixture,
    );

Future<void> runHfTokenizerCase(
  WordPieceTokenizer tokenizer,
  Map<String, dynamic> fixture,
) async {
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
  if (fixture['pretokenized'] == true) {
    List<String> words(dynamic value) => (value as List).cast<String>();
    if (fixture['pair_batch'] case final List<dynamic> batch) {
      final pairs = batch.map((p) => (words(p[0]), words(p[1]))).toList();
      sequential = tokenizer.encodePreTokenizedPairBatch(pairs);
      parallel = await tokenizer.encodePreTokenizedPairBatchParallel(
        pairs,
        numWorkers: 2,
      );
    } else if (fixture['batch'] case final List<dynamic> batch) {
      final inputs = batch.map(words).toList();
      sequential = tokenizer.encodePreTokenizedBatch(inputs);
      parallel = await tokenizer.encodePreTokenizedBatchParallel(
        inputs,
        numWorkers: 2,
      );
    } else {
      final input = words(fixture['input']);
      final special = fixture['add_special_tokens'] as bool?;
      final encoding = fixture['pair'] == null
          ? tokenizer.encodePreTokenized(input, addSpecialTokens: special)
          : tokenizer.encodePreTokenizedPair(
              input,
              words(fixture['pair']),
              addSpecialTokens: special,
            );
      expectHfEncoding(
        tokenizer,
        encoding,
        fixture['expected'] as Map<String, dynamic>,
      );
      return;
    }
  } else if (fixture['pair_batch'] case final List<dynamic> batch) {
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

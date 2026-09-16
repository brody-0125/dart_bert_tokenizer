import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
import 'package:test/test.dart';

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
  if (fixture['pair_batch'] case final List<dynamic> batch) {
    final pairs = batch.map((p) => (p[0] as String, p[1] as String)).toList();
    final expected = fixture['expected'] as List<dynamic>;
    final sequential = tokenizer.encodePairBatch(pairs);
    final parallel = await tokenizer.encodePairBatchParallel(
      pairs,
      numWorkers: 2,
    );
    for (var i = 0; i < pairs.length; i++) {
      expectHfEncoding(
        tokenizer,
        sequential[i],
        expected[i] as Map<String, dynamic>,
      );
      expectHfEncoding(
        tokenizer,
        parallel[i],
        expected[i] as Map<String, dynamic>,
      );
    }
  } else if (fixture['batch'] case final List<dynamic> batch) {
    final texts = batch.cast<String>();
    final expected = fixture['expected'] as List<dynamic>;
    final sequential = tokenizer.encodeBatch(texts);
    final parallel = await tokenizer.encodeBatchParallel(texts, numWorkers: 2);
    for (var i = 0; i < texts.length; i++) {
      expectHfEncoding(
        tokenizer,
        sequential[i],
        expected[i] as Map<String, dynamic>,
      );
      expectHfEncoding(
        tokenizer,
        parallel[i],
        expected[i] as Map<String, dynamic>,
      );
    }
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
  }
}

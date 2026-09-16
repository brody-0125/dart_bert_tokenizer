import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
import 'package:test/test.dart';

void main() {
  late WordPieceTokenizer tokenizer;

  setUp(() {
    final vocab = Vocabulary.fromTokens([
      '[PAD]',
      ...List.generate(99, (i) => '[unused$i]'),
      '[UNK]',
      '[CLS]',
      '[SEP]',
      '[MASK]',
      ...List.generate(896, (i) => '[unused${99 + i}]'),
      'the',
      'a',
      'is',
      'it',
      'this',
      'that',
      'what',
      'how',
      'hello',
      'world',
      'test',
      'word',
      'sentence',
      ',',
      '.',
      '!',
      '?',
      '##s',
      '##ed',
      '##ing',
    ]);
    tokenizer = WordPieceTokenizer(vocab: vocab);
  });

  group('P2 Parallel Batch Encoding', () {
    test('encodeBatchParallel produces same results as encodeBatch', () async {
      final texts = List.generate(20, (i) => 'hello world test sentence $i');

      final sequential = tokenizer.encodeBatch(texts);
      final parallel = await tokenizer.encodeBatchParallel(texts);

      expect(parallel.length, equals(sequential.length));

      for (var i = 0; i < texts.length; i++) {
        expect(
          parallel[i].ids.toList(),
          equals(sequential[i].ids.toList()),
          reason: 'IDs mismatch at index $i',
        );
        expect(
          parallel[i].tokens.toList(),
          equals(sequential[i].tokens.toList()),
          reason: 'Tokens mismatch at index $i',
        );
      }
    });

    test(
      'encodeBatchParallel falls back to sequential for small batches',
      () async {
        final texts = ['hello', 'world']; // < 8 items

        final sequential = tokenizer.encodeBatch(texts);
        final parallel = await tokenizer.encodeBatchParallel(texts);

        expect(parallel.length, equals(sequential.length));
        for (var i = 0; i < texts.length; i++) {
          expect(parallel[i].ids.toList(), equals(sequential[i].ids.toList()));
        }
      },
    );

    test('encodeBatchParallel respects numWorkers parameter', () async {
      final texts = List.generate(16, (i) => 'hello world $i');

      // Should work with different worker counts
      final result1 = await tokenizer.encodeBatchParallel(texts, numWorkers: 1);
      final result2 = await tokenizer.encodeBatchParallel(texts, numWorkers: 2);
      final result4 = await tokenizer.encodeBatchParallel(texts, numWorkers: 4);

      expect(result1.length, equals(texts.length));
      expect(result2.length, equals(texts.length));
      expect(result4.length, equals(texts.length));

      // All should produce same results
      for (var i = 0; i < texts.length; i++) {
        expect(result2[i].ids.toList(), equals(result1[i].ids.toList()));
        expect(result4[i].ids.toList(), equals(result1[i].ids.toList()));
      }
    });

    test(
      'encodePairBatchParallel produces same results as encodePairBatch',
      () async {
        final pairs = List.generate(
          16,
          (i) => ('hello world $i', 'test sentence $i'),
        );

        final sequential = tokenizer.encodePairBatch(pairs);
        final parallel = await tokenizer.encodePairBatchParallel(pairs);

        expect(parallel.length, equals(sequential.length));

        for (var i = 0; i < pairs.length; i++) {
          expect(
            parallel[i].ids.toList(),
            equals(sequential[i].ids.toList()),
            reason: 'IDs mismatch at index $i',
          );
          expect(
            parallel[i].typeIds.toList(),
            equals(sequential[i].typeIds.toList()),
            reason: 'TypeIds mismatch at index $i',
          );
        }
      },
    );

    test('encodeBatchParallel with padding and truncation', () async {
      final configuredTokenizer = WordPieceTokenizer(vocab: tokenizer.vocab)
        ..enableTruncation(maxLength: 10)
        ..enablePadding(length: 10);

      final texts = List.generate(12, (i) => 'hello world test $i');

      final sequential = configuredTokenizer.encodeBatch(texts);
      final parallel = await configuredTokenizer.encodeBatchParallel(texts);

      expect(parallel.length, equals(sequential.length));

      for (var i = 0; i < texts.length; i++) {
        expect(parallel[i].length, equals(10));
        expect(parallel[i].ids.toList(), equals(sequential[i].ids.toList()));
      }
    });
  });
}

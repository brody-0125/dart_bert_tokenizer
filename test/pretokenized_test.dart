import 'dart:convert';
import 'dart:io';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
import 'package:test/test.dart';
import 'hf_fixture_support.dart';

WordPieceTokenizer tokenizer() => WordPieceTokenizer(
  vocab: Vocabulary.fromTokens([
    '[PAD]',
    '[UNK]',
    '[CLS]',
    '[SEP]',
    '[MASK]',
    'hello',
    'world',
    ',',
    'cafe',
    'a',
    '##b',
    '😀',
  ]),
);

void main() {
  test('JSON config override and invalid alignment queries', () {
    final t = WordPieceTokenizer.fromTokenizerJsonSync(
      'test/fixtures/huggingface/bert-base-uncased.reduced.json',
      configOverride: const WordPieceConfig(
        lowercase: false,
        addClsToken: false,
        addSepToken: false,
      ),
    );
    final e = t.encodePreTokenizedPair(['Hello'], ['hello']);
    expect(e.tokens, ['[UNK]', 'hello']);
    expect(e.wordIds, [0, 0]);
    expect(e.sequenceIds, [0, 1]);
    expect(e.charToToken(0, sequenceIndex: 1, wordIndex: 0), 1);
    expect(e.charToWord(0, sequenceIndex: 2), isNull);
    expect(e.charToToken(0, wordIndex: -1), isNull);
    expect(e.wordToTokens(99), isNull);
  });
  final golden = jsonDecode(
    File(
      'test/fixtures/huggingface/pretokenized.golden.json',
    ).readAsStringSync(),
  );
  for (final row in golden['cases']) {
    test('HF pre-tokenized pipeline: ${row['name']}', () async {
      final t = WordPieceTokenizer.fromTokenizerJsonString(
        jsonEncode(row['pipeline']),
      );
      t.addTokens([
        for (final a in row['additions'])
          AddedToken(
            a['content'] as String,
            singleWord: a['single_word'] as bool? ?? false,
            lstrip: a['lstrip'] as bool? ?? false,
            rstrip: a['rstrip'] as bool? ?? false,
            normalized: a['normalized'] as bool?,
            special: a['special'] as bool? ?? false,
          ),
      ]);
      await runHfTokenizerCase(t, row['fixture'] as Map<String, dynamic>);
    });
  }
  test('word-local alignment retains empty input indices and splits', () {
    final e = tokenizer().encodePreTokenized(['', 'Hello,', 'world', 'ab']);
    expect(e.wordIds, [null, 1, 1, 2, 3, 3, null]);
    expect(e.offsets, [(0, 0), (0, 5), (5, 6), (0, 5), (0, 1), (1, 2), (0, 0)]);
    expect(e.charToToken(0), 1);
    expect(e.charToToken(0, wordIndex: 2), 3);
    expect(e.charToWord(1, wordIndex: 3), 3);
    expect(e.charToToken(0, wordIndex: 0), isNull);
    expect(e.charToToken(-1, wordIndex: 1), isNull);
    expect(e.wordToTokens(3), (4, 6));
    expect(e.wordToChars(3), (0, 2));
    expect(e.tokenToWord(5), 3);
    expect(e.tokenToChars(5), (1, 2));
    expect(e.tokenToSequence(5), 0);
  });

  test('added tokens never cross items and retain word identity', () {
    final t = tokenizer()
      ..addTokens([
        const AddedToken('hello world'),
        const AddedToken(
          '<X>',
          singleWord: true,
          lstrip: true,
          rstrip: true,
          normalized: false,
        ),
        const AddedToken('CAFÉ', normalized: true),
      ]);
    final e = t.encodePreTokenized([
      'hello',
      'world',
      'a',
      '  <X>  ',
      'a',
      'Cafe\u0301',
    ]);
    expect(e.tokens, [
      '[CLS]',
      'hello',
      'world',
      'a',
      '  <X>  ',
      'a',
      'cafe',
      '[SEP]',
    ]);
    expect(e.wordIds, [null, 0, 1, 2, 3, 4, 5, null]);
    expect(e.offsets[4], (0, 7));
    expect(e.offsets[6], (0, 4));
    expect(t.encodePreTokenized(['hello world']).tokens, [
      '[CLS]',
      'hello world',
      '[SEP]',
    ]);
  });

  for (final size in [0, 1, 8]) {
    test(
      'parallel snapshot and nested inputs single/pair size $size',
      () async {
        final t = tokenizer()
          ..enablePadding(length: 12)
          ..enableTruncation(maxLength: 8);
        final words = List.generate(size, (_) => ['hello', '', 'ab', 'world']);
        final pairs = [
          for (final w in words) (w, ['world', 'hello']),
        ];
        final expected = t
            .encodePreTokenizedBatch(words)
            .map((e) => e.toMap())
            .toList();
        final expectedPairs = t
            .encodePreTokenizedPairBatch(pairs)
            .map((e) => e.toMap())
            .toList();
        final singleFuture = t.encodePreTokenizedBatchParallel(
          words,
          numWorkers: 2,
        );
        final pairFuture = t.encodePreTokenizedPairBatchParallel(
          pairs,
          numWorkers: 2,
        );
        for (final w in words) {
          w.clear();
        }
        for (final p in pairs) {
          p.$2.clear();
        }
        words.clear();
        pairs.clear();
        t.addTokens([const AddedToken('ab')]);
        t.enablePadding(length: 24);
        t.enableTruncation(maxLength: 3);
        expect((await singleFuture).map((e) => e.toMap()).toList(), expected);
        expect(
          (await pairFuture).map((e) => e.toMap()).toList(),
          expectedPairs,
        );
      },
    );
  }
  test('invalid workers and impossible lengths fail', () async {
    final t = tokenizer();
    await expectLater(
      t.encodePreTokenizedBatchParallel([], numWorkers: 0),
      throwsArgumentError,
    );
    await expectLater(
      t.encodePreTokenizedPairBatchParallel([], numWorkers: -1),
      throwsArgumentError,
    );
    expect(
      () => t.encodePreTokenizedPair(['hello'], ['world'], maxLength: 2),
      throwsArgumentError,
    );
    expect(
      () => t.encodePreTokenizedPair(
        ['hello'],
        ['world', 'world', 'world'],
        maxLength: 4,
        truncationStrategy: TruncationStrategy.onlyFirst,
      ),
      throwsArgumentError,
    );
  });
}

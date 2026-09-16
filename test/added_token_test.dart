import 'dart:convert';
import 'dart:io';

import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
import 'package:test/test.dart';

import 'hf_fixture_support.dart';

AddedToken added(dynamic value) => value is String
    ? AddedToken(value)
    : AddedToken(
        value['content'] as String,
        singleWord: value['single_word'] as bool? ?? false,
        lstrip: value['lstrip'] as bool? ?? false,
        rstrip: value['rstrip'] as bool? ?? false,
        normalized: value['normalized'] as bool?,
        special: value['special'] as bool? ?? false,
      );

void main() {
  final golden =
      jsonDecode(
            File(
              'test/fixtures/huggingface/added_tokens.golden.json',
            ).readAsStringSync(),
          )
          as Map<String, dynamic>;
  for (final c in golden['cases'] as List<dynamic>) {
    test('HF AddedToken: ${c['name']}', () {
      final t = WordPieceTokenizer.fromTokenizerJsonString(
        jsonEncode(c['initial']),
      );
      for (final step in c['steps'] as List<dynamic>) {
        final values = step['tokens'] as List<dynamic>;
        final count = step['method'] == 'add_tokens'
            ? t.addTokens(values.map(added).toList())
            : t.addSpecialTokens(values.cast<String>());
        expect(count, step['count']);
        expectHfEncoding(
          t,
          t.encode(c['input'] as String),
          step['expected'] as Map<String, dynamic>,
        );
        final restored = WordPieceTokenizer.fromTokenizerJsonString(
          jsonEncode(step['pipeline']),
        );
        expectHfEncoding(
          restored,
          restored.encode(c['input'] as String),
          step['reloaded_expected'] as Map<String, dynamic>,
        );
      }
    });
  }

  for (final interaction in golden['interactions'] as List<dynamic>) {
    final fixture = interaction['fixture'] as Map<String, dynamic>;
    test(fixture['name'] as String, () async {
      final t = WordPieceTokenizer.fromTokenizerJsonString(
        jsonEncode(interaction['initial']),
      );
      t.addTokens(
        (interaction['additions'] as List<dynamic>).map(added).toList(),
      );
      await runHfTokenizerCase(t, fixture);
    });
  }

  WordPieceTokenizer tokenizer() => WordPieceTokenizer(
    vocab: Vocabulary.fromTokens([
      '[PAD]',
      '[UNK]',
      '[CLS]',
      '[SEP]',
      '[MASK]',
      'hello',
      'world',
    ]),
  );

  test('Unicode word boundary range endpoints match the HF scalar oracle', () {
    final oracle = jsonDecode(
      File(
        'test/fixtures/huggingface/unicode_word_boundaries.json',
      ).readAsStringSync(),
    );
    final ranges = oracle['ranges'] as List<dynamic>;
    final pipeline = jsonDecode(jsonEncode(golden['cases'][0]['initial']));
    pipeline['normalizer'] = null;
    pipeline['pre_tokenizer'] = null;
    final t = WordPieceTokenizer.fromTokenizerJsonString(jsonEncode(pipeline));
    t.addTokens([const AddedToken('<T>', singleWord: true, normalized: false)]);
    final id = t.vocab.tokenToId('<T>');
    final endpoints = <int>{};
    for (final range in ranges) {
      endpoints.addAll(
        [range[0] - 1, range[0], range[1], range[1] + 1].cast<int>(),
      );
    }
    for (final cp in endpoints.where((cp) => cp >= 0 && cp <= 0x10ffff)) {
      final isWord = ranges.any((r) => r[0] <= cp && cp <= r[1]);
      final char = String.fromCharCode(cp);
      for (final text in ['<T>$char', '$char<T>']) {
        expect(
          t.encode(text).ids.contains(id),
          !isWord,
          reason: 'U+${cp.toRadixString(16)} in $text',
        );
      }
    }
  });

  test('normalized collisions choose lowest ID deterministically', () {
    final t = tokenizer();
    t.addTokens([const AddedToken('CAFÉ'), const AddedToken('café')]);
    expect(t.encode('cafe').ids[1], t.vocab.tokenToId('CAFÉ'));
  });

  test(
    'already absorbed whitespace cannot create a reversed or empty match',
    () {
      final t = tokenizer();
      t.addTokens([
        const AddedToken('<X>', rstrip: true, normalized: false),
        const AddedToken(' ', lstrip: true, normalized: false),
      ]);
      final e = t.encode('<X>   hello');
      expect(e.tokens, ['[CLS]', '<X>   ', 'hello', '[SEP]']);
      expect(e.offsets, [(0, 0), (0, 6), (6, 11), (0, 0)]);
    },
  );

  test('registration does not change the model trie or shared vocabulary', () {
    final t = tokenizer();
    final before = t.vocab;
    final other = WordPieceTokenizer(vocab: before);
    t.addTokens([const AddedToken('z', singleWord: true)]);
    expect(t.encode('zzz').tokens, ['[CLS]', '[UNK]', '[SEP]']);
    expect(before.contains('z'), isFalse);
    expect(other.vocab.contains('z'), isFalse);
    expect(t.vocab.contains('z'), isTrue);
    expect(t.convertIdsToTokens(t.convertTokensToIds(['z'])), ['z']);
  });

  test('sparse ID allocation never replaces a model token', () {
    final t = WordPieceTokenizer(
      vocab: Vocabulary.fromMap({'[UNK]': 0, 'hello': 100}),
    );
    t.addTokens([const AddedToken('new')]);
    expect(t.vocab.tokenToId('new'), 101);
    expect(t.vocab.idToToken(100), 'hello');
  });

  test('overflow registration is atomic', () {
    final t = WordPieceTokenizer(
      vocab: Vocabulary.fromMap({'[UNK]': 0, 'last': 0x7ffffffe}),
    );
    expect(
      () => t.addTokens([const AddedToken('a'), const AddedToken('b')]),
      throwsArgumentError,
    );
    expect(t.vocab.contains('a'), isFalse);
    expect(t.addTokens([const AddedToken('a')]), 1);
    expect(t.vocab.tokenToId('a'), 0x7fffffff);
  });

  test('dynamic special normalized IDs are skipped and cannot be demoted', () {
    final t = tokenizer();
    t.addTokens([const AddedToken('CAFÉ', normalized: true, special: true)]);
    expect(t.decode(t.encode('cafe').ids), '');
    expect(
      t.decode([t.vocab.tokenToId('CAFÉ')], skipSpecialTokens: false),
      'cafe',
    );
    t.addTokens([const AddedToken('CAFÉ', normalized: false)]);
    expect(t.decode([t.vocab.tokenToId('CAFÉ')]), '');
    expect(
      t.decode([t.vocab.tokenToId('CAFÉ')], skipSpecialTokens: false),
      'CAFÉ',
    );
  });

  test(
    'single and pair workers capture registration and padding state',
    () async {
      final t = tokenizer();
      t.addTokens([const AddedToken('<X>', lstrip: true, rstrip: true)]);
      t.enablePadding(padToMultipleOf: 8).enableTruncation(maxLength: 7);
      final texts = List.filled(8, 'hello  <X>  world hello');
      final pairs = texts.map((text) => (text, 'hello <X>')).toList();
      final expected = t.encodeBatch(texts).map((e) => e.toMap()).toList();
      final expectedPairs = t
          .encodePairBatch(pairs)
          .map((e) => e.toMap())
          .toList();
      final pending = t.encodeBatchParallel(texts, numWorkers: 2);
      final pendingPairs = t.encodePairBatchParallel(pairs, numWorkers: 2);
      t.addTokens([
        const AddedToken('hello  <X>  world hello', normalized: false),
      ]);
      t.enablePadding(length: 24);
      expect((await pending).map((e) => e.toMap()).toList(), expected);
      expect(
        (await pendingPairs).map((e) => e.toMap()).toList(),
        expectedPairs,
      );
    },
  );
}

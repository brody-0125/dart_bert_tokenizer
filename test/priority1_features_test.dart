/// Tests for Priority 1 features:
/// - addSpecialTokens parameter
/// - Truncation strategies
/// - vocabularyMap getter
/// - numSpecialTokensToAdd() method
library;

import 'package:test/test.dart';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() {
  late WordPieceTokenizer tokenizer;

  setUpAll(() {
    tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt');
  });

  group('addSpecialTokens parameter', () {
    test('encode with addSpecialTokens=true adds CLS and SEP', () {
      final encoding = tokenizer.encode('hello world', addSpecialTokens: true);

      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));
    });

    test('encode with addSpecialTokens=false omits CLS and SEP', () {
      final encoding = tokenizer.encode('hello world', addSpecialTokens: false);

      expect(encoding.tokens.first, isNot(equals('[CLS]')));
      expect(encoding.tokens.last, isNot(equals('[SEP]')));
      expect(encoding.tokens, contains('hello'));
      expect(encoding.tokens, contains('world'));
    });

    test('encode with addSpecialTokens=null uses config defaults', () {
      final encoding = tokenizer.encode('hello world');

      // Default config has addClsToken=true and addSepToken=true
      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));
    });

    test('encodeBatch with addSpecialTokens=false', () {
      final encodings = tokenizer.encodeBatch([
        'hello',
        'world',
      ], addSpecialTokens: false);

      for (final encoding in encodings) {
        expect(encoding.tokens, isNot(contains('[CLS]')));
        expect(encoding.tokens, isNot(contains('[SEP]')));
      }
    });

    test('encodePair with addSpecialTokens=false', () {
      final encoding = tokenizer.encodePair(
        'question',
        'answer',
        addSpecialTokens: false,
      );

      expect(encoding.tokens, isNot(contains('[CLS]')));
      expect(encoding.tokens, isNot(contains('[SEP]')));
    });

    test('special tokens mask is correct when addSpecialTokens=false', () {
      final encoding = tokenizer.encode('hello', addSpecialTokens: false);

      // No special tokens should be marked
      expect(encoding.specialTokensMask, everyElement(equals(0)));
    });
  });

  group('numSpecialTokensToAdd', () {
    test('returns 2 for single sequence (CLS + SEP)', () {
      expect(tokenizer.numSpecialTokensToAdd(isPair: false), equals(2));
    });

    test('returns 3 for pair (CLS + SEP + SEP)', () {
      expect(tokenizer.numSpecialTokensToAdd(isPair: true), equals(3));
    });

    test('returns 0 when special tokens disabled', () {
      final noSpecialTokenizer = WordPieceTokenizer(
        vocab: tokenizer.vocab,
        config: const WordPieceConfig(addClsToken: false, addSepToken: false),
      );

      expect(
        noSpecialTokenizer.numSpecialTokensToAdd(isPair: false),
        equals(0),
      );
      expect(noSpecialTokenizer.numSpecialTokensToAdd(isPair: true), equals(0));
    });

    test('returns 1 when only CLS enabled', () {
      final clsOnlyTokenizer = WordPieceTokenizer(
        vocab: tokenizer.vocab,
        config: const WordPieceConfig(addClsToken: true, addSepToken: false),
      );

      expect(clsOnlyTokenizer.numSpecialTokensToAdd(isPair: false), equals(1));
      expect(clsOnlyTokenizer.numSpecialTokensToAdd(isPair: true), equals(1));
    });
  });

  group('vocabularyMap', () {
    test('returns complete vocabulary map', () {
      final vocab = tokenizer.vocab.vocabularyMap;

      expect(vocab, isA<Map<String, int>>());
      expect(vocab.length, equals(tokenizer.vocab.size));
    });

    test('vocab map contains special tokens', () {
      final vocab = tokenizer.vocab.vocabularyMap;

      expect(vocab.containsKey('[CLS]'), isTrue);
      expect(vocab.containsKey('[SEP]'), isTrue);
      expect(vocab.containsKey('[PAD]'), isTrue);
      expect(vocab.containsKey('[UNK]'), isTrue);
      expect(vocab.containsKey('[MASK]'), isTrue);
    });

    test('vocab map IDs match tokenToId', () {
      final vocab = tokenizer.vocab.vocabularyMap;

      expect(vocab['[CLS]'], equals(tokenizer.vocab.clsTokenId));
      expect(vocab['[SEP]'], equals(tokenizer.vocab.sepTokenId));
      expect(vocab['[PAD]'], equals(tokenizer.vocab.padTokenId));
    });

    test('vocab map is unmodifiable', () {
      final vocab = tokenizer.vocab.vocabularyMap;

      expect(
        () => vocab['new_token'] = 99999,
        throwsA(isA<UnsupportedError>()),
      );
    });
  });

  group('TruncationStrategy.longestFirst', () {
    test('truncates longer sequence first', () {
      // Create long texts
      final longText = 'word ' * 100; // ~100 words
      const shortText = 'short text';

      final encoding = tokenizer.encodePair(
        longText,
        shortText,
        maxLength: 20,
        truncationStrategy: TruncationStrategy.longestFirst,
      );

      expect(encoding.length, equals(20));
      // Both sequences should be present
      expect(encoding.typeIds, contains(0));
      expect(encoding.typeIds, contains(1));
    });

    test('balances truncation between equal-length sequences', () {
      const textA = 'one two three four five';
      const textB = 'six seven eight nine ten';

      final encoding = tokenizer.encodePair(
        textA,
        textB,
        maxLength: 10,
        truncationStrategy: TruncationStrategy.longestFirst,
      );

      expect(encoding.length, equals(10));
    });
  });

  group('TruncationStrategy.onlyFirst', () {
    test('only truncates first sequence', () {
      final longText = 'word ' * 50;
      const shortText = 'keep this intact';

      final encoding = tokenizer.encodePair(
        longText,
        shortText,
        maxLength: 15,
        truncationStrategy: TruncationStrategy.onlyFirst,
      );

      // Second sequence should be intact
      final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);
      expect(decoded, contains('keep'));
      expect(decoded, contains('intact'));
    });

    test('preserves second sequence completely', () {
      const textA = 'first sequence here';
      const textB = 'second';

      final encodingB = tokenizer.encode('second', addSpecialTokens: false);

      final encoding = tokenizer.encodePair(
        textA,
        textB,
        maxLength: 10,
        truncationStrategy: TruncationStrategy.onlyFirst,
      );

      // Count tokens from second sequence (typeId = 1, excluding [SEP])
      final typeIdOneCount =
          encoding.typeIds.where((t) => t == 1).length -
          1; // Exclude final [SEP]

      expect(typeIdOneCount, equals(encodingB.length));
    });
  });

  group('TruncationStrategy.onlySecond', () {
    test('only truncates second sequence', () {
      const shortText = 'keep this';
      final longText = 'word ' * 50;

      final encoding = tokenizer.encodePair(
        shortText,
        longText,
        maxLength: 15,
        truncationStrategy: TruncationStrategy.onlySecond,
      );

      // First sequence should be intact
      final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);
      expect(decoded, contains('keep'));
    });

    test('preserves first sequence completely', () {
      const textA = 'first';
      const textB = 'second sequence here very long';

      final encodingA = tokenizer.encode('first', addSpecialTokens: false);

      final encoding = tokenizer.encodePair(
        textA,
        textB,
        maxLength: 10,
        truncationStrategy: TruncationStrategy.onlySecond,
      );

      // Count tokens from first sequence (typeId = 0, excluding [CLS] and first [SEP])
      final typeIdZeroCount =
          encoding.typeIds.where((t) => t == 0).length -
          2; // Exclude [CLS] and [SEP]

      expect(typeIdZeroCount, equals(encodingA.length));
    });
  });

  group('TruncationStrategy.doNotTruncate', () {
    test('does not truncate even when exceeding maxLength', () {
      const textA = 'first sequence';
      const textB = 'second sequence';

      final encodingNoTrunc = tokenizer.encodePair(
        textA,
        textB,
        maxLength: 5,
        truncationStrategy: TruncationStrategy.doNotTruncate,
      );

      final encodingNormal = tokenizer.encodePair(textA, textB);

      expect(encodingNoTrunc.length, equals(encodingNormal.length));
    });
  });

  group('Encoding.truncatePair static method', () {
    test('handles empty result when maxLength too small', () {
      final encodingA = tokenizer.encode('hello', addSpecialTokens: false);
      final encodingB = tokenizer.encode('world', addSpecialTokens: false);

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 2, // Only room for special tokens
        numSpecialTokens: 3,
      );

      expect(truncA.isEmpty, isTrue);
      expect(truncB.isEmpty, isTrue);
    });

    test('returns unchanged when under maxLength', () {
      final encodingA = tokenizer.encode('hi', addSpecialTokens: false);
      final encodingB = tokenizer.encode('bye', addSpecialTokens: false);

      final (truncA, truncB) = Encoding.truncatePair(
        encodingA: encodingA,
        encodingB: encodingB,
        maxLength: 100,
        numSpecialTokens: 3,
      );

      expect(truncA.length, equals(encodingA.length));
      expect(truncB.length, equals(encodingB.length));
    });
  });

  group('encodePairBatch with truncation', () {
    test('applies truncation to all pairs', () {
      final pairs = [
        ('question one is here', 'answer one is here too'),
        ('question two', 'answer two'),
        ('q three ' * 10, 'a three ' * 10),
      ];

      final encodings = tokenizer.encodePairBatch(
        pairs,
        maxLength: 15,
        truncationStrategy: TruncationStrategy.longestFirst,
      );

      for (final encoding in encodings) {
        expect(encoding.length, lessThanOrEqualTo(15));
      }
    });

    test('applies addSpecialTokens to all pairs', () {
      final pairs = [('a', 'b'), ('c', 'd')];

      final encodings = tokenizer.encodePairBatch(
        pairs,
        addSpecialTokens: false,
      );

      for (final encoding in encodings) {
        expect(encoding.tokens, isNot(contains('[CLS]')));
        expect(encoding.tokens, isNot(contains('[SEP]')));
      }
    });
  });

  group('Integration tests', () {
    test('full pipeline: encode without special, truncate, then decode', () {
      const text = 'This is a test sentence for tokenization';

      // Encode without special tokens
      final encoding = tokenizer.encode(text, addSpecialTokens: false);

      // Truncate
      final truncated = encoding.withTruncation(maxLength: 5);

      // Decode
      final decoded = tokenizer.decode(truncated.ids, skipSpecialTokens: true);

      expect(truncated.length, equals(5));
      expect(decoded.isNotEmpty, isTrue);
    });

    test('pair encoding respects maxLength exactly', () {
      final textA = 'word ' * 20;
      final textB = 'text ' * 20;

      for (final maxLen in [10, 20, 50, 100]) {
        final encoding = tokenizer.encodePair(
          textA,
          textB,
          maxLength: maxLen,
          truncationStrategy: TruncationStrategy.longestFirst,
        );

        expect(
          encoding.length,
          lessThanOrEqualTo(maxLen),
          reason: 'Should respect maxLength=$maxLen',
        );
      }
    });
  });
}

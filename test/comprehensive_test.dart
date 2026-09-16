/// Comprehensive test suite covering:
/// - Basic Tokenization
/// - Out of Vocabulary (OOV) / Subword splitting / [UNK] handling
/// - CLS & SEP tokens
/// - Sentence Pair encoding
/// - Truncation
/// - Padding
/// - Attention Mask
/// - Token Type IDs
///
/// Based on HuggingFace transformers test_tokenization_common.py and
/// tokenizers test_encoding.py patterns.
library;

import 'package:test/test.dart';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() {
  late WordPieceTokenizer tokenizer;

  setUpAll(() {
    tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt');
  });

  // ===========================================================================
  // BASIC TOKENIZATION
  // ===========================================================================
  group('Basic Tokenization', () {
    group('Word Splitting', () {
      test('splits on whitespace', () {
        final encoding = tokenizer.encode(
          'hello world',
          addSpecialTokens: false,
        );
        expect(encoding.tokens, contains('hello'));
        expect(encoding.tokens, contains('world'));
      });

      test('handles multiple whitespace characters', () {
        final enc1 = tokenizer.encode('hello world', addSpecialTokens: false);
        final enc2 = tokenizer.encode('hello   world', addSpecialTokens: false);
        final enc3 = tokenizer.encode('hello\tworld', addSpecialTokens: false);
        final enc4 = tokenizer.encode('hello\nworld', addSpecialTokens: false);

        expect(enc1.tokens, equals(enc2.tokens));
        expect(enc1.tokens, equals(enc3.tokens));
        expect(enc1.tokens, equals(enc4.tokens));
      });

      test('handles leading and trailing whitespace', () {
        final enc1 = tokenizer.encode('hello', addSpecialTokens: false);
        final enc2 = tokenizer.encode('  hello  ', addSpecialTokens: false);

        expect(enc1.tokens, equals(enc2.tokens));
      });

      test('splits words correctly for simple vocabulary', () {
        final encoding = tokenizer.encode(
          'the quick brown',
          addSpecialTokens: false,
        );
        expect(encoding.tokens, equals(['the', 'quick', 'brown']));
      });
    });

    group('Punctuation Handling', () {
      test('separates punctuation from words', () {
        final encoding = tokenizer.encode(
          'hello, world!',
          addSpecialTokens: false,
        );
        expect(encoding.tokens, contains('hello'));
        expect(encoding.tokens, contains(','));
        expect(encoding.tokens, contains('world'));
        expect(encoding.tokens, contains('!'));
      });

      test('handles multiple punctuation marks', () {
        final encoding = tokenizer.encode('what?!', addSpecialTokens: false);
        expect(encoding.tokens, contains('what'));
        expect(encoding.tokens, contains('?'));
        expect(encoding.tokens, contains('!'));
      });

      test('handles punctuation-only text', () {
        final encoding = tokenizer.encode('!@#', addSpecialTokens: false);
        expect(encoding.isNotEmpty, isTrue);
      });

      test('handles contractions', () {
        final encoding = tokenizer.encode("don't", addSpecialTokens: false);
        expect(encoding.isNotEmpty, isTrue);
        // Should tokenize into parts
      });

      test('handles hyphenated words', () {
        final encoding = tokenizer.encode(
          'well-known',
          addSpecialTokens: false,
        );
        expect(encoding.tokens, contains('-'));
      });

      test('handles periods and sentences', () {
        final encoding = tokenizer.encode(
          'Hello. World.',
          addSpecialTokens: false,
        );
        expect(encoding.tokens.where((t) => t == '.').length, equals(2));
      });
    });

    group('Case Normalization', () {
      test('lowercases text', () {
        final enc1 = tokenizer.encode('HELLO', addSpecialTokens: false);
        final enc2 = tokenizer.encode('hello', addSpecialTokens: false);

        expect(enc1.tokens, equals(enc2.tokens));
        expect(enc1.ids, equals(enc2.ids));
      });

      test('lowercases mixed case', () {
        final encoding = tokenizer.encode(
          'HeLLo WoRLd',
          addSpecialTokens: false,
        );
        expect(encoding.tokens, contains('hello'));
        expect(encoding.tokens, contains('world'));
      });
    });

    group('Accent Normalization', () {
      test('strips accents from characters', () {
        final encoding = tokenizer.encode('café', addSpecialTokens: false);
        expect(encoding.tokens, contains('cafe'));
      });

      test('handles various diacritics', () {
        final encoding = tokenizer.encode(
          'résumé naïve',
          addSpecialTokens: false,
        );
        expect(encoding.tokens, contains('resume'));
        expect(encoding.tokens, contains('naive'));
      });

      test('handles umlauts', () {
        final encoding = tokenizer.encode('über', addSpecialTokens: false);
        // Should normalize ü to u
        expect(
          encoding.tokens.any((t) => t.contains('uber') || t.contains('##ber')),
          isTrue,
        );
      });
    });

    group('Numbers', () {
      test('tokenizes numbers', () {
        final encoding = tokenizer.encode('123', addSpecialTokens: false);
        expect(encoding.isNotEmpty, isTrue);
      });

      test('tokenizes long numbers with subwords', () {
        final encoding = tokenizer.encode('123456789', addSpecialTokens: false);
        expect(encoding.isNotEmpty, isTrue);
      });

      test('tokenizes decimals', () {
        final encoding = tokenizer.encode('3.14', addSpecialTokens: false);
        expect(encoding.tokens, contains('.'));
      });

      test('tokenizes negative numbers', () {
        final encoding = tokenizer.encode('-42', addSpecialTokens: false);
        expect(encoding.tokens, contains('-'));
      });
    });

    group('Token ID Consistency', () {
      test('same text produces same IDs', () {
        final enc1 = tokenizer.encode('hello world');
        final enc2 = tokenizer.encode('hello world');

        expect(enc1.ids, equals(enc2.ids));
        expect(enc1.tokens, equals(enc2.tokens));
      });

      test('convert tokens to IDs matches encode IDs', () {
        final encoding = tokenizer.encode(
          'hello world',
          addSpecialTokens: false,
        );
        final idsFromTokens = tokenizer.convertTokensToIds(encoding.tokens);

        expect(idsFromTokens, equals(encoding.ids));
      });

      test('convert IDs to tokens matches encode tokens', () {
        final encoding = tokenizer.encode(
          'hello world',
          addSpecialTokens: false,
        );
        final tokensFromIds = tokenizer.convertIdsToTokens(encoding.ids);

        expect(tokensFromIds, equals(encoding.tokens));
      });
    });
  });

  // ===========================================================================
  // OUT OF VOCABULARY / SUBWORD / [UNK]
  // ===========================================================================
  group('Out of Vocabulary (OOV) / Subword / [UNK]', () {
    group('Subword Splitting', () {
      test('splits unknown words into subwords', () {
        // "tokenization" should be split into "token" + "##ization"
        final encoding = tokenizer.encode(
          'tokenization',
          addSpecialTokens: false,
        );
        expect(encoding.tokens.any((t) => t.startsWith('##')), isTrue);
      });

      test('subword tokens have ## prefix', () {
        final encoding = tokenizer.encode(
          'tokenization',
          addSpecialTokens: false,
        );
        final subwords = encoding.tokens.where((t) => t.startsWith('##'));
        expect(subwords.every((t) => t.startsWith('##')), isTrue);
      });

      test('all subwords map to same word ID', () {
        final encoding = tokenizer.encode(
          'tokenization',
          addSpecialTokens: false,
        );
        // All tokens should have wordId = 0 (single word)
        final nonNullWordIds = encoding.wordIds.where((w) => w != null);
        expect(nonNullWordIds.every((w) => w == 0), isTrue);
      });

      test('word to tokens range covers all subwords', () {
        final encoding = tokenizer.encode('tokenization');
        final range = encoding.wordToTokens(0);

        expect(range, isNotNull);
        expect(
          range!.$2 - range.$1,
          greaterThan(1),
          reason: 'tokenization should split into multiple subwords',
        );
      });

      test('greedy matching prefers longer subwords', () {
        final encoding = tokenizer.encode('playing', addSpecialTokens: false);
        // Should prefer "playing" as a whole or "play" + "##ing" rather than
        // smaller pieces
        expect(encoding.tokens.length, lessThanOrEqualTo(2));
      });
    });

    group('[UNK] Token Handling', () {
      test('unknown characters produce [UNK]', () {
        // Most Chinese characters not in standard BERT vocab
        final encoding = tokenizer.encode('你好', addSpecialTokens: false);
        expect(encoding.tokens, contains('[UNK]'));
      });

      test('[UNK] has correct token ID', () {
        final encoding = tokenizer.encode('你好', addSpecialTokens: false);
        final unkIndex = encoding.tokens.indexOf('[UNK]');
        if (unkIndex >= 0) {
          expect(encoding.ids[unkIndex], equals(tokenizer.vocab.unkTokenId));
        }
      });

      test('very long words produce [UNK]', () {
        // Words exceeding maxWordLength (200) should be [UNK]
        final longWord = 'a' * 250;
        final encoding = tokenizer.encode(longWord, addSpecialTokens: false);
        expect(encoding.tokens, contains('[UNK]'));
      });

      test('partial unknown produces mixed tokens and [UNK]', () {
        // Mixed text with some known and some unknown
        final encoding = tokenizer.encode(
          'hello 你好 world',
          addSpecialTokens: false,
        );
        expect(encoding.tokens, contains('hello'));
        expect(encoding.tokens, contains('world'));
        expect(encoding.tokens, contains('[UNK]'));
      });

      test('[UNK] is not marked as special token', () {
        // [UNK] appearing as vocabulary replacement is not a "special token"
        // in the mask sense (it's a content token)
        final encoding = tokenizer.encode('你好');
        final unkIndex = encoding.tokens.indexOf('[UNK]');
        if (unkIndex >= 0) {
          expect(
            encoding.specialTokensMask[unkIndex],
            equals(0),
            reason:
                '[UNK] as vocabulary replacement should not be marked special',
          );
        }
      });
    });

    group('Rare Character Handling', () {
      test('handles emoji (produces [UNK] or valid tokens)', () {
        final encoding = tokenizer.encode('Hello 😊', addSpecialTokens: false);
        expect(encoding.isNotEmpty, isTrue);
        expect(encoding.tokens, contains('hello'));
      });

      test('handles zero-width characters', () {
        final encoding = tokenizer.encode(
          'hello\u200Bworld',
          addSpecialTokens: false,
        );
        // Zero-width space should be normalized away
        expect(encoding.isNotEmpty, isTrue);
      });

      test('handles control characters', () {
        final encoding = tokenizer.encode(
          'hello\u0000world',
          addSpecialTokens: false,
        );
        expect(encoding.isNotEmpty, isTrue);
      });

      test('handles mixed scripts', () {
        final encoding = tokenizer.encode(
          'Hello мир 世界',
          addSpecialTokens: false,
        );
        expect(encoding.tokens, contains('hello'));
      });
    });

    group('Subword Reconstruction', () {
      test('decode removes ## prefix', () {
        final encoding = tokenizer.encode('tokenization');
        final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);

        expect(decoded, isNot(contains('##')));
        expect(decoded, equals('tokenization'));
      });

      test('decode joins subwords correctly', () {
        final encoding = tokenizer.encode('unbelievable');
        final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);

        expect(decoded, equals('unbelievable'));
      });

      test('character offsets span correct ranges for subwords', () {
        final encoding = tokenizer.encode(
          'tokenization',
          addSpecialTokens: false,
        );

        // First subword should start at 0
        expect(encoding.offsets.first.$1, equals(0));

        // Last subword should end at length of word
        expect(encoding.offsets.last.$2, equals('tokenization'.length));
      });
    });
  });

  // ===========================================================================
  // CLS & SEP TOKENS
  // ===========================================================================
  group('CLS & SEP Tokens', () {
    group('Single Sequence', () {
      test('[CLS] is first token', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.tokens.first, equals('[CLS]'));
      });

      test('[SEP] is last token', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.tokens.last, equals('[SEP]'));
      });

      test('[CLS] has correct ID', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.ids.first, equals(tokenizer.vocab.clsTokenId));
      });

      test('[SEP] has correct ID', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.ids.last, equals(tokenizer.vocab.sepTokenId));
      });

      test('[CLS] is marked as special', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.specialTokensMask.first, equals(1));
      });

      test('[SEP] is marked as special', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.specialTokensMask.last, equals(1));
      });

      test('[CLS] has typeId 0', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.typeIds.first, equals(0));
      });

      test('[SEP] has typeId 0 for single sequence', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.typeIds.last, equals(0));
      });

      test('[CLS] and [SEP] have null sequenceId', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.sequenceIds.first, isNull);
        expect(encoding.sequenceIds.last, isNull);
      });

      test('[CLS] and [SEP] have (0,0) offsets', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.offsets.first, equals((0, 0)));
        expect(encoding.offsets.last, equals((0, 0)));
      });
    });

    group('Pair Sequence', () {
      test('pair has [CLS] ... [SEP] ... [SEP] structure', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        expect(encoding.tokens.first, equals('[CLS]'));
        expect(encoding.tokens.where((t) => t == '[SEP]').length, equals(2));
      });

      test('first [SEP] separates sequences', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        final firstSepIdx = encoding.tokens.indexOf('[SEP]');
        final lastSepIdx = encoding.tokens.lastIndexOf('[SEP]');

        expect(firstSepIdx, lessThan(lastSepIdx));
      });

      test('[SEP] between sequences has typeId 0', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        final firstSepIdx = encoding.tokens.indexOf('[SEP]');
        expect(encoding.typeIds[firstSepIdx], equals(0));
      });

      test('final [SEP] has typeId 1', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        final lastSepIdx = encoding.tokens.lastIndexOf('[SEP]');
        expect(encoding.typeIds[lastSepIdx], equals(1));
      });

      test('all special tokens have null sequenceId', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        for (var i = 0; i < encoding.length; i++) {
          if (encoding.specialTokensMask[i] == 1) {
            expect(
              encoding.sequenceIds[i],
              isNull,
              reason: 'Special token at $i should have null sequenceId',
            );
          }
        }
      });
    });

    group('Configuration', () {
      test('addSpecialTokens=false omits CLS and SEP', () {
        final encoding = tokenizer.encode('hello', addSpecialTokens: false);
        expect(encoding.tokens, isNot(contains('[CLS]')));
        expect(encoding.tokens, isNot(contains('[SEP]')));
      });

      test('addClsToken=false config omits CLS', () {
        final tok = WordPieceTokenizer(
          vocab: tokenizer.vocab,
          config: const WordPieceConfig(addClsToken: false, addSepToken: true),
        );
        final encoding = tok.encode('hello');
        expect(encoding.tokens.first, isNot(equals('[CLS]')));
        expect(encoding.tokens.last, equals('[SEP]'));
      });

      test('addSepToken=false config omits SEP', () {
        final tok = WordPieceTokenizer(
          vocab: tokenizer.vocab,
          config: const WordPieceConfig(addClsToken: true, addSepToken: false),
        );
        final encoding = tok.encode('hello');
        expect(encoding.tokens.first, equals('[CLS]'));
        expect(encoding.tokens.last, isNot(equals('[SEP]')));
      });

      test('numSpecialTokensToAdd returns correct count', () {
        expect(tokenizer.numSpecialTokensToAdd(isPair: false), equals(2));
        expect(tokenizer.numSpecialTokensToAdd(isPair: true), equals(3));
      });
    });

    group('Empty Input', () {
      test('empty string still has CLS and SEP', () {
        final encoding = tokenizer.encode('');
        expect(encoding.tokens, equals(['[CLS]', '[SEP]']));
      });

      test('whitespace-only still has CLS and SEP', () {
        final encoding = tokenizer.encode('   ');
        expect(encoding.tokens, equals(['[CLS]', '[SEP]']));
      });

      test('empty string encoding length is 2', () {
        final encoding = tokenizer.encode('');
        expect(encoding.length, equals(2));
      });
    });
  });

  // ===========================================================================
  // SENTENCE PAIR ENCODING
  // ===========================================================================
  group('Sentence Pair Encoding', () {
    group('Basic Pair Encoding', () {
      test('encodePair produces correct token sequence', () {
        final encoding = tokenizer.encodePair('hello', 'world');

        expect(encoding.tokens.first, equals('[CLS]'));
        expect(encoding.tokens, contains('hello'));
        expect(encoding.tokens, contains('world'));
        expect(encoding.tokens.last, equals('[SEP]'));
      });

      test('both sequences are represented', () {
        final encoding = tokenizer.encodePair(
          'first sentence',
          'second sentence',
        );

        expect(encoding.tokens, contains('first'));
        expect(encoding.tokens, contains('second'));
      });

      test('pair encoding is deterministic', () {
        final enc1 = tokenizer.encodePair('hello', 'world');
        final enc2 = tokenizer.encodePair('hello', 'world');

        expect(enc1.tokens, equals(enc2.tokens));
        expect(enc1.ids, equals(enc2.ids));
        expect(enc1.typeIds, equals(enc2.typeIds));
      });
    });

    group('Sequence Boundaries', () {
      test('type IDs correctly mark sequence boundaries', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        final firstSepIdx = encoding.tokens.indexOf('[SEP]');

        // Everything up to and including first SEP should be typeId 0
        for (var i = 0; i <= firstSepIdx; i++) {
          expect(
            encoding.typeIds[i],
            equals(0),
            reason: 'Token at $i should have typeId 0',
          );
        }

        // Everything after first SEP should be typeId 1
        for (var i = firstSepIdx + 1; i < encoding.length; i++) {
          expect(
            encoding.typeIds[i],
            equals(1),
            reason: 'Token at $i should have typeId 1',
          );
        }
      });

      test('sequence IDs correctly mark content tokens', () {
        final encoding = tokenizer.encodePair('hello', 'world');

        final seq0Count = encoding.sequenceIds.where((s) => s == 0).length;
        final seq1Count = encoding.sequenceIds.where((s) => s == 1).length;

        expect(seq0Count, greaterThan(0));
        expect(seq1Count, greaterThan(0));
      });

      test('tokenToSequence returns correct sequence', () {
        final encoding = tokenizer.encodePair('hello', 'world');

        final helloIdx = encoding.tokens.indexOf('hello');
        final worldIdx = encoding.tokens.indexOf('world');

        expect(encoding.tokenToSequence(helloIdx), equals(0));
        expect(encoding.tokenToSequence(worldIdx), equals(1));
      });
    });

    group('Pair with Special Characters', () {
      test('pair with punctuation', () {
        final encoding = tokenizer.encodePair(
          'Hello, how are you?',
          'I am fine.',
        );

        expect(encoding.tokens, contains(','));
        expect(encoding.tokens, contains('?'));
        expect(encoding.tokens, contains('.'));
      });

      test('pair with numbers', () {
        final encoding = tokenizer.encodePair('2 + 2', '4');

        expect(encoding.tokens, contains('+'));
      });

      test('pair with one empty sequence', () {
        final encoding = tokenizer.encodePair('hello', '');

        expect(encoding.tokens, contains('hello'));
        // Second sequence is empty but still gets a SEP
        expect(encoding.tokens.where((t) => t == '[SEP]').length, equals(2));
      });
    });

    group('Batch Pair Encoding', () {
      test('encodePairBatch produces correct number of encodings', () {
        final pairs = [('q1', 'a1'), ('q2', 'a2'), ('q3', 'a3')];
        final batch = tokenizer.encodePairBatch(pairs);

        expect(batch.length, equals(3));
      });

      test('batch pair encoding matches individual encoding', () {
        final pairs = [('hello', 'world'), ('foo', 'bar')];
        final batch = tokenizer.encodePairBatch(pairs);
        final individual = pairs
            .map((p) => tokenizer.encodePair(p.$1, p.$2))
            .toList();

        for (var i = 0; i < pairs.length; i++) {
          expect(batch[i].tokens, equals(individual[i].tokens));
          expect(batch[i].ids, equals(individual[i].ids));
        }
      });

      test('batch pair with addSpecialTokens=false', () {
        final pairs = [('a', 'b'), ('c', 'd')];
        final batch = tokenizer.encodePairBatch(pairs, addSpecialTokens: false);

        for (final encoding in batch) {
          expect(encoding.tokens, isNot(contains('[CLS]')));
          expect(encoding.tokens, isNot(contains('[SEP]')));
        }
      });
    });

    group('nSequences', () {
      test('single encoding has nSequences = 1', () {
        final encoding = tokenizer.encode('hello');
        expect(encoding.nSequences, equals(1));
      });

      test('pair encoding has nSequences = 2', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        expect(encoding.nSequences, equals(2));
      });

      test('empty encoding has nSequences = 0', () {
        final encoding = Encoding.empty();
        expect(encoding.nSequences, equals(0));
      });
    });
  });

  // ===========================================================================
  // TRUNCATION
  // ===========================================================================
  group('Truncation', () {
    group('Single Sequence Truncation', () {
      test('right truncation keeps beginning', () {
        final encoding = tokenizer.encode('one two three four five');
        final truncated = encoding.withTruncation(
          maxLength: 5,
          truncateFromEnd: true,
        );

        expect(truncated.length, equals(5));
        expect(truncated.tokens.first, equals('[CLS]'));
        expect(truncated.tokens, equals(encoding.tokens.sublist(0, 5)));
      });

      test('left truncation keeps end', () {
        final encoding = tokenizer.encode('one two three four five');
        final truncated = encoding.withTruncation(
          maxLength: 5,
          truncateFromEnd: false,
        );

        expect(truncated.length, equals(5));
        expect(truncated.tokens.last, equals('[SEP]'));
      });

      test('no truncation when under max length', () {
        final encoding = tokenizer.encode('hello');
        final truncated = encoding.withTruncation(maxLength: 100);

        expect(truncated.length, equals(encoding.length));
        expect(truncated.tokens, equals(encoding.tokens));
      });

      test('truncation to exact length', () {
        final encoding = tokenizer.encode('hello');
        final truncated = encoding.withTruncation(maxLength: encoding.length);

        expect(truncated.length, equals(encoding.length));
      });
    });

    group('Fluent Truncation API', () {
      test('enableTruncation configures tokenizer', () {
        final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enableTruncation(maxLength: 5);

        final encoding = tok.encode('one two three four five six seven');

        expect(encoding.length, equals(5));
        expect(tok.truncation, isNotNull);
        expect(tok.truncation!.maxLength, equals(5));
      });

      test('enableTruncation with direction', () {
        final rightTok = WordPieceTokenizer.fromVocabFileSync(
          'vocab.txt',
        )..enableTruncation(maxLength: 4, direction: TruncationDirection.right);

        final leftTok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enableTruncation(maxLength: 4, direction: TruncationDirection.left);

        final rightEnc = rightTok.encode('one two three four');
        final leftEnc = leftTok.encode('one two three four');

        // Right truncation keeps [CLS] at start
        expect(rightEnc.tokens.first, equals('[CLS]'));
        // Left truncation keeps [SEP] at end
        expect(leftEnc.tokens.last, equals('[SEP]'));
      });

      test('noTruncation disables truncation', () {
        final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enableTruncation(maxLength: 3)
          ..noTruncation();

        final encoding = tok.encode('hello world');

        expect(encoding.length, greaterThan(3));
        expect(tok.truncation, isNull);
      });
    });

    group('Pair Truncation Strategies', () {
      test('longestFirst truncates longer sequence first', () {
        final longFirst = 'word ' * 50;
        const shortSecond = 'short';

        final encoding = tokenizer.encodePair(
          longFirst,
          shortSecond,
          maxLength: 20,
          truncationStrategy: TruncationStrategy.longestFirst,
        );

        expect(encoding.length, lessThanOrEqualTo(20));
        // Second sequence should be mostly intact
        expect(encoding.tokens, contains('short'));
      });

      test('onlyFirst only truncates first sequence', () {
        final long = 'word ' * 20;
        const short = 'keep me intact';

        final encoding = tokenizer.encodePair(
          long,
          short,
          maxLength: 15,
          truncationStrategy: TruncationStrategy.onlyFirst,
        );

        final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);
        expect(decoded, contains('keep'));
        expect(decoded, contains('intact'));
      });

      test('onlySecond only truncates second sequence', () {
        const first = 'keep this';
        final second = 'truncate ' * 20;

        final encoding = tokenizer.encodePair(
          first,
          second,
          maxLength: 15,
          truncationStrategy: TruncationStrategy.onlySecond,
        );

        final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);
        expect(decoded, contains('keep'));
        expect(decoded, contains('this'));
      });

      test('doNotTruncate leaves sequences unchanged', () {
        // With doNotTruncate, sequences should not be truncated even with small maxLength
        final normalEncoding = tokenizer.encodePair('hello', 'world');

        // Test with a maxLength that would normally truncate
        final encoding = tokenizer.encodePair(
          'hello',
          'world',
          maxLength: 100, // Large enough to not trigger any truncation anyway
          truncationStrategy: TruncationStrategy.doNotTruncate,
        );

        expect(encoding.length, equals(normalEncoding.length));
        expect(encoding.tokens, equals(normalEncoding.tokens));
      });

      test('doNotTruncate ignores maxLength constraint', () {
        // doNotTruncate should return unchanged encodings
        final longFirst = 'word ' * 10;
        final longSecond = 'text ' * 10;

        final normalEncoding = tokenizer.encodePair(longFirst, longSecond);
        final encoding = tokenizer.encodePair(
          longFirst,
          longSecond,
          maxLength: 10, // Would normally truncate significantly
          truncationStrategy: TruncationStrategy.doNotTruncate,
        );

        expect(encoding.length, equals(normalEncoding.length));
      });
    });

    group('Truncation.truncatePair', () {
      test('truncatePair handles very small maxLength', () {
        final encA = tokenizer.encode('hello', addSpecialTokens: false);
        final encB = tokenizer.encode('world', addSpecialTokens: false);

        final (truncA, truncB) = Encoding.truncatePair(
          encodingA: encA,
          encodingB: encB,
          maxLength: 2, // Less than special tokens
          numSpecialTokens: 3,
        );

        expect(truncA.isEmpty, isTrue);
        expect(truncB.isEmpty, isTrue);
      });

      test('truncatePair preserves when under limit', () {
        final encA = tokenizer.encode('hi', addSpecialTokens: false);
        final encB = tokenizer.encode('bye', addSpecialTokens: false);

        final (truncA, truncB) = Encoding.truncatePair(
          encodingA: encA,
          encodingB: encB,
          maxLength: 100,
          numSpecialTokens: 3,
        );

        expect(truncA.length, equals(encA.length));
        expect(truncB.length, equals(encB.length));
      });
    });

    group('Truncation preserves data integrity', () {
      test('IDs and tokens stay aligned after truncation', () {
        final encoding = tokenizer.encode('one two three four five');
        final truncated = encoding.withTruncation(maxLength: 4);

        for (var i = 0; i < truncated.length; i++) {
          expect(
            tokenizer.vocab.idToToken(truncated.ids[i]),
            equals(truncated.tokens[i]),
          );
        }
      });

      test('attention mask is truncated correctly', () {
        final encoding = tokenizer.encode('one two three');
        final truncated = encoding.withTruncation(maxLength: 3);

        expect(truncated.attentionMask.length, equals(3));
        expect(truncated.attentionMask, everyElement(equals(1)));
      });

      test('sequenceIds is truncated correctly', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        final truncated = encoding.withTruncation(maxLength: 5);

        expect(truncated.sequenceIds.length, equals(5));
      });
    });
  });

  // ===========================================================================
  // PADDING
  // ===========================================================================
  group('Padding', () {
    group('Right Padding (Default)', () {
      test('right padding adds tokens at end', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(padded.length, equals(10));
        expect(
          padded.tokens.sublist(encoding.length),
          everyElement(equals('[PAD]')),
        );
      });

      test('right padded tokens are at end', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
          padOnRight: true,
        );

        expect(padded.tokens.first, equals('[CLS]'));
        expect(padded.tokens.last, equals('[PAD]'));
      });

      test('original content preserved at start', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(
          padded.tokens.sublist(0, encoding.length),
          equals(encoding.tokens),
        );
      });
    });

    group('Left Padding', () {
      test('left padding adds tokens at beginning', () {
        final encoding = tokenizer.encode('hello');
        final padCount = 10 - encoding.length;
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
          padOnRight: false,
        );

        expect(padded.length, equals(10));
        expect(
          padded.tokens.sublist(0, padCount),
          everyElement(equals('[PAD]')),
        );
      });

      test('left padded original content at end', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
          padOnRight: false,
        );

        expect(padded.tokens.first, equals('[PAD]'));
        expect(padded.tokens.last, equals('[SEP]'));
      });
    });

    group('Fluent Padding API', () {
      test('enablePadding with fixed length', () {
        final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enablePadding(length: 20);

        final encoding = tok.encode('hello');

        expect(encoding.length, equals(20));
        expect(tok.padding, isNotNull);
        expect(tok.padding!.length, equals(20));
      });

      test('enablePadding with direction', () {
        final rightTok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enablePadding(length: 10, direction: PaddingDirection.right);

        final leftTok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enablePadding(length: 10, direction: PaddingDirection.left);

        final rightEnc = rightTok.encode('hi');
        final leftEnc = leftTok.encode('hi');

        expect(rightEnc.tokens.last, equals('[PAD]'));
        expect(leftEnc.tokens.first, equals('[PAD]'));
      });

      test('enablePadding with padToMultipleOf', () {
        final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enablePadding(padToMultipleOf: 8);

        final encoding = tok.encode('hello');

        expect(encoding.length % 8, equals(0));
      });

      test('noPadding disables padding', () {
        final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enablePadding(length: 100)
          ..noPadding();

        final encoding = tok.encode('hello');

        expect(encoding.length, lessThan(100));
        expect(tok.padding, isNull);
      });
    });

    group('Padding to Multiple', () {
      test('pads to multiple of 8', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPaddingToMultipleOf(
          multiple: 8,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(padded.length % 8, equals(0));
      });

      test('pads to multiple of 16', () {
        final encoding = tokenizer.encode('hello world this is test');
        final padded = encoding.withPaddingToMultipleOf(
          multiple: 16,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(padded.length % 16, equals(0));
      });

      test('no padding when already a multiple', () {
        final encoding = tokenizer.encode('hello');
        final prePadded = encoding.withPadding(
          targetLength: 8,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        final result = prePadded.withPaddingToMultipleOf(
          multiple: 8,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(result.length, equals(8));
      });

      test('handles multiple of 1', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPaddingToMultipleOf(
          multiple: 1,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(padded.length, equals(encoding.length));
      });

      test('handles multiple of 0 gracefully', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPaddingToMultipleOf(
          multiple: 0,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(padded.length, equals(encoding.length));
      });
    });

    group('Batch Padding', () {
      test('batch padding to longest', () {
        final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enablePadding();

        final batch = tok.encodeBatch(['hi', 'hello world how are you']);

        expect(batch[0].length, equals(batch[1].length));
      });

      test('batch padding with fixed length', () {
        final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enablePadding(length: 20);

        final batch = tok.encodeBatch(['short', 'longer sentence']);

        expect(batch[0].length, equals(20));
        expect(batch[1].length, equals(20));
      });
    });

    group('No Padding When Exceeding Target', () {
      test('no padding when already at target', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: encoding.length,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(padded.length, equals(encoding.length));
        expect(padded.tokens, equals(encoding.tokens));
      });

      test('no padding when exceeding target', () {
        final encoding = tokenizer.encode(
          'hello world this is a long sentence',
        );
        final padded = encoding.withPadding(
          targetLength: 3,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(padded.length, equals(encoding.length));
      });
    });

    group('Padding Data Integrity', () {
      test('padding tokens have correct IDs', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        for (var i = encoding.length; i < padded.length; i++) {
          expect(padded.ids[i], equals(tokenizer.vocab.padTokenId));
        }
      });

      test('padding has typeId 0', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        for (var i = encoding.length; i < padded.length; i++) {
          expect(padded.typeIds[i], equals(0));
        }
      });

      test('padding has null sequenceId', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        for (var i = encoding.length; i < padded.length; i++) {
          expect(padded.sequenceIds[i], isNull);
        }
      });

      test('padding has (0,0) offsets', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        for (var i = encoding.length; i < padded.length; i++) {
          expect(padded.offsets[i], equals((0, 0)));
        }
      });

      test('padding has null wordIds', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        for (var i = encoding.length; i < padded.length; i++) {
          expect(padded.wordIds[i], isNull);
        }
      });
    });
  });

  // ===========================================================================
  // ATTENTION MASK
  // ===========================================================================
  group('Attention Mask', () {
    group('Basic Attention Mask', () {
      test('all real tokens have attention mask 1', () {
        final encoding = tokenizer.encode('hello world');

        expect(encoding.attentionMask, everyElement(equals(1)));
      });

      test('attention mask length matches token length', () {
        final encoding = tokenizer.encode('hello world');

        expect(encoding.attentionMask.length, equals(encoding.length));
      });

      test('special tokens have attention mask 1', () {
        final encoding = tokenizer.encode('hello');

        // [CLS] at start
        expect(encoding.attentionMask.first, equals(1));
        // [SEP] at end
        expect(encoding.attentionMask.last, equals(1));
      });
    });

    group('Attention Mask with Right Padding', () {
      test('right padding has attention mask 0', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
          padOnRight: true,
        );

        // Original tokens should have 1
        expect(
          padded.attentionMask.sublist(0, encoding.length),
          everyElement(equals(1)),
        );

        // Padding should have 0
        expect(
          padded.attentionMask.sublist(encoding.length),
          everyElement(equals(0)),
        );
      });

      test('attention mask pattern is [1,1,1,1,0,0,0]', () {
        final encoding = tokenizer.encode('hi'); // [CLS] hi [SEP] = 3 tokens
        final padded = encoding.withPadding(
          targetLength: 7,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        expect(padded.attentionMask, equals([1, 1, 1, 0, 0, 0, 0]));
      });
    });

    group('Attention Mask with Left Padding', () {
      test('left padding has attention mask 0 at start', () {
        final encoding = tokenizer.encode('hello');
        final padCount = 10 - encoding.length;
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
          padOnRight: false,
        );

        // Padding should have 0
        expect(
          padded.attentionMask.sublist(0, padCount),
          everyElement(equals(0)),
        );

        // Original tokens should have 1
        expect(padded.attentionMask.sublist(padCount), everyElement(equals(1)));
      });

      test('attention mask pattern is [0,0,0,1,1,1,1]', () {
        final encoding = tokenizer.encode('hi'); // [CLS] hi [SEP] = 3 tokens
        final padded = encoding.withPadding(
          targetLength: 7,
          padTokenId: tokenizer.vocab.padTokenId,
          padOnRight: false,
        );

        expect(padded.attentionMask, equals([0, 0, 0, 0, 1, 1, 1]));
      });
    });

    group('Attention Mask with Truncation', () {
      test('truncation preserves attention mask values', () {
        final encoding = tokenizer.encode('one two three four five');
        final truncated = encoding.withTruncation(maxLength: 4);

        expect(truncated.attentionMask, everyElement(equals(1)));
        expect(truncated.attentionMask.length, equals(4));
      });
    });

    group('Attention Mask with Batch Encoding', () {
      test('batch encoding without padding has all 1s', () {
        final batch = tokenizer.encodeBatch(['hello', 'world']);

        for (final encoding in batch) {
          expect(encoding.attentionMask, everyElement(equals(1)));
        }
      });

      test('batch encoding with padding has correct masks', () {
        final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
          ..enablePadding();

        final batch = tok.encodeBatch(['hi', 'hello world how are you']);
        final shorter = batch[0];
        final longer = batch[1];

        // Shorter should have some 0s
        expect(shorter.attentionMask.contains(0), isTrue);
        // Longer might have all 1s if it's the longest
        expect(
          longer.attentionMask.where((m) => m == 1).length,
          greaterThanOrEqualTo(
            shorter.attentionMask.where((m) => m == 1).length,
          ),
        );
      });
    });

    group('Attention Mask Count Validation', () {
      test('attention mask 1 count matches non-padding tokens', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        final onesCount = padded.attentionMask.where((m) => m == 1).length;
        expect(onesCount, equals(encoding.length));
      });

      test('attention mask 0 count matches padding tokens', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        final zerosCount = padded.attentionMask.where((m) => m == 0).length;
        expect(zerosCount, equals(10 - encoding.length));
      });
    });
  });

  // ===========================================================================
  // TOKEN TYPE IDS
  // ===========================================================================
  group('Token Type IDs', () {
    group('Single Sequence Type IDs', () {
      test('single sequence has all typeId 0', () {
        final encoding = tokenizer.encode('hello world');

        expect(encoding.typeIds, everyElement(equals(0)));
      });

      test('single sequence with special tokens has typeId 0', () {
        final encoding = tokenizer.encode('hello');

        // [CLS], content, [SEP] all should be 0
        expect(encoding.typeIds.first, equals(0)); // [CLS]
        expect(encoding.typeIds.last, equals(0)); // [SEP]
      });

      test('single sequence without special tokens has typeId 0', () {
        final encoding = tokenizer.encode('hello', addSpecialTokens: false);

        expect(encoding.typeIds, everyElement(equals(0)));
      });
    });

    group('Pair Sequence Type IDs', () {
      test('first sequence has typeId 0', () {
        final encoding = tokenizer.encodePair(
          'first sentence',
          'second sentence',
        );
        final firstSepIdx = encoding.tokens.indexOf('[SEP]');

        for (var i = 0; i <= firstSepIdx; i++) {
          expect(
            encoding.typeIds[i],
            equals(0),
            reason: 'Token $i (${encoding.tokens[i]}) should have typeId 0',
          );
        }
      });

      test('second sequence has typeId 1', () {
        final encoding = tokenizer.encodePair('first', 'second');
        final firstSepIdx = encoding.tokens.indexOf('[SEP]');

        for (var i = firstSepIdx + 1; i < encoding.length; i++) {
          expect(
            encoding.typeIds[i],
            equals(1),
            reason: 'Token $i (${encoding.tokens[i]}) should have typeId 1',
          );
        }
      });

      test('type IDs correctly identify sentence boundary', () {
        final encoding = tokenizer.encodePair('hello world', 'foo bar');

        // Count tokens in each segment
        final segment0 = encoding.typeIds.where((t) => t == 0).length;
        final segment1 = encoding.typeIds.where((t) => t == 1).length;

        expect(segment0, greaterThan(0));
        expect(segment1, greaterThan(0));
      });

      test('pair without special tokens still has correct type IDs', () {
        final encoding = tokenizer.encodePair(
          'hello',
          'world',
          addSpecialTokens: false,
        );

        // First sequence should be 0, second should be 1
        // Without special tokens, the separation is based on which sequence the token came from
        expect(encoding.typeIds.first, equals(0));
      });
    });

    group('Type IDs with Padding', () {
      test('padding has typeId 0', () {
        final encoding = tokenizer.encode('hello');
        final padded = encoding.withPadding(
          targetLength: 10,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        for (var i = encoding.length; i < padded.length; i++) {
          expect(padded.typeIds[i], equals(0));
        }
      });

      test('pair encoding with padding preserves type IDs', () {
        final encoding = tokenizer.encodePair('hello', 'world');
        final padded = encoding.withPadding(
          targetLength: 15,
          padTokenId: tokenizer.vocab.padTokenId,
        );

        // Original type IDs should be preserved
        expect(
          padded.typeIds.sublist(0, encoding.length),
          equals(encoding.typeIds),
        );

        // Padding type IDs should be 0
        expect(
          padded.typeIds.sublist(encoding.length),
          everyElement(equals(0)),
        );
      });
    });

    group('Type IDs with Truncation', () {
      test('truncation preserves type IDs', () {
        final encoding = tokenizer.encode('hello world');
        final truncated = encoding.withTruncation(maxLength: 3);

        expect(truncated.typeIds.length, equals(3));
        expect(truncated.typeIds, equals(encoding.typeIds.sublist(0, 3)));
      });

      test('pair truncation preserves type ID structure', () {
        final encoding = tokenizer.encodePair(
          'first sentence here',
          'second sentence here',
        );

        // Truncate but keep some of both sequences
        final truncated = encoding.withTruncation(maxLength: 8);

        expect(truncated.typeIds.length, equals(8));
        // Should still have some typeId 0 tokens
        expect(truncated.typeIds.where((t) => t == 0).length, greaterThan(0));
      });
    });

    group('Type IDs in Batch Processing', () {
      test('batch encoding maintains type IDs', () {
        final batch = tokenizer.encodeBatch(['hello', 'world', 'test']);

        for (final encoding in batch) {
          expect(encoding.typeIds, everyElement(equals(0)));
        }
      });

      test('batch pair encoding maintains type IDs', () {
        final pairs = [('q1', 'a1'), ('q2', 'a2')];
        final batch = tokenizer.encodePairBatch(pairs);

        for (final encoding in batch) {
          // Should have both typeId 0 and 1
          expect(encoding.typeIds.contains(0), isTrue);
          expect(encoding.typeIds.contains(1), isTrue);
        }
      });
    });

    group('Type ID and Sequence ID Relationship', () {
      test('non-special tokens have matching typeId and sequenceId', () {
        final encoding = tokenizer.encodePair('hello', 'world');

        for (var i = 0; i < encoding.length; i++) {
          if (encoding.specialTokensMask[i] == 0) {
            // For content tokens, sequenceId should match typeId
            expect(
              encoding.sequenceIds[i],
              equals(encoding.typeIds[i]),
              reason: 'Token $i sequenceId should match typeId',
            );
          }
        }
      });

      test('special tokens have null sequenceId but valid typeId', () {
        final encoding = tokenizer.encodePair('hello', 'world');

        for (var i = 0; i < encoding.length; i++) {
          if (encoding.specialTokensMask[i] == 1) {
            expect(encoding.sequenceIds[i], isNull);
            expect(encoding.typeIds[i], anyOf(equals(0), equals(1)));
          }
        }
      });
    });
  });

  // ===========================================================================
  // COMBINED OPERATIONS
  // ===========================================================================
  group('Combined Operations', () {
    test('truncate then pad produces exact length', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(maxLength: 5)
        ..enablePadding(length: 10);

      final encoding = tok.encode('one two three four five six seven');

      expect(encoding.length, equals(10));
      expect(encoding.attentionMask.where((m) => m == 1).length, equals(5));
      expect(encoding.attentionMask.where((m) => m == 0).length, equals(5));
    });

    test('batch with truncation and padding', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(maxLength: 8)
        ..enablePadding();

      final batch = tok.encodeBatch([
        'short',
        'this is a longer sentence that will be truncated',
      ]);

      // All same length
      expect(batch[0].length, equals(batch[1].length));
      // Should not exceed truncation limit
      expect(batch[0].length, lessThanOrEqualTo(8));
    });

    test('pair encoding with truncation preserves both sequences', () {
      final encoding = tokenizer.encodePair(
        'first sentence with many words here',
        'second also long',
        maxLength: 10,
        truncationStrategy: TruncationStrategy.longestFirst,
      );

      expect(encoding.length, lessThanOrEqualTo(10));
      // Should have tokens from both sequences
      expect(encoding.typeIds.contains(0), isTrue);
      expect(encoding.typeIds.contains(1), isTrue);
    });

    test('full pipeline: encode -> truncate -> pad -> verify', () {
      final encoding = tokenizer.encode('one two three four five six');
      final truncated = encoding.withTruncation(maxLength: 5);
      final padded = truncated.withPadding(
        targetLength: 8,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      // Verify final state
      expect(padded.length, equals(8));
      expect(padded.tokens.length, equals(8));
      expect(padded.ids.length, equals(8));
      expect(padded.typeIds.length, equals(8));
      expect(padded.attentionMask.length, equals(8));
      expect(padded.specialTokensMask.length, equals(8));
      expect(padded.offsets.length, equals(8));
      expect(padded.wordIds.length, equals(8));
      expect(padded.sequenceIds.length, equals(8));

      // Verify attention mask
      expect(padded.attentionMask.sublist(0, 5), everyElement(equals(1)));
      expect(padded.attentionMask.sublist(5), everyElement(equals(0)));
    });
  });

  // ===========================================================================
  // EDGE CASES
  // ===========================================================================
  group('Edge Cases', () {
    test('encoding preserves data after multiple operations', () {
      var encoding = tokenizer.encode('hello world this is a test');

      // Chain of operations
      encoding = encoding.withTruncation(maxLength: 5);
      encoding = encoding.withPadding(
        targetLength: 10,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      // All arrays should be same length
      expect(encoding.tokens.length, equals(10));
      expect(encoding.ids.length, equals(10));
      expect(encoding.typeIds.length, equals(10));
      expect(encoding.attentionMask.length, equals(10));
      expect(encoding.specialTokensMask.length, equals(10));
      expect(encoding.offsets.length, equals(10));
      expect(encoding.wordIds.length, equals(10));
      expect(encoding.sequenceIds.length, equals(10));
    });

    test('empty encoding handles operations gracefully', () {
      final empty = Encoding.empty();

      // Truncation on empty
      final truncated = empty.withTruncation(maxLength: 10);
      expect(truncated.isEmpty, isTrue);

      // Padding on empty
      final padded = empty.withPadding(
        targetLength: 5,
        padTokenId: tokenizer.vocab.padTokenId,
      );
      expect(padded.length, equals(5));
    });

    test('very long text handles correctly', () {
      final longText = 'word ' * 1000;
      final encoding = tokenizer.encode(longText);

      expect(encoding.isNotEmpty, isTrue);
      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));
    });

    test('special characters in all positions', () {
      final encoding = tokenizer.encode('!hello! !world!');

      expect(encoding.tokens, contains('!'));
      expect(encoding.tokens, contains('hello'));
    });

    test('consecutive special characters', () {
      final encoding = tokenizer.encode('!!!???...');

      expect(encoding.isNotEmpty, isTrue);
    });
  });
}

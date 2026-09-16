/// Tests for Priority 2 features:
/// - Offset mapping methods (charToToken, tokenToChars, etc.)
/// - sequenceIds property
/// - Fluent padding/truncation configuration
/// - Encoding.merge() method
library;

import 'package:test/test.dart';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() {
  late WordPieceTokenizer tokenizer;

  setUpAll(() {
    tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt');
  });

  group('Offset Mapping Methods', () {
    group('charToToken', () {
      test('maps character position to token index', () {
        // "hello world" - after normalization
        final encoding = tokenizer.encode('hello world');

        // Character 0-4 is "hello" -> should map to token after [CLS]
        final tokenIdx = encoding.charToToken(0);
        expect(tokenIdx, isNotNull);
        expect(encoding.tokens[tokenIdx!], equals('hello'));
      });

      test('returns null for out-of-range positions', () {
        final encoding = tokenizer.encode('hi');

        expect(encoding.charToToken(100), isNull);
        expect(encoding.charToToken(-1), isNull);
      });

      test('works with pair encodings', () {
        final encoding = tokenizer.encodePair('hello', 'world');

        // First sequence
        final tokenA = encoding.charToToken(0, sequenceIndex: 0);
        expect(tokenA, isNotNull);

        // Second sequence
        final tokenB = encoding.charToToken(0, sequenceIndex: 1);
        expect(tokenB, isNotNull);
        expect(tokenB, isNot(equals(tokenA)));
      });
    });

    group('charToWord', () {
      test('maps character position to word index', () {
        final encoding = tokenizer.encode('hello world');

        // First word
        expect(encoding.charToWord(0), equals(0));
        // Second word (after space)
        expect(encoding.charToWord(6), equals(1));
      });
    });

    group('tokenToChars', () {
      test('returns character span for a token', () {
        final encoding = tokenizer.encode('hello world');

        // Find the "hello" token
        final helloIdx = encoding.tokens.indexOf('hello');
        final span = encoding.tokenToChars(helloIdx);

        expect(span, isNotNull);
        expect(span!.$1, greaterThanOrEqualTo(0));
        expect(span.$2, greaterThan(span.$1));
      });

      test('returns null for special tokens', () {
        final encoding = tokenizer.encode('hello');

        // [CLS] is at index 0
        final clsSpan = encoding.tokenToChars(0);
        expect(clsSpan, isNull);
      });

      test('returns null for invalid index', () {
        final encoding = tokenizer.encode('hello');

        expect(encoding.tokenToChars(-1), isNull);
        expect(encoding.tokenToChars(1000), isNull);
      });
    });

    group('tokenToWord', () {
      test('returns word index for a token', () {
        final encoding = tokenizer.encode('hello world');

        // "hello" token
        final helloIdx = encoding.tokens.indexOf('hello');
        expect(encoding.tokenToWord(helloIdx), equals(0));

        // "world" token
        final worldIdx = encoding.tokens.indexOf('world');
        expect(encoding.tokenToWord(worldIdx), equals(1));
      });

      test('returns null for special tokens', () {
        final encoding = tokenizer.encode('hello');

        // [CLS] at index 0
        expect(encoding.tokenToWord(0), isNull);
      });
    });

    group('tokenToSequence', () {
      test('returns sequence index for single sequence', () {
        final encoding = tokenizer.encode('hello');

        // Content tokens should be sequence 0
        final helloIdx = encoding.tokens.indexOf('hello');
        expect(encoding.tokenToSequence(helloIdx), equals(0));
      });

      test('returns sequence index for pair encoding', () {
        final encoding = tokenizer.encodePair('hello', 'world');

        // Find tokens in each sequence
        final helloIdx = encoding.tokens.indexOf('hello');
        final worldIdx = encoding.tokens.indexOf('world');

        expect(encoding.tokenToSequence(helloIdx), equals(0));
        expect(encoding.tokenToSequence(worldIdx), equals(1));
      });

      test('returns null for special tokens', () {
        final encoding = tokenizer.encode('hello');

        // [CLS] at index 0
        expect(encoding.tokenToSequence(0), isNull);
      });
    });

    group('wordToChars', () {
      test('returns character span for a word', () {
        final encoding = tokenizer.encode('hello world');

        // First word "hello"
        final span0 = encoding.wordToChars(0);
        expect(span0, isNotNull);

        // Second word "world"
        final span1 = encoding.wordToChars(1);
        expect(span1, isNotNull);
        expect(span1!.$1, greaterThan(span0!.$2 - 1));
      });

      test('returns null for non-existent word', () {
        final encoding = tokenizer.encode('hello');

        expect(encoding.wordToChars(100), isNull);
      });
    });

    group('wordToTokens', () {
      test('returns token range for a word', () {
        final encoding = tokenizer.encode('tokenization');

        // "tokenization" gets split into subwords
        final range = encoding.wordToTokens(0);
        expect(range, isNotNull);
        expect(range!.$2, greaterThan(range.$1));
      });

      test('returns single token range for simple word', () {
        final encoding = tokenizer.encode('hello');

        final range = encoding.wordToTokens(0);
        expect(range, isNotNull);
        expect(range!.$2 - range.$1, equals(1));
      });
    });
  });

  group('sequenceIds', () {
    test('single sequence has all 0s for content tokens', () {
      final encoding = tokenizer.encode('hello world');

      for (var i = 0; i < encoding.length; i++) {
        if (encoding.specialTokensMask[i] == 0) {
          expect(encoding.sequenceIds[i], equals(0));
        }
      }
    });

    test('special tokens have null sequenceId', () {
      final encoding = tokenizer.encode('hello');

      // [CLS] and [SEP]
      expect(encoding.sequenceIds.first, isNull);
      expect(encoding.sequenceIds.last, isNull);
    });

    test('pair encoding has correct sequence IDs', () {
      final encoding = tokenizer.encodePair('hello', 'world');

      // Count sequences
      final seq0Count = encoding.sequenceIds.where((s) => s == 0).length;
      final seq1Count = encoding.sequenceIds.where((s) => s == 1).length;
      final nullCount = encoding.sequenceIds.where((s) => s == null).length;

      expect(seq0Count, greaterThan(0));
      expect(seq1Count, greaterThan(0));
      expect(nullCount, equals(3)); // [CLS], [SEP], [SEP]
    });

    test('nSequences returns correct count', () {
      final single = tokenizer.encode('hello');
      final pair = tokenizer.encodePair('hello', 'world');
      final empty = Encoding.empty();

      expect(single.nSequences, equals(1));
      expect(pair.nSequences, equals(2));
      expect(empty.nSequences, equals(0));
    });
  });

  group('Encoding.merge', () {
    test('merges two encodings', () {
      final enc1 = tokenizer.encode('hello', addSpecialTokens: false);
      final enc2 = tokenizer.encode('world', addSpecialTokens: false);

      final merged = Encoding.merge([enc1, enc2]);

      expect(merged.length, equals(enc1.length + enc2.length));
      expect(merged.tokens, contains('hello'));
      expect(merged.tokens, contains('world'));
    });

    test('empty list returns empty encoding', () {
      final merged = Encoding.merge([]);

      expect(merged.isEmpty, isTrue);
    });

    test('single encoding returns same encoding', () {
      final enc = tokenizer.encode('hello');
      final merged = Encoding.merge([enc]);

      expect(merged.tokens, equals(enc.tokens));
      expect(merged.ids, equals(enc.ids));
    });

    test('growingOffsets adjusts offsets sequentially', () {
      final enc1 = tokenizer.encode('hi', addSpecialTokens: false);
      final enc2 = tokenizer.encode('bye', addSpecialTokens: false);

      final merged = Encoding.merge([enc1, enc2], growingOffsets: true);

      // Second encoding's offsets should be shifted
      final enc2StartOffset = enc1.offsets.last.$2;
      final mergedSecondOffset = merged.offsets[enc1.length];

      expect(mergedSecondOffset.$1, greaterThanOrEqualTo(enc2StartOffset));
    });

    test('preserves offsets when growingOffsets is false', () {
      final enc1 = tokenizer.encode('hi', addSpecialTokens: false);
      final enc2 = tokenizer.encode('bye', addSpecialTokens: false);

      final merged = Encoding.merge([enc1, enc2], growingOffsets: false);

      // Second encoding's offsets should be preserved
      expect(merged.offsets[enc1.length], equals(enc2.offsets.first));
    });

    test('word IDs are adjusted', () {
      final enc1 = tokenizer.encode('hello world', addSpecialTokens: false);
      final enc2 = tokenizer.encode('foo bar', addSpecialTokens: false);

      final merged = Encoding.merge([enc1, enc2]);

      // enc2's word IDs should be shifted by enc1's word count
      final enc1MaxWordId = enc1.wordIds
          .where((w) => w != null)
          .fold<int>(0, (max, w) => w! > max ? w : max);

      final enc2FirstWordId = merged.wordIds[enc1.length];
      expect(enc2FirstWordId, greaterThan(enc1MaxWordId));
    });
  });

  group('Fluent Padding Configuration', () {
    test('enablePadding with fixed length', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enablePadding(length: 20);

      final encoding = tok.encode('hello');

      expect(encoding.length, equals(20));
      expect(tok.padding, isNotNull);
      expect(tok.padding!.length, equals(20));
    });

    test('enablePadding with direction', () {
      final rightPad = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enablePadding(length: 10, direction: PaddingDirection.right);

      final leftPad = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enablePadding(length: 10, direction: PaddingDirection.left);

      final rightEnc = rightPad.encode('hi');
      final leftEnc = leftPad.encode('hi');

      // Right padding: [CLS] hi [SEP] [PAD] [PAD] ...
      expect(rightEnc.tokens.last, equals('[PAD]'));

      // Left padding: [PAD] [PAD] ... [CLS] hi [SEP]
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

    test('batch encoding pads to longest', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enablePadding();

      final batch = tok.encodeBatch(['hi', 'hello world how are you']);

      // All should be same length (longest in batch)
      expect(batch[0].length, equals(batch[1].length));
    });
  });

  group('Fluent Truncation Configuration', () {
    test('enableTruncation with maxLength', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(maxLength: 5);

      final encoding = tok.encode('this is a very long sentence');

      expect(encoding.length, equals(5));
      expect(tok.truncation, isNotNull);
      expect(tok.truncation!.maxLength, equals(5));
    });

    test('enableTruncation with direction', () {
      final rightTrunc = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(maxLength: 4, direction: TruncationDirection.right);

      final leftTrunc = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(maxLength: 4, direction: TruncationDirection.left);

      final rightEnc = rightTrunc.encode('one two three four five');
      final leftEnc = leftTrunc.encode('one two three four five');

      // Right truncation keeps beginning
      expect(rightEnc.tokens[1], equals('one'));

      // Left truncation keeps end
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

    test('truncation applies to pair encoding', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(
          maxLength: 10,
          strategy: TruncationStrategy.longestFirst,
        );

      final encoding = tok.encodePair('very long first sentence here', 'short');

      expect(encoding.length, lessThanOrEqualTo(10));
    });
  });

  group('Combined Padding and Truncation', () {
    test('truncate then pad', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(maxLength: 5)
        ..enablePadding(length: 10);

      final encoding = tok.encode('this is a very long sentence');

      expect(encoding.length, equals(10));
      // First 5 tokens are content, rest are padding
      expect(encoding.attentionMask.where((m) => m == 1).length, equals(5));
      expect(encoding.attentionMask.where((m) => m == 0).length, equals(5));
    });

    test('method chaining works', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(maxLength: 10)
        ..enablePadding(length: 10, padToMultipleOf: 8);

      expect(tok.truncation, isNotNull);
      expect(tok.padding, isNotNull);
    });

    test('batch with both truncation and padding', () {
      final tok = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
        ..enableTruncation(maxLength: 8)
        ..enablePadding();

      final batch = tok.encodeBatch([
        'short',
        'this is a longer sentence that will be truncated',
      ]);

      // All same length after truncation and padding
      expect(batch[0].length, equals(batch[1].length));
      // Should not exceed max length
      expect(batch[0].length, lessThanOrEqualTo(8));
    });
  });

  group('Edge Cases', () {
    test('sequenceIds preserved after padding', () {
      final encoding = tokenizer.encode('hello');
      final padded = encoding.withPadding(
        targetLength: 10,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      // Padding tokens should have null sequenceId
      expect(padded.sequenceIds.last, isNull);
    });

    test('sequenceIds preserved after truncation', () {
      final encoding = tokenizer.encode('hello world this is test');
      final truncated = encoding.withTruncation(maxLength: 5);

      expect(truncated.sequenceIds.length, equals(5));
    });

    test('offset methods handle subword tokens', () {
      final encoding = tokenizer.encode('tokenization');

      // "tokenization" -> "token" + "##ization"
      final tokenRange = encoding.wordToTokens(0);
      expect(tokenRange, isNotNull);
      expect(tokenRange!.$2 - tokenRange.$1, greaterThan(1));

      // All subword tokens should map to same word
      for (var i = tokenRange.$1; i < tokenRange.$2; i++) {
        expect(encoding.tokenToWord(i), equals(0));
      }
    });
  });
}

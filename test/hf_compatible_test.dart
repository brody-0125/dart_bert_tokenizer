/// Test cases migrated from HuggingFace transformers test_tokenization_common.py
///
/// These tests ensure compatibility with HuggingFace tokenizers behavior.
library;

import 'package:test/test.dart';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() {
  late WordPieceTokenizer tokenizer;

  setUpAll(() {
    // Load real BERT vocabulary for comprehensive testing
    tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt');
  });

  group('Internal Consistency (test_internal_consistency)', () {
    // Validates that tokenize → convert_tokens_to_ids → encode produce matching results
    test('tokenize and encode produce consistent IDs', () {
      const text = 'Hello, world!';

      final encoding = tokenizer.encode(text);

      // Extract non-special tokens
      final contentTokens = <String>[];
      final contentIds = <int>[];
      for (var i = 0; i < encoding.length; i++) {
        if (encoding.specialTokensMask[i] == 0) {
          contentTokens.add(encoding.tokens[i]);
          contentIds.add(encoding.ids[i]);
        }
      }

      // Convert tokens back to IDs - should match
      final convertedIds = tokenizer.convertTokensToIds(contentTokens);
      expect(convertedIds, equals(contentIds));
    });

    test('encode and decode round-trip', () {
      const text = 'The quick brown fox jumps over the lazy dog';

      final encoding = tokenizer.encode(text);
      final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);

      expect(decoded.toLowerCase(), equals(text.toLowerCase()));
    });

    test('batch encoding produces consistent results', () {
      const texts = ['Hello world', 'Goodbye world'];

      final batchResults = tokenizer.encodeBatch(texts);
      final individualResults = texts.map(tokenizer.encode).toList();

      for (var i = 0; i < texts.length; i++) {
        expect(batchResults[i].tokens, equals(individualResults[i].tokens));
        expect(batchResults[i].ids, equals(individualResults[i].ids));
      }
    });
  });

  group('Padding Tests (test_encode_basic_padding)', () {
    test('right padding (default)', () {
      final encoding = tokenizer.encode('hello');
      final padded = encoding.withPadding(
        targetLength: 10,
        padTokenId: tokenizer.vocab.padTokenId,
        padToken: '[PAD]',
        padOnRight: true,
      );

      expect(padded.length, equals(10));
      // Original tokens should be at the beginning
      expect(padded.tokens.first, equals('[CLS]'));
      expect(padded.tokens[1], equals('hello'));
      expect(padded.tokens[2], equals('[SEP]'));
      // Padding should be at the end
      expect(padded.tokens.sublist(3), everyElement(equals('[PAD]')));
      // Attention mask should be 0 for padding
      expect(padded.attentionMask.sublist(3), everyElement(equals(0)));
    });

    test('left padding', () {
      final encoding = tokenizer.encode('hello');
      final padded = encoding.withPadding(
        targetLength: 10,
        padTokenId: tokenizer.vocab.padTokenId,
        padToken: '[PAD]',
        padOnRight: false,
      );

      expect(padded.length, equals(10));
      // Padding should be at the beginning
      final padCount = 10 - encoding.length;
      expect(padded.tokens.sublist(0, padCount), everyElement(equals('[PAD]')));
      // Original tokens should be at the end
      expect(padded.tokens.last, equals('[SEP]'));
      // Attention mask should be 0 for padding
      expect(
        padded.attentionMask.sublist(0, padCount),
        everyElement(equals(0)),
      );
    });

    test('no padding when already at target length', () {
      final encoding = tokenizer.encode('hello');
      final originalLength = encoding.length;
      final padded = encoding.withPadding(
        targetLength: originalLength,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      expect(padded.length, equals(originalLength));
      expect(padded.tokens, equals(encoding.tokens));
    });

    test('no padding when exceeding target length', () {
      final encoding = tokenizer.encode('This is a longer sentence');
      final padded = encoding.withPadding(
        targetLength: 3,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      // Should remain unchanged
      expect(padded.length, equals(encoding.length));
    });
  });

  group('Padding to Multiple (test_padding_to_multiple_of)', () {
    test('pads to multiple of 8', () {
      final encoding = tokenizer.encode('hello'); // likely 3 tokens
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 8,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      expect(padded.length % 8, equals(0));
      expect(padded.length, greaterThanOrEqualTo(encoding.length));
    });

    test('pads to multiple of 16', () {
      final encoding = tokenizer.encode('This is a test sentence');
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 16,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      expect(padded.length % 16, equals(0));
    });

    test('no padding when already multiple', () {
      final encoding = tokenizer.encode('hello'); // 3 tokens
      // Pad first to get exact multiple
      final prePadded = encoding.withPadding(
        targetLength: 8,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      final result = prePadded.withPaddingToMultipleOf(
        multiple: 8,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      expect(result.length, equals(8));
      expect(result.length, equals(prePadded.length));
    });

    test('left padding to multiple', () {
      final encoding = tokenizer.encode('hello');
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 8,
        padTokenId: tokenizer.vocab.padTokenId,
        padOnRight: false,
      );

      expect(padded.length % 8, equals(0));
      // Original tokens should be at the end
      expect(padded.tokens.last, equals('[SEP]'));
    });

    test('handles multiple of 1', () {
      final encoding = tokenizer.encode('hello');
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 1,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      expect(padded.length, equals(encoding.length)); // No change
    });

    test('handles invalid multiple gracefully', () {
      final encoding = tokenizer.encode('hello');
      final padded = encoding.withPaddingToMultipleOf(
        multiple: 0,
        padTokenId: tokenizer.vocab.padTokenId,
      );

      expect(padded.length, equals(encoding.length)); // No change
    });
  });

  group('Truncation Tests (test_right_and_left_truncation)', () {
    test('right truncation (default)', () {
      final encoding = tokenizer.encode(
        'This is a very long sentence for testing',
      );
      final truncated = encoding.withTruncation(
        maxLength: 5,
        truncateFromEnd: true,
      );

      expect(truncated.length, equals(5));
      // Should keep beginning tokens
      expect(truncated.tokens.first, equals('[CLS]'));
      expect(truncated.tokens, equals(encoding.tokens.sublist(0, 5)));
    });

    test('left truncation', () {
      final encoding = tokenizer.encode(
        'This is a very long sentence for testing',
      );
      final originalLength = encoding.length;
      final truncated = encoding.withTruncation(
        maxLength: 5,
        truncateFromEnd: false,
      );

      expect(truncated.length, equals(5));
      // Should keep ending tokens
      expect(truncated.tokens.last, equals('[SEP]'));
      expect(
        truncated.tokens,
        equals(encoding.tokens.sublist(originalLength - 5)),
      );
    });

    test('no truncation when under max length', () {
      final encoding = tokenizer.encode('hello');
      final truncated = encoding.withTruncation(maxLength: 100);

      expect(truncated.length, equals(encoding.length));
      expect(truncated.tokens, equals(encoding.tokens));
    });
  });

  group('Special Tokens Tests (test_tokenizers_common_properties)', () {
    test('vocabulary has all required special tokens', () {
      expect(tokenizer.vocab.clsTokenId, isNotNull);
      expect(tokenizer.vocab.sepTokenId, isNotNull);
      expect(tokenizer.vocab.padTokenId, isNotNull);
      expect(tokenizer.vocab.unkTokenId, isNotNull);
      expect(tokenizer.vocab.maskTokenId, isNotNull);
    });

    test('special token IDs are valid', () {
      expect(tokenizer.vocab.clsTokenId, greaterThanOrEqualTo(0));
      expect(tokenizer.vocab.sepTokenId, greaterThanOrEqualTo(0));
      expect(tokenizer.vocab.padTokenId, greaterThanOrEqualTo(0));
      expect(tokenizer.vocab.unkTokenId, greaterThanOrEqualTo(0));
      expect(tokenizer.vocab.maskTokenId, greaterThanOrEqualTo(0));
    });

    test('special tokens can be retrieved by ID', () {
      expect(
        tokenizer.vocab.idToToken(tokenizer.vocab.clsTokenId),
        equals('[CLS]'),
      );
      expect(
        tokenizer.vocab.idToToken(tokenizer.vocab.sepTokenId),
        equals('[SEP]'),
      );
      expect(
        tokenizer.vocab.idToToken(tokenizer.vocab.padTokenId),
        equals('[PAD]'),
      );
      expect(
        tokenizer.vocab.idToToken(tokenizer.vocab.unkTokenId),
        equals('[UNK]'),
      );
      expect(
        tokenizer.vocab.idToToken(tokenizer.vocab.maskTokenId),
        equals('[MASK]'),
      );
    });
  });

  group('Special Tokens Mask (test_special_tokens_mask)', () {
    test('single sequence has correct special tokens mask', () {
      final encoding = tokenizer.encode('hello world');

      // First token ([CLS]) should be marked as special
      expect(encoding.specialTokensMask.first, equals(1));
      // Last token ([SEP]) should be marked as special
      expect(encoding.specialTokensMask.last, equals(1));
      // Middle tokens should not be special
      for (var i = 1; i < encoding.length - 1; i++) {
        expect(
          encoding.specialTokensMask[i],
          equals(0),
          reason: 'Token at index $i should not be special',
        );
      }
    });

    test('pair sequence has correct special tokens mask', () {
      final encoding = tokenizer.encodePair('hello', 'world');

      // [CLS] at start
      expect(encoding.specialTokensMask.first, equals(1));
      // [SEP] at end
      expect(encoding.specialTokensMask.last, equals(1));

      // Find the middle [SEP]
      var sepCount = 0;
      for (var i = 0; i < encoding.length; i++) {
        if (encoding.tokens[i] == '[SEP]') {
          expect(encoding.specialTokensMask[i], equals(1));
          sepCount++;
        }
      }
      expect(sepCount, equals(2)); // Should have two [SEP] tokens
    });

    test('special tokens count matches mask', () {
      final encoding = tokenizer.encode('hello world');

      final specialCount = encoding.specialTokensMask
          .where((m) => m == 1)
          .length;
      final specialTokensInOutput = encoding.tokens.where(
        (t) => t.startsWith('[') && t.endsWith(']'),
      );

      expect(specialCount, equals(specialTokensInOutput.length));
    });
  });

  group('Token Type IDs (test_token_type_ids)', () {
    test('single sequence has all typeId = 0', () {
      final encoding = tokenizer.encode('hello world');

      expect(encoding.typeIds, everyElement(equals(0)));
    });

    test('pair sequence has correct type IDs', () {
      final encoding = tokenizer.encodePair(
        'first sentence',
        'second sentence',
      );

      // Find the first [SEP] position (end of first sequence)
      final firstSepIndex = encoding.tokens.indexOf('[SEP]');

      // Everything before and including first [SEP] should be typeId = 0
      for (var i = 0; i <= firstSepIndex; i++) {
        expect(
          encoding.typeIds[i],
          equals(0),
          reason: 'Token at index $i should have typeId 0',
        );
      }

      // Everything after first [SEP] should be typeId = 1
      for (var i = firstSepIndex + 1; i < encoding.length; i++) {
        expect(
          encoding.typeIds[i],
          equals(1),
          reason: 'Token at index $i should have typeId 1',
        );
      }
    });
  });

  group('Number of Added Tokens (test_number_of_added_tokens)', () {
    test('single sequence adds 2 special tokens', () {
      final encoding = tokenizer.encode('hello');
      final specialCount = encoding.specialTokensMask
          .where((m) => m == 1)
          .length;

      expect(specialCount, equals(2)); // [CLS] and [SEP]
    });

    test('pair sequence adds 3 special tokens', () {
      final encoding = tokenizer.encodePair('hello', 'world');
      final specialCount = encoding.specialTokensMask
          .where((m) => m == 1)
          .length;

      expect(specialCount, equals(3)); // [CLS], [SEP], [SEP]
    });
  });

  group('Unicode and Edge Cases', () {
    test('emoji handling - should produce [UNK] or valid tokens', () {
      // Emojis are typically not in BERT vocab
      final encoding = tokenizer.encode('Hello 😊');

      expect(encoding.tokens, contains('[CLS]'));
      expect(encoding.tokens, contains('[SEP]'));
      // Should not crash, and should handle gracefully
      expect(encoding.length, greaterThan(2));
    });

    test('Chinese characters are split per character', () {
      final encoding = tokenizer.encode('生活的真谛是');

      // Each Chinese character should be a separate token (or [UNK])
      expect(encoding.length, greaterThan(2));
    });

    test('Thai text handling', () {
      final encoding = tokenizer.encode('ปี');

      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));
    });

    test('mixed language text', () {
      final encoding = tokenizer.encode('Hello 世界 Bonjour');

      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));
      expect(encoding.tokens.contains('hello'), isTrue);
    });

    test('multiple spaces are handled', () {
      final encoding1 = tokenizer.encode('hello world');
      final encoding2 = tokenizer.encode('hello    world');

      // Both should produce same tokens (whitespace is normalized)
      expect(encoding1.tokens, equals(encoding2.tokens));
    });

    test('accented characters are normalized', () {
      final encoding = tokenizer.encode('café résumé naïve');

      // Accents should be stripped
      expect(encoding.tokens.contains('cafe'), isTrue);
      expect(encoding.tokens.contains('resume'), isTrue);
      expect(encoding.tokens.contains('naive'), isTrue);
    });

    test('empty string produces only special tokens', () {
      final encoding = tokenizer.encode('');

      expect(encoding.length, equals(2)); // [CLS] and [SEP] only
      expect(encoding.tokens, equals(['[CLS]', '[SEP]']));
    });

    test('whitespace only produces only special tokens', () {
      final encoding = tokenizer.encode('   \t\n  ');

      expect(encoding.length, equals(2));
      expect(encoding.tokens, equals(['[CLS]', '[SEP]']));
    });

    test('very long word handling', () {
      final longWord = 'a' * 250; // Exceeds maxWordLength (200)
      final encoding = tokenizer.encode(longWord);

      // Should be marked as [UNK]
      expect(encoding.tokens.contains('[UNK]'), isTrue);
    });
  });

  group('Batch Processing Consistency (test_batch_padded_input_match)', () {
    test('batch encoding maintains order', () {
      final texts = ['first', 'second', 'third'];
      final batch = tokenizer.encodeBatch(texts);

      expect(batch.length, equals(3));
      expect(batch[0].tokens.contains('first'), isTrue);
      expect(batch[1].tokens.contains('second'), isTrue);
      expect(batch[2].tokens.contains('third'), isTrue);
    });

    test('batch with mixed lengths', () {
      final texts = [
        'short',
        'This is a much longer sentence for testing',
        'medium length',
      ];
      final batch = tokenizer.encodeBatch(texts);

      expect(batch.length, equals(3));
      // Each should have proper structure
      for (final encoding in batch) {
        expect(encoding.tokens.first, equals('[CLS]'));
        expect(encoding.tokens.last, equals('[SEP]'));
        expect(encoding.attentionMask, everyElement(equals(1)));
      }
    });

    test('batch pair encoding', () {
      final pairs = [('question1', 'context1'), ('question2', 'context2')];
      final batch = tokenizer.encodePairBatch(pairs);

      expect(batch.length, equals(2));
      for (final encoding in batch) {
        // Should have [CLS] ... [SEP] ... [SEP] structure
        final sepCount = encoding.tokens.where((t) => t == '[SEP]').length;
        expect(sepCount, equals(2));
      }
    });
  });

  group('Decode Tests', () {
    test('decode with special tokens', () {
      final encoding = tokenizer.encode('hello world');
      final withSpecial = tokenizer.decode(
        encoding.ids,
        skipSpecialTokens: false,
      );

      expect(withSpecial, contains('[CLS]'));
      expect(withSpecial, contains('[SEP]'));
    });

    test('decode without special tokens', () {
      final encoding = tokenizer.encode('hello world');
      final withoutSpecial = tokenizer.decode(
        encoding.ids,
        skipSpecialTokens: true,
      );

      expect(withoutSpecial, isNot(contains('[CLS]')));
      expect(withoutSpecial, isNot(contains('[SEP]')));
      expect(withoutSpecial, equals('hello world'));
    });

    test('decode subword tokens are joined', () {
      final encoding = tokenizer.encode('tokenization');
      final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);

      // Should not have ## in the output
      expect(decoded, isNot(contains('##')));
    });

    test('batch decode', () {
      final texts = ['hello', 'world'];
      final batch = tokenizer.encodeBatch(texts);
      final decoded = tokenizer.decodeBatch(
        batch.map((e) => e.ids).toList(),
        skipSpecialTokens: true,
      );

      expect(decoded, equals(['hello', 'world']));
    });
  });

  group('Offset Mapping Tests', () {
    test('offsets point to correct character positions', () {
      final encoding = tokenizer.encode('Hello World');

      // Skip special tokens (which have (0, 0) offsets)
      for (var i = 0; i < encoding.length; i++) {
        if (encoding.specialTokensMask[i] == 0) {
          final offset = encoding.offsets[i];
          expect(
            offset.$1,
            lessThanOrEqualTo(offset.$2),
            reason: 'Start should be <= end',
          );
          expect(
            offset.$2,
            greaterThan(0),
            reason: 'Non-special tokens should have non-zero offsets',
          );
        }
      }
    });

    test('word IDs group subword tokens', () {
      final encoding = tokenizer.encode('tokenization is great');

      // Find tokens that are subwords (start with ##)
      for (var i = 0; i < encoding.length; i++) {
        if (encoding.tokens[i].startsWith('##')) {
          // Subword should have same wordId as previous non-special token
          expect(encoding.wordIds[i], isNotNull);
        }
      }
    });
  });

  group('Config Options', () {
    test('can disable CLS token', () {
      final noClsTokenizer = WordPieceTokenizer(
        vocab: tokenizer.vocab,
        config: const WordPieceConfig(addClsToken: false),
      );

      final encoding = noClsTokenizer.encode('hello');
      expect(encoding.tokens.first, isNot(equals('[CLS]')));
    });

    test('can disable SEP token', () {
      final noSepTokenizer = WordPieceTokenizer(
        vocab: tokenizer.vocab,
        config: const WordPieceConfig(addSepToken: false),
      );

      final encoding = noSepTokenizer.encode('hello');
      expect(encoding.tokens.last, isNot(equals('[SEP]')));
    });

    test('can disable both special tokens', () {
      final noSpecialTokenizer = WordPieceTokenizer(
        vocab: tokenizer.vocab,
        config: const WordPieceConfig(addClsToken: false, addSepToken: false),
      );

      final encoding = noSpecialTokenizer.encode('hello');
      expect(encoding.tokens, isNot(contains('[CLS]')));
      expect(encoding.tokens, isNot(contains('[SEP]')));
      expect(encoding.tokens, contains('hello'));
    });
  });
}

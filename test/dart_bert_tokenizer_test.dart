import 'package:test/test.dart';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() {
  group('Trie', () {
    late Trie trie;

    setUp(() {
      trie = Trie();
    });

    test('insert and lookup', () {
      trie.insert('hello', 1);
      trie.insert('world', 2);
      trie.insert('hel', 3);

      expect(trie.lookup('hello'), equals(1));
      expect(trie.lookup('world'), equals(2));
      expect(trie.lookup('hel'), equals(3));
      expect(trie.lookup('unknown'), isNull);
    });

    test('contains', () {
      trie.insert('test', 1);

      expect(trie.contains('test'), isTrue);
      expect(trie.contains('unknown'), isFalse);
    });

    test('findLongestPrefix', () {
      trie.insert('hello', 1);
      trie.insert('hel', 2);
      trie.insert('he', 3);

      final match = trie.findLongestPrefix('helloworld');
      expect(match, isNotNull);
      expect(match!.token, equals('hello'));
      expect(match.tokenId, equals(1));
      expect(match.start, equals(0));
      expect(match.end, equals(5));
    });

    test('findLongestPrefix with offset', () {
      trie.insert('world', 1);

      final match = trie.findLongestPrefix('helloworld', 5);
      expect(match, isNotNull);
      expect(match!.token, equals('world'));
    });

    test('findAllPrefixes', () {
      trie.insert('a', 1);
      trie.insert('ab', 2);
      trie.insert('abc', 3);

      final matches = trie.findAllPrefixes('abcd');
      expect(matches.length, equals(3));
      expect(matches[0].token, equals('a'));
      expect(matches[1].token, equals('ab'));
      expect(matches[2].token, equals('abc'));
    });

    test('size', () {
      expect(trie.size, equals(0));
      trie.insert('a', 1);
      expect(trie.size, equals(1));
      trie.insert('b', 2);
      expect(trie.size, equals(2));
      // Inserting same token updates, doesn't increase size
      trie.insert('a', 3);
      expect(trie.size, equals(2));
    });
  });

  group('Vocabulary', () {
    test('fromTokens', () {
      final vocab = Vocabulary.fromTokens([
        '[PAD]',
        '[UNK]',
        '[CLS]',
        '[SEP]',
        '[MASK]',
        'hello',
        'world',
        '##ing',
      ]);

      expect(vocab.size, equals(8));
      expect(vocab.tokenToId('[PAD]'), equals(0));
      expect(vocab.tokenToId('[UNK]'), equals(1));
      expect(vocab.tokenToId('hello'), equals(5));
      expect(vocab.idToToken(5), equals('hello'));
    });

    test('special token IDs', () {
      final vocab = Vocabulary.fromTokens([
        '[PAD]', // 0
        ...List.generate(99, (i) => '[unused$i]'), // 1-99
        '[UNK]', // 100
        '[CLS]', // 101
        '[SEP]', // 102
        '[MASK]', // 103
      ]);

      expect(vocab.padTokenId, equals(0));
      expect(vocab.unkTokenId, equals(100));
      expect(vocab.clsTokenId, equals(101));
      expect(vocab.sepTokenId, equals(102));
      expect(vocab.maskTokenId, equals(103));
    });

    test('trie lookup for regular tokens', () {
      final vocab = Vocabulary.fromTokens([
        '[PAD]',
        '[UNK]',
        'hello',
        'hel',
        '##lo',
      ]);

      // Regular tokens should be in the main trie
      expect(vocab.trie.contains('hello'), isTrue);
      expect(vocab.trie.contains('hel'), isTrue);

      // Subword tokens should be in subword trie (without prefix)
      expect(vocab.subwordTrie.contains('lo'), isTrue);

      // Special tokens should not be in tries
      expect(vocab.trie.contains('[PAD]'), isFalse);
    });
  });

  group('BertPreTokenizer', () {
    const preTokenizer = BertPreTokenizer();

    test('basic whitespace splitting', () {
      final tokens = preTokenizer.preTokenize('hello world');

      expect(tokens.length, equals(2));
      expect(tokens[0].text, equals('hello'));
      expect(tokens[1].text, equals('world'));
    });

    test('punctuation splitting', () {
      final tokens = preTokenizer.preTokenize('hello, world!');

      expect(tokens.length, equals(4));
      expect(
        tokens.map((t) => t.text).toList(),
        equals(['hello', ',', 'world', '!']),
      );
    });

    test('lowercase', () {
      final tokens = preTokenizer.preTokenize('Hello WORLD');

      expect(tokens[0].text, equals('hello'));
      expect(tokens[1].text, equals('world'));
    });

    test('Chinese characters', () {
      final tokens = preTokenizer.preTokenize('你好');

      expect(tokens.length, equals(2));
      expect(tokens[0].text, equals('你'));
      expect(tokens[1].text, equals('好'));
    });

    test('mixed text', () {
      final tokens = preTokenizer.preTokenize('Hello 世界!');

      expect(tokens.length, equals(4));
      expect(tokens[0].text, equals('hello'));
      expect(tokens[1].text, equals('世'));
      expect(tokens[2].text, equals('界'));
      expect(tokens[3].text, equals('!'));
    });

    test('empty string', () {
      final tokens = preTokenizer.preTokenize('');
      expect(tokens, isEmpty);
    });

    test('whitespace only', () {
      final tokens = preTokenizer.preTokenize('   ');
      expect(tokens, isEmpty);
    });
  });

  group('Encoding', () {
    test('empty encoding', () {
      final encoding = Encoding.empty();

      expect(encoding.isEmpty, isTrue);
      expect(encoding.length, equals(0));
    });

    test('withPadding', () {
      final encoding = Encoding(
        tokens: ['hello', 'world'],
        ids: [1, 2],
        typeIds: [0, 0],
        attentionMask: [1, 1],
        specialTokensMask: [0, 0],
        offsets: [(0, 5), (6, 11)],
        wordIds: [0, 1],
      );

      final padded = encoding.withPadding(
        targetLength: 5,
        padTokenId: 0,
        padToken: '[PAD]',
      );

      expect(padded.length, equals(5));
      expect(padded.tokens.sublist(2), equals(['[PAD]', '[PAD]', '[PAD]']));
      expect(padded.attentionMask.sublist(2), equals([0, 0, 0]));
    });

    test('withTruncation', () {
      final encoding = Encoding(
        tokens: ['a', 'b', 'c', 'd', 'e'],
        ids: [1, 2, 3, 4, 5],
        typeIds: [0, 0, 0, 0, 0],
        attentionMask: [1, 1, 1, 1, 1],
        specialTokensMask: [0, 0, 0, 0, 0],
        offsets: [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
        wordIds: [0, 1, 2, 3, 4],
      );

      final truncated = encoding.withTruncation(maxLength: 3);

      expect(truncated.length, equals(3));
      expect(truncated.tokens, equals(['a', 'b', 'c']));
    });
  });

  group('WordPieceTokenizer', () {
    late WordPieceTokenizer tokenizer;

    setUp(() {
      // Create a minimal vocabulary for testing
      final vocab = Vocabulary.fromTokens([
        '[PAD]', // 0
        ...List.generate(99, (i) => '[unused$i]'), // 1-99
        '[UNK]', // 100
        '[CLS]', // 101
        '[SEP]', // 102
        '[MASK]', // 103
        ...List.generate(993, (i) => '[unused${99 + i}]'), // 104-1096
        'hello', // 1097
        'world', // 1098
        'test', // 1099
        ',', // 1100
        '.', // 1101
        '!', // 1102
        '##ing', // 1103
        '##ed', // 1104
        '##s', // 1105
      ]);

      tokenizer = WordPieceTokenizer(vocab: vocab);
    });

    test('encode basic text', () {
      final encoding = tokenizer.encode('hello world');

      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));
      expect(encoding.tokens.contains('hello'), isTrue);
      expect(encoding.tokens.contains('world'), isTrue);
    });

    test('encode with punctuation', () {
      final encoding = tokenizer.encode('hello, world!');

      expect(encoding.tokens, contains(','));
      expect(encoding.tokens, contains('!'));
    });

    test('unknown tokens', () {
      final encoding = tokenizer.encode('xyz');

      // 'xyz' is not in vocabulary, should be [UNK]
      expect(encoding.tokens, contains('[UNK]'));
    });

    test('attention mask', () {
      final encoding = tokenizer.encode('hello world');

      // All tokens should have attention mask = 1
      expect(encoding.attentionMask.every((m) => m == 1), isTrue);
    });

    test('special tokens mask', () {
      final encoding = tokenizer.encode('hello world');

      // First and last should be special tokens
      expect(encoding.specialTokensMask.first, equals(1));
      expect(encoding.specialTokensMask.last, equals(1));

      // Middle tokens should not be special
      expect(encoding.specialTokensMask[1], equals(0));
    });

    test('encodePair', () {
      final encoding = tokenizer.encodePair('hello', 'world');

      // Should have format: [CLS] hello [SEP] world [SEP]
      expect(encoding.tokens.first, equals('[CLS]'));
      expect(encoding.tokens.last, equals('[SEP]'));

      // Check type IDs
      // First sentence should have typeId = 0
      // Second sentence should have typeId = 1
      final sepIndex = encoding.tokens.indexOf('[SEP]');
      expect(
        encoding.typeIds.sublist(0, sepIndex + 1).every((t) => t == 0),
        isTrue,
      );
    });

    test('decode', () {
      final encoding = tokenizer.encode('hello world');
      final decoded = tokenizer.decode(encoding.ids);

      expect(decoded, equals('hello world'));
    });

    test('decode skips special tokens', () {
      final encoding = tokenizer.encode('hello');
      final decoded = tokenizer.decode(encoding.ids, skipSpecialTokens: true);

      expect(decoded, isNot(contains('[CLS]')));
      expect(decoded, isNot(contains('[SEP]')));
    });

    test('encodeBatch', () {
      final encodings = tokenizer.encodeBatch(['hello', 'world', 'test']);

      expect(encodings.length, equals(3));
      for (final encoding in encodings) {
        expect(encoding.tokens.first, equals('[CLS]'));
        expect(encoding.tokens.last, equals('[SEP]'));
      }
    });

    test('convertTokensToIds', () {
      final ids = tokenizer.convertTokensToIds(['hello', 'world']);

      expect(ids, equals([1097, 1098]));
    });

    test('convertIdsToTokens', () {
      final tokens = tokenizer.convertIdsToTokens([1097, 1098]);

      expect(tokens, equals(['hello', 'world']));
    });
  });

  group('EncodingBuilder', () {
    test('build encoding', () {
      final builder = EncodingBuilder();

      builder.addSpecialToken(token: '[CLS]', id: 101, typeId: 0);
      builder.addToken(
        token: 'hello',
        id: 1,
        typeId: 0,
        offset: (0, 5),
        wordId: 0,
      );
      builder.addSpecialToken(token: '[SEP]', id: 102, typeId: 0);

      final encoding = builder.build();

      expect(encoding.length, equals(3));
      expect(encoding.tokens, equals(['[CLS]', 'hello', '[SEP]']));
      expect(encoding.specialTokensMask, equals([1, 0, 1]));
    });
  });
}

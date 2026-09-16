import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() async {
  print('Loading vocab.txt (30,522 tokens)...');
  final stopwatch = Stopwatch()..start();

  final tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt');

  print('Loaded in ${stopwatch.elapsedMilliseconds}ms');
  print('Vocabulary size: ${tokenizer.vocab.size}');
  print('');

  // Test cases
  final testCases = [
    'Hello, world!',
    'The quick brown fox jumps over the lazy dog.',
    'BERT is a transformer-based model.',
    'Tokenization is the process of breaking text into tokens.',
    'I love natural language processing!',
    'TensorFlow and PyTorch are popular deep learning frameworks.',
    'The United States of America is a country.',
    'Machine learning models can understand human language.',
    'This is an example of subword tokenization.',
    'Unbelievably, the tokenizer handles unknown words gracefully.',
  ];

  print('=== Tokenization Tests ===\n');

  for (final text in testCases) {
    final encoding = tokenizer.encode(text);
    print('Input: "$text"');
    print('Tokens: ${encoding.tokens}');
    print('IDs: ${encoding.ids}');
    print('Length: ${encoding.length}');
    print('');
  }

  // Test subword tokenization
  print('=== Subword Tokenization Tests ===\n');

  final subwordTests = [
    'playing',
    'unbelievable',
    'tokenization',
    'transformers',
    'preprocessing',
  ];

  for (final word in subwordTests) {
    final encoding = tokenizer.encode(word);
    // Remove [CLS] and [SEP]
    final tokens = encoding.tokens.sublist(1, encoding.tokens.length - 1);
    print('$word -> $tokens');
  }

  print('');

  // Test sentence pair
  print('=== Sentence Pair Test ===\n');
  final pairEncoding = tokenizer.encodePair(
    'What is the capital of France?',
    'Paris is the capital of France.',
  );
  print('Question: "What is the capital of France?"');
  print('Answer: "Paris is the capital of France."');
  print('Tokens: ${pairEncoding.tokens}');
  print('Type IDs: ${pairEncoding.typeIds}');
  print('');

  // Test decode
  print('=== Decode Test ===\n');
  const originalText = 'Hello, how are you today?';
  final enc = tokenizer.encode(originalText);
  final decoded = tokenizer.decode(enc.ids);
  print('Original: "$originalText"');
  print('Encoded: ${enc.tokens}');
  print('Decoded: "$decoded"');
  print('');

  // Test batch encoding
  print('=== Batch Encoding Test ===\n');
  stopwatch.reset();
  stopwatch.start();

  final batchTexts = List.generate(
    100,
    (i) => 'This is test sentence number $i.',
  );
  final batchEncodings = tokenizer.encodeBatch(batchTexts);

  print('Encoded 100 sentences in ${stopwatch.elapsedMilliseconds}ms');
  print(
    'Average tokens per sentence: ${batchEncodings.map((e) => e.length).reduce((a, b) => a + b) / batchEncodings.length}',
  );
  print('');

  // Test special characters and edge cases
  print('=== Edge Cases ===\n');

  final edgeCases = [
    '', // empty
    '   ', // whitespace only
    '!!!???', // punctuation only
    '12345', // numbers
    'café', // accents
    '你好世界', // Chinese
    'Hello 世界!', // mixed
    'a' * 100, // long word (should be [UNK])
  ];

  for (final text in edgeCases) {
    final encoding = tokenizer.encode(text);
    final display = text.isEmpty
        ? '(empty)'
        : (text.trim().isEmpty ? '(whitespace)' : text);
    print('$display -> ${encoding.tokens}');
  }

  print('');

  // Performance test
  print('=== Performance Test ===\n');

  const longText = '''
  Natural language processing (NLP) is a subfield of linguistics, computer science,
  and artificial intelligence concerned with the interactions between computers and
  human language, in particular how to program computers to process and analyze large
  amounts of natural language data. The result is a computer capable of understanding
  the contents of documents, including the contextual nuances of the language within them.
  ''';

  stopwatch.reset();
  stopwatch.start();

  const iterations = 1000;
  for (var i = 0; i < iterations; i++) {
    tokenizer.encode(longText);
  }

  final elapsed = stopwatch.elapsedMilliseconds;
  print('Tokenized long text $iterations times in ${elapsed}ms');
  print(
    'Average: ${(elapsed / iterations).toStringAsFixed(2)}ms per tokenization',
  );

  final singleEncoding = tokenizer.encode(longText);
  print('Tokens per text: ${singleEncoding.length}');
  print(
    'Throughput: ${(singleEncoding.length * iterations / elapsed * 1000).toStringAsFixed(0)} tokens/sec',
  );
}

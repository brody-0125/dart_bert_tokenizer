import 'dart:io';
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() {
  final tokenizer = _loadTokenizer();

  print('=== Basic Encoding ===');
  final encoding = tokenizer.encode('Hello, world!');
  print('Text: "Hello, world!"');
  print('Tokens: ${encoding.tokens}');
  print('IDs: ${encoding.ids}');
  print('Attention: ${encoding.attentionMask}');
  print('');

  print('=== Without Special Tokens ===');
  final raw = tokenizer.encode('Hello, world!', addSpecialTokens: false);
  print('Tokens: ${raw.tokens}');
  print('');

  print('=== Decoding ===');
  print(
    'Decoded: "${tokenizer.decode(encoding.ids, skipSpecialTokens: true)}"',
  );
  print('');

  print('=== Sentence Pair ===');
  final pair = tokenizer.encodePair('What is this?', 'This is a test.');
  print('Tokens: ${pair.tokens}');
  print('Type IDs: ${pair.typeIds}');
  print('Sequence IDs: ${pair.sequenceIds}');
  print('');

  print('=== Offset Mapping ===');
  final enc = tokenizer.encode('hello world');
  print('Text: "hello world"');
  final tokenIdx = enc.charToToken(6);
  if (tokenIdx != null) {
    print('Char 6 ("w") → Token: "${enc.tokens[tokenIdx]}"');
  }
  final span = enc.tokenToChars(1);
  print('Token "hello" → Chars: $span');
  print('Word 0 → Tokens: ${enc.wordToTokens(0)}');
  print('');

  print('=== Truncation ===');
  const longText = 'This is a very long sentence that needs truncation';
  final truncated = tokenizer.encode(longText).withTruncation(maxLength: 6);
  print('Original: "$longText"');
  print('Truncated: ${truncated.tokens}');
  print('');

  print('=== Truncation Strategies ===');
  for (final strategy in TruncationStrategy.values) {
    final result = tokenizer.encodePair(
      'word ' * 10,
      'short',
      maxLength: 12,
      truncationStrategy: strategy,
    );
    print('${strategy.name}: ${result.length} tokens');
  }
  print('');

  print('=== Padding ===');
  final short = tokenizer.encode('hi');
  print('Original: ${short.tokens} (${short.length})');

  final rightPadded = short.withPadding(
    targetLength: 8,
    padTokenId: tokenizer.vocab.padTokenId,
  );
  print('Right: ${rightPadded.tokens}');
  print('Mask:  ${rightPadded.attentionMask}');

  final leftPadded = short.withPadding(
    targetLength: 8,
    padTokenId: tokenizer.vocab.padTokenId,
    padOnRight: false,
  );
  print('Left:  ${leftPadded.tokens}');
  print('');

  print('=== Fluent Configuration ===');
  final configured = WordPieceTokenizer(vocab: tokenizer.vocab)
    ..enableTruncation(maxLength: 10)
    ..enablePadding(length: 10);

  final result = configured.encode('This is a test sentence');
  print('Truncation(10) + Padding(10):');
  print('Tokens: ${result.tokens}');
  print('Length: ${result.length}');
  print('');

  print('=== Batch Encoding ===');
  final batchTokenizer = WordPieceTokenizer(vocab: tokenizer.vocab)
    ..enablePadding();

  final texts = ['short', 'a bit longer', 'the longest sentence here'];
  final batch = batchTokenizer.encodeBatch(texts);
  for (var i = 0; i < texts.length; i++) {
    print('"${texts[i]}" → ${batch[i].length} tokens');
  }
  print('All same length: ${batch.map((e) => e.length).toSet().length == 1}');
  print('');

  print('=== Vocabulary ===');
  print('Size: ${tokenizer.vocab.size}');
  print(
    '[CLS]=${tokenizer.vocab.clsTokenId}, [SEP]=${tokenizer.vocab.sepTokenId}, [PAD]=${tokenizer.vocab.padTokenId}',
  );
  print(
    'Special tokens (single): ${tokenizer.numSpecialTokensToAdd(isPair: false)}',
  );
  print(
    'Special tokens (pair): ${tokenizer.numSpecialTokensToAdd(isPair: true)}',
  );
  print('');

  print('=== Encoding Merge ===');
  final enc1 = tokenizer.encode('hello', addSpecialTokens: false);
  final enc2 = tokenizer.encode('world', addSpecialTokens: false);
  final merged = Encoding.merge([enc1, enc2]);
  print('Merged: ${merged.tokens}');
}

WordPieceTokenizer _loadTokenizer() {
  if (File('vocab.txt').existsSync()) {
    print('Loading from vocab.txt...\n');
    return WordPieceTokenizer.fromVocabFileSync('vocab.txt');
  }

  print('Using demo vocabulary...\n');
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
    'short',
    'long',
    'very',
    'sentence',
    'bit',
    'longer',
    'longest',
    'here',
    'needs',
    'be',
    'to',
    'hi',
    'truncation',
    ',',
    '.',
    '!',
    '?',
    "'",
    '##s',
    '##ed',
    '##ing',
    '##er',
    '##est',
    '##ly',
  ]);
  return WordPieceTokenizer(vocab: vocab);
}

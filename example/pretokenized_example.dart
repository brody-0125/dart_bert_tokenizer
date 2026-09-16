import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

/// An external segmenter supplies words; the tokenizer preserves their indices.
void main() {
  final tokenizer = WordPieceTokenizer(
    vocab: Vocabulary.fromTokens([
      '[PAD]',
      '[UNK]',
      '[CLS]',
      '[SEP]',
      '[MASK]',
      'hello',
      ',',
      'world',
      'play',
      '##ing',
    ]),
  );
  final words = ['Hello,', 'world', 'playing'];
  final labels = [1, 0, 2];
  final encoding = tokenizer.encodePreTokenized(words);
  int? previousWord;
  final tokenLabels = <int>[];
  for (final word in encoding.wordIds) {
    tokenLabels.add(word == null || word == previousWord ? -100 : labels[word]);
    previousWord = word;
  }
  if (tokenLabels.join(',') != '-100,1,-100,0,2,-100,-100' ||
      encoding.charToToken(0, wordIndex: 1) != 3 ||
      encoding.wordToTokens(2) != (4, 6)) {
    throw StateError('Pre-tokenized alignment example failed');
  }
  print(encoding.tokens);
  print(tokenLabels);
  // Pair labels must additionally inspect sequenceIds to select the label list.
  // char positions here belong to each word, not to a reconstructed sentence.
}

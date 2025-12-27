/// A high-performance BERT WordPiece tokenizer implementation for Dart.
///
/// This library provides a complete implementation of the WordPiece tokenization
/// algorithm used in BERT and other transformer models. It is designed to be
/// compatible with HuggingFace tokenizers while offering excellent performance
/// for Dart applications.
///
/// ## Features
///
/// - Full WordPiece tokenization compatible with HuggingFace tokenizers
/// - Support for single text and text pair encoding
/// - Batch processing with optional parallel execution using isolates
/// - Configurable padding and truncation strategies
/// - Pre-tokenization with BERT-style normalization
/// - Efficient trie-based vocabulary lookup
///
/// ## Basic Usage
///
/// ```dart
/// import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
///
/// void main() async {
///   // Load tokenizer from vocabulary file
///   final tokenizer = await WordPieceTokenizer.fromVocabFile('vocab.txt');
///
///   // Encode text
///   final encoding = tokenizer.encode('Hello, world!');
///   print(encoding.ids);     // Token IDs
///   print(encoding.tokens);  // Token strings
///
///   // Decode back to text
///   final decoded = tokenizer.decode(encoding.ids);
///   print(decoded);  // 'hello , world !'
/// }
/// ```
///
/// ## Text Pair Encoding
///
/// For tasks like question answering or sentence similarity:
///
/// ```dart
/// final encoding = tokenizer.encodePair(
///   'What is Dart?',
///   'Dart is a programming language.',
/// );
/// print(encoding.typeIds);  // Segment IDs (0 for first, 1 for second)
/// ```
///
/// ## Batch Processing
///
/// For processing multiple texts efficiently:
///
/// ```dart
/// final encodings = await tokenizer.encodeBatchParallel([
///   'First sentence.',
///   'Second sentence.',
///   'Third sentence.',
/// ]);
/// ```
library dart_bert_tokenizer;

export 'src/encoding.dart' show Encoding, EncodingBuilder, TruncationStrategy;
export 'src/pre_tokenizer.dart' show BertPreTokenizer, PreToken;
export 'src/trie.dart' show Trie, TrieNode, TrieMatch;
export 'src/vocabulary.dart' show Vocabulary, SpecialTokens;
export 'src/wordpiece_tokenizer.dart'
    show
        WordPieceTokenizer,
        WordPieceConfig,
        PaddingConfig,
        PaddingDirection,
        TruncationConfig,
        TruncationDirection;

/// A pure Dart BERT WordPiece tokenizer.
///
/// This library provides a complete implementation of the WordPiece tokenization
/// algorithm used in BERT and other transformer models. Supported Hugging Face
/// WordPiece pipelines are verified with pinned model fixtures; see the README
/// for supported components, model coverage and compatibility limits.
///
/// ## Features
///
/// - WordPiece tokenization with supported Hugging Face JSON pipelines
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
/// print(encoding.typeIds);  // Template-defined type IDs; sequenceIds identifies input A/B
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
library;

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

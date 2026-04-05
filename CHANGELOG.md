# Changelog

## 1.1.0

- Add `tokenizer.json` loading support for HuggingFace tokenizer files
  - `WordPieceTokenizer.fromTokenizerJson()` - async file loading
  - `WordPieceTokenizer.fromTokenizerJsonSync()` - sync file loading
  - `WordPieceTokenizer.fromTokenizerJsonString()` - load from JSON string
- Add `Vocabulary.fromMap()` factory for token-to-ID map construction
- Automatically extract normalizer, post-processor, and added tokens from JSON
- Support optional `configOverride` parameter for advanced configuration
- 25 new tests including vocab.txt vs tokenizer.json equivalence verification

## 1.0.1

- Add comprehensive dartdoc documentation to all public API elements
- Document library, classes, methods, and properties following Effective Dart guidelines
- Improve pub.dev documentation score (target: 20%+ API documentation)

## 1.0.0

- Initial release
- Pure Dart implementation of BERT WordPiece tokenizer
- 100% HuggingFace tokenizers compatibility
- Memory-efficient typed arrays (Int32List, Uint8List)
- Single text and sentence pair encoding
- Batch encoding (sequential and parallel with Isolates)
- Padding and truncation support
- Offset mapping (char-to-token, token-to-char, word-to-tokens)
- Vocabulary access and token conversion utilities

# dart_bert_tokenizer

![Dart](https://img.shields.io/badge/Dart-3.0+-0175C2.svg?logo=dart)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Compatible-FF6600)

A lightweight, pure Dart implementation of BERT WordPiece tokenizer.

## Features

- **Pure Dart** - Zero dependencies, works everywhere (Flutter, Server, CLI, Web)
- **Memory Efficient** - Typed arrays (`Int32List`, `Uint8List`) for 50-70% memory reduction
- **Full API** - Encoding, decoding, padding, truncation, offset mapping
- **Batch Processing** - Sequential and parallel (Isolate-based) batch encoding
- **Well Tested** - 298 tests with 100% pass rate

## Installation

```yaml
dependencies:
  dart_bert_tokenizer: ^1.0.2
```

## Quick Start

```dart
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';

void main() {
  // Load tokenizer
  final tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt');

  // Encode text
  final encoding = tokenizer.encode('Hello, world!');
  print(encoding.tokens); // [[CLS], hello, ,, world, !, [SEP]]
  print(encoding.ids);    // [101, 7592, 1010, 2088, 999, 102]

  // Decode back to text
  final text = tokenizer.decode(encoding.ids, skipSpecialTokens: true);
  print(text); // hello , world !
}
```

## Usage

### Single Text Encoding

```dart
final encoding = tokenizer.encode('Hello world');

print(encoding.tokens);        // Token strings
print(encoding.ids);           // Token IDs (Int32List)
print(encoding.attentionMask); // Attention mask (Uint8List)
print(encoding.typeIds);       // Type IDs (Uint8List)
print(encoding.offsets);       // Character offsets [(start, end), ...]
print(encoding.wordIds);       // Word indices
print(encoding.sequenceIds);   // Sequence indices (0, 1, or null)

// Without special tokens
final raw = tokenizer.encode('Hello', addSpecialTokens: false);
```

### Sentence Pair Encoding

```dart
// For QA, NLI, sentence similarity tasks
final encoding = tokenizer.encodePair(
  'What is machine learning?',
  'Machine learning is a subset of AI.',
);

print(encoding.typeIds);     // [0,0,0,0,0,0, 1,1,1,1,1,1,1]
print(encoding.sequenceIds); // [null,0,0,0,0,null, 1,1,1,1,1,1,null]
```

### Batch Encoding

```dart
// Sequential batch
final encodings = tokenizer.encodeBatch(['Hello', 'World', 'Test']);

// Parallel batch (uses Isolates for batches >= 8)
final encodings = await tokenizer.encodeBatchParallel(texts);

// Pair batch
final pairs = [('Q1', 'A1'), ('Q2', 'A2')];
final encodings = tokenizer.encodePairBatch(pairs);
```

### Padding

```dart
// Fluent API
final tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
  ..enablePadding(length: 512, direction: PaddingDirection.right);

// Or pad to longest in batch
tokenizer.enablePadding(); // Auto-pads to longest

// Manual padding
final padded = encoding.withPadding(
  targetLength: 128,
  padTokenId: tokenizer.vocab.padTokenId,
  padOnRight: true,
);

// Pad to multiple of N
final padded = encoding.withPaddingToMultipleOf(
  multiple: 8,
  padTokenId: tokenizer.vocab.padTokenId,
);
```

### Truncation

```dart
// Fluent API
final tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
  ..enableTruncation(maxLength: 512, direction: TruncationDirection.right);

// Manual truncation
final truncated = encoding.withTruncation(maxLength: 64);

// Truncation strategies for pairs
tokenizer.encodePair(textA, textB,
  maxLength: 128,
  truncationStrategy: TruncationStrategy.longestFirst,
);
```

**Truncation Strategies:**
- `longestFirst` - Remove from longest sequence iteratively
- `onlyFirst` - Truncate first sequence only
- `onlySecond` - Truncate second sequence only
- `doNotTruncate` - No truncation

### Offset Mapping

```dart
final encoding = tokenizer.encode('Hello world');

// Character position -> Token index
final tokenIdx = encoding.charToToken(6); // 'w' -> token index

// Token index -> Character span
final (start, end) = encoding.tokenToChars(1)!; // token -> (0, 5)

// Word index -> Token span
final (startToken, endToken) = encoding.wordToTokens(0)!;

// Token -> Word index
final wordIdx = encoding.tokenToWord(1);

// Token -> Sequence index (0, 1, or null for special tokens)
final seqIdx = encoding.tokenToSequence(1);
```

### Vocabulary Access

```dart
print(tokenizer.vocab.size);       // 30522
print(tokenizer.vocab.clsTokenId); // 101
print(tokenizer.vocab.sepTokenId); // 102
print(tokenizer.vocab.padTokenId); // 0
print(tokenizer.vocab.unkTokenId); // 100
print(tokenizer.vocab.maskTokenId); // 103

// Token <-> ID conversion
tokenizer.convertTokensToIds(['hello', 'world']); // [7592, 2088]
tokenizer.convertIdsToTokens([7592, 2088]);       // ['hello', 'world']

// Check if token exists
tokenizer.vocab.contains('hello'); // true
```

### Decoding

```dart
// Decode with special tokens
final text = tokenizer.decode(encoding.ids, skipSpecialTokens: false);

// Decode without special tokens (default: true)
final text = tokenizer.decode(encoding.ids);

// Batch decode
final texts = tokenizer.decodeBatch(idsBatch);
```

## ONNX Runtime Integration

Use with ONNX Runtime for on-device ML inference:

```dart
import 'package:dart_bert_tokenizer/dart_bert_tokenizer.dart';
import 'dart:typed_data';

final tokenizer = WordPieceTokenizer.fromVocabFileSync('vocab.txt')
  ..enableTruncation(maxLength: 512);

final encoding = tokenizer.encode('search_query: What is ML?');

// Encoding.ids is already Int32List, convert to Int64List for ONNX
final inputIds = Int64List.fromList(encoding.ids);
final attentionMask = Int64List.fromList(encoding.attentionMask);

// Pass to ONNX session
// final outputs = await session.run({
//   'input_ids': inputIds,
//   'attention_mask': attentionMask,
// });
```

## Configuration

```dart
final tokenizer = WordPieceTokenizer(
  vocab: vocab,
  config: WordPieceConfig(
    lowercase: true,           // Convert to lowercase (default: true)
    stripAccents: true,        // Remove accents (default: true)
    handleChineseChars: true,  // Space around Chinese chars (default: true)
    subwordPrefix: '##',       // Subword prefix (default: '##')
    maxWordLength: 200,        // Max word length before [UNK] (default: 200)
    addClsToken: true,         // Add [CLS] token (default: true)
    addSepToken: true,         // Add [SEP] token (default: true)
  ),
);
```

## API Reference

### WordPieceTokenizer

| Method | Description |
|--------|-------------|
| `fromVocabFile(path)` | Load from vocab file (async) |
| `fromVocabFileSync(path)` | Load from vocab file (sync) |
| `encode(text)` | Encode single text |
| `encodePair(textA, textB)` | Encode text pair |
| `encodeBatch(texts)` | Encode multiple texts |
| `encodeBatchParallel(texts)` | Parallel batch encoding |
| `decode(ids)` | Decode IDs to text |
| `decodeBatch(idsBatch)` | Batch decode |
| `enablePadding()` / `noPadding()` | Configure padding |
| `enableTruncation()` / `noTruncation()` | Configure truncation |
| `convertTokensToIds(tokens)` | Convert tokens to IDs |
| `convertIdsToTokens(ids)` | Convert IDs to tokens |
| `numSpecialTokensToAdd(isPair)` | Get special token count |

### Encoding

| Property | Type | Description |
|----------|------|-------------|
| `tokens` | `List<String>` | Token strings |
| `ids` | `Int32List` | Token IDs |
| `attentionMask` | `Uint8List` | Attention mask (1=attend, 0=ignore) |
| `typeIds` | `Uint8List` | Token type IDs (0=first, 1=second) |
| `specialTokensMask` | `Uint8List` | Special token mask |
| `offsets` | `List<(int, int)>` | Character offsets |
| `wordIds` | `List<int?>` | Word indices |
| `sequenceIds` | `List<int?>` | Sequence indices |
| `length` | `int` | Number of tokens |

## Performance

| Metric | Value |
|--------|-------|
| Throughput | ~2M tokens/sec |
| Vocab loading | ~40ms (30K tokens) |
| Memory (vocab) | ~5MB |
| Lookup complexity | O(m) per token |
| HuggingFace compatibility | 100% (34 test cases) |

## Vocabulary File

Download BERT vocabulary from HuggingFace:
- [bert-base-uncased](https://huggingface.co/bert-base-uncased/raw/main/vocab.txt)
- [bert-base-cased](https://huggingface.co/bert-base-cased/raw/main/vocab.txt)

Format: One token per line, line number (0-indexed) = token ID.

## Testing

```bash
# Run all tests (298 tests)
dart test

# Run specific test file
dart test test/hf_compatible_test.dart

# Run benchmarks
dart run benchmark/performance_benchmark.dart

# Run HuggingFace compatibility benchmark
dart run benchmark/hf_compatibility_benchmark.dart
```

### HuggingFace Compatibility Verification

```bash
# Run HuggingFace compatibility benchmark (85 tests, 100% accuracy)
dart run benchmark/hf_compatibility_benchmark.dart

# Regenerate benchmark expected values (requires Python + tokenizers)
pip install tokenizers
python scripts/generate_hf_benchmark_data.py
```

## License

MIT License

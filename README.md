# dart_bert_tokenizer

![Dart](https://img.shields.io/badge/Dart-3.10.7+-0175C2.svg?logo=dart)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Compatible-FF6600)

A lightweight, pure Dart implementation of BERT WordPiece tokenizer.

## Features

- **Pure Dart** - Zero runtime dependencies for Flutter native, server and CLI applications
- **Memory Efficient** - Typed arrays (`Int32List`, `Uint8List`) for token IDs and masks
- **Full API** - Encoding, decoding, padding, truncation, offset mapping
- **Batch Processing** - Sequential and parallel (Isolate-based) batch encoding
- **HuggingFace tokenizer.json** - Load directly from HuggingFace tokenizer files
- **Well Tested** - Offline HF goldens and pinned network fixtures for twelve supported model pipelines and one unsupported-pipeline boundary

## Installation

```yaml
dependencies:
  dart_bert_tokenizer: ^1.3.0
```

## Quick Start

The token strings and IDs below assume the `google-bert/bert-base-uncased` vocabulary. Other models have different vocabularies and special-token IDs.

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

### Loading from tokenizer.json

Load supported Hugging Face WordPiece `tokenizer.json` files. Normalization, post-processing, decoder, padding and truncation settings are extracted; unsupported configurations raise `FormatException`. See [compatibility and limits](#hugging-face-compatibility-in-130).

```dart
// From file (async)
final tokenizer = await WordPieceTokenizer.fromTokenizerJson('tokenizer.json');

// From file (sync)
final tokenizer = WordPieceTokenizer.fromTokenizerJsonSync('tokenizer.json');

// From JSON string (e.g., embedded asset)
final tokenizer = WordPieceTokenizer.fromTokenizerJsonString(jsonString);

// Override extracted config if needed
final tokenizer = WordPieceTokenizer.fromTokenizerJsonSync(
  'tokenizer.json',
  configOverride: WordPieceConfig(lowercase: false),
);
```

`configOverride` replaces the exposed WordPiece settings as a whole, including defaults for fields you omit; it is not a partial merge. It also replaces the JSON template with the legacy CLS/SEP configuration. Prefer loading without an override when matching HF output.

## Usage

### Single Text Encoding

```dart
final encoding = tokenizer.encode('Hello world');

print(encoding.tokens);        // Token strings
print(encoding.ids);           // Token IDs (Int32List)
print(encoding.attentionMask); // Attention mask (Uint8List)
print(encoding.typeIds);       // Type IDs (Uint8List)
print(encoding.offsets);       // Original code-point offsets [(start, end), ...]
print(encoding.wordIds);       // Word indices
print(encoding.sequenceIds);   // Sequence indices (0, 1, or null)

// Without special tokens
final raw = tokenizer.encode('Hello', addSpecialTokens: false);
```

### Pre-tokenized Words (1.3.0)

```dart
final encoding = tokenizer.encodePreTokenized(['Hello,', 'world!']);
// tokens: [CLS], hello, ',', world, '!', [SEP]
// wordIds: null, 0, 0, 1, 1, null
// offsets: (0,0), (0,5), (5,6), (0,5), (5,6), (0,0)
final token = encoding.charToToken(0, wordIndex: 1); // 3
final span = encoding.wordToTokens(1); // (3, 5)
```

Each item still passes through the configured normalization, AddedToken matching,
pre-tokenizer and WordPiece stages. All resulting subwords retain that item's
original list index, including gaps from empty items. Offsets are Unicode
code-point positions **within each item**, not positions in a joined sentence.
AddedToken matches never cross item boundaries. Truncation preserves original
word IDs and offsets; `wordToChars` reports only the surviving token span.

Use `wordIndex` with `charToToken`/`charToWord` to disambiguate repeated offsets.
Omitting it preserves the existing first-match behavior. Pair lookups also take
`sequenceIndex`; `sequenceIds` identify inputs even when their type IDs are equal.

`encodePreTokenizedPair`, `encodePreTokenizedBatch`,
`encodePreTokenizedPairBatch`, `encodePreTokenizedBatchParallel` and
`encodePreTokenizedPairBatchParallel` support the same settings as their text
counterparts. Parallel calls snapshot nested input lists and tokenizer settings.
`encodeBatch(List<String>)` continues to mean a batch of raw strings.

For NER/POS, map `wordIds` to input labels. One common policy labels the first
subword and assigns -100 to later subwords and inserted special/padding tokens.
See the runnable [NER alignment example](example/pretokenized_example.dart).
External morphological analysis, full-sentence offset reconstruction, and label
policy are caller responsibilities. Pre-tokenized input does not make an
unsupported tokenizer.json pipeline supported.

### Sentence Pair Encoding

```dart
// For QA, NLI, sentence similarity tasks
final encoding = tokenizer.encodePair(
  'What is machine learning?',
  'Machine learning is a subset of AI.',
);

print(encoding.typeIds);     // Token type IDs assigned by the template
print(encoding.sequenceIds); // 0/1 for input A/B; null for special/padding tokens
```

### Batch Encoding

```dart
// Sequential batch
final encodings = tokenizer.encodeBatch(['Hello', 'World', 'Test']);

// Parallel batch (uses Isolates for batches >= 8)
final parallel = await tokenizer.encodeBatchParallel(texts);

// Pair batch
final pairs = [('Q1', 'A1'), ('Q2', 'A2')];
final pairEncodings = tokenizer.encodePairBatch(pairs);
final parallelPairs = await tokenizer.encodePairBatchParallel(pairs);
```

### Added tokens (1.2.0)

```dart
final changed = tokenizer.addTokens([
  const AddedToken('custom', singleWord: true),
  const AddedToken('<entity>', lstrip: true, rstrip: true),
]);
tokenizer.addSpecialTokens(['[ENTITY]']);
// For special tokens with explicit matching options:
tokenizer.addTokens([
  const AddedToken('[LINK]', special: true, normalized: true),
]);
final result = tokenizer.encode('custom  <entity>  [ENTITY]');
final id = tokenizer.vocab.tokenToId('[ENTITY]');
```

`singleWord` checks Unicode word characters on both sides. `lstrip`/`rstrip`
include neighboring Unicode whitespace in the matched token and its original
code-point offsets. Normal tokens default to `normalized: true`; special tokens
default to false. Raw tokens are extracted first, then normalized tokens from
the remaining text. Unmatched text proceeds through BERT splitting and WordPiece.

Registration returns the number of new or changed definitions, not just new IDs.
Existing token IDs are reused, identical definitions and empty strings are
ignored. Updates are atomic; new registrations do not change another tokenizer
sharing the original vocabulary. Parallel calls capture registration and padding/
truncation settings when invoked. Registering tokens does **not** resize the
model's embedding matrix: the model must support the resulting IDs.

Token lookup preserves the original spelling; decode can use its normalized
spelling. Special IDs are skipped by default, even if normalized, and remain
special after re-registration as ordinary tokens. Additional tokens are kept
out of the base WordPiece trie so failed boundary matches cannot bypass the
boundary rule through subword segmentation. These APIs also work with vocab.txt.

HF 0.23.2 edge-case policies:

- If different IDs normalize to the same pattern, the lowest ID wins. HF's
  choice can vary with hash-map order.
- Special-token filtering uses IDs, including normalized special tokens.
  Changing `normalized` refreshes decode spelling; HF can retain a stale cache.
- Fully consumed whitespace-only matches are skipped rather than producing
  the invalid slice that can crash HF 0.23.2 after combined stripping.
- Sparse vocabularies allocate above the highest occupied ID, preventing
  collisions; new IDs must fit Int32. This replaces 1.1.0's HF-style count-based
  allocation for sparse JSON vocabularies.

These differences are explicit fixture contracts, not blanket HF equivalence.
AraBERT fixtures validate its tokenizer JSON; its separately recommended
`ArabertPreprocessor` is not implemented here.

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
final paddedToMultiple = encoding.withPaddingToMultipleOf(
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

The tokenizer truncates content before inserting special tokens; `maxLength` includes those special tokens. `onlyFirst`/`onlySecond` throw when the selected sequence cannot remove enough tokens. `Encoding.withTruncation()` instead slices an already-built encoding and can remove special tokens.

### Offset Mapping

```dart
final encoding = tokenizer.encode('Hello world');

// Original Unicode code-point position -> Token index
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
final withoutSpecials = tokenizer.decode(encoding.ids);

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
| `fromTokenizerJson(path)` | Load from tokenizer.json (async) |
| `fromTokenizerJsonSync(path)` | Load from tokenizer.json (sync) |
| `fromTokenizerJsonString(json)` | Load from JSON string |
| `addTokens(tokens)` | Register or update AddedToken definitions |
| `addSpecialTokens(tokens)` | Register raw special token strings |
| `encode(text)` | Encode single text |
| `encodePair(textA, textB)` | Encode text pair |
| `encodeBatch(texts)` | Encode multiple texts |
| `encodeBatchParallel(texts)` | Parallel batch encoding |
| `encodePairBatch(pairs)` | Batch encoding of text pairs |
| `encodePairBatchParallel(pairs)` | Parallel encoding of text pairs |
| `decode(ids)` | Decode IDs to text |
| `decodeBatch(idsBatch)` | Batch decode |
| `enablePadding()` / `noPadding()` | Configure padding |
| `enableTruncation()` / `noTruncation()` | Configure truncation |
| `convertTokensToIds(tokens)` | Convert tokens to IDs |
| `convertIdsToTokens(ids)` | Convert IDs to tokens |
| `numSpecialTokensToAdd(isPair: true)` | Get special token count |

### Encoding

| Property | Type | Description |
|----------|------|-------------|
| `tokens` | `List<String>` | Token strings |
| `ids` | `Int32List` | Token IDs |
| `attentionMask` | `Uint8List` | Attention mask (1=attend, 0=ignore) |
| `typeIds` | `Uint8List` | Template-defined token type IDs (0–255) |
| `specialTokensMask` | `Uint8List` | Special token mask |
| `offsets` | `List<(int, int)>` | Original Unicode code-point offsets |
| `wordIds` | `List<int?>` | Word indices local to each input sequence |
| `sequenceIds` | `List<int?>` | Sequence indices |
| `length` | `int` | Number of tokens |

## Performance

Run the benchmarks below on your target device and vocabulary. This release
does not publish hardware-independent throughput or memory guarantees.

## Vocabulary Files

You can load from either `vocab.txt` or `tokenizer.json`:

| Format | Method | Description |
|--------|--------|-------------|
| `vocab.txt` | `fromVocabFile()` / `fromVocabFileSync()` | One token per line, zero-based line index = ID |
| `tokenizer.json` | `fromTokenizerJson()` / `fromTokenizerJsonSync()` | Supported Hugging Face WordPiece pipeline config |

Download from HuggingFace:
- [bert-base-uncased vocab.txt](https://huggingface.co/bert-base-uncased/raw/main/vocab.txt)
- [bert-base-uncased tokenizer.json](https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json)

## Testing

```bash
# Run offline tests (network fixtures are opt-in)
dart test

# Run specific test file
dart test test/hf_compatible_test.dart

# Run benchmarks
dart run benchmark/performance_benchmark.dart

# Run HuggingFace compatibility benchmark
dart run benchmark/hf_compatibility_benchmark.dart
```

### Reproducible HF fixtures

```sh
python -m pip install -r scripts/requirements-fixtures.txt
python scripts/generate_hf_fixtures.py
python scripts/generate_added_token_fixtures.py
python scripts/generate_unicode_word_boundaries.py
```

The pinned generator checks original file hashes and verifies that reduced
vocabularies reproduce the full-model expected values. Generated goldens are
checked in, so ordinary Dart tests do not require Python. The legacy benchmark
is a separate diagnostic; the fixture suite is the release compatibility gate.

## Hugging Face compatibility in 1.3.0

The suite pins twelve successful model pipelines and one unsupported-pipeline
boundary by repository revision and source-file SHA-256:

| Model(s) | What is verified |
|---|---|
| `google-bert/bert-base-uncased`, `bert-base-cased`, `bert-base-multilingual-cased` | Official JSON pipelines |
| `sentence-transformers/all-MiniLM-L6-v2` | Official JSON, including serialized padding |
| `klue/bert-base`, `klue/roberta-base` | Official JSON; Korean normalization and pair type/sequence IDs |
| `google-bert/bert-base-chinese`, `hfl/chinese-roberta-wwm-ext` | Official JSON; Chinese text and differing normalization settings |
| `asafaya/bert-base-arabic`, `dbmdz/bert-base-turkish-cased`, `google/muril-base-cased` | Official vocabularies with explicitly recorded HF WordPiece builder settings |
| `aubmindlab/bert-base-arabertv02` | Official JSON, including normalized special tokens with Unicode word boundaries |
| `ai4bharat/IndicBERTv2-MLM-only` | Explicit rejection of the unsupported Whitespace pre-tokenizer |

Vocabulary-derived fixtures validate the recorded conversion pipeline; they do
not establish full Transformers `AutoTokenizer` equivalence. Tohoku Japanese v3
and LINE Japanese DistilBERT require MeCab/UniDic preprocessing and are outside
this package's verified pipelines. Language coverage does not imply support for
every model of that language.

Python `tokenizers==0.23.2` generates the checked-in expected values. Tests compare
IDs, tokens, type IDs, attention/special masks, offsets, word/sequence IDs and
both decoder modes, including sequential and parallel batches. The current suite
has 1,900 offline tests and 1,340 opt-in network tests. CI checks Linux, Windows,
Dart 3.10.7 and stable, plus analysis, formatting and publish dry-run.
See [fixture provenance and regeneration](test/fixtures/huggingface/README.md).

Supported JSON components are WordPiece, BertNormalizer, BertPreTokenizer,
BertProcessing, WordPiece decoder, and TemplateProcessing with one occurrence
of each input sequence and single vocabulary-token special entries. Null
normalizer, pre-tokenizer, post-processor and decoder preserve their respective
absence. Added tokens support `normalized`, `single_word`, `lstrip`, `rstrip`
and `special`. Unsupported component types, invalid option types and nonzero
truncation stride fail explicitly with `FormatException`.
Type IDs come from the template, independently of sequence IDs: KLUE RoBERTa
uses type ID zero for both inputs. Type IDs must fit the public `Uint8List` representation (0–255).

JSON padding and truncation settings are applied, including MiniLM's serialized
128-token padding. Use `noPadding()`/`noTruncation()` to disable them. JSON decoder
cleanup is honored: for example `Hello, world!` decodes to `hello, world!` with
the uncased JSON pipeline. The vocab.txt API keeps its legacy spaced decoding
(`hello , world !`). `configOverride` overrides the exposed WordPiece settings
and uses the legacy CLS/SEP configuration instead of the JSON template.

Offsets and character lookup APIs use **original Unicode code-point indices**,
not normalized-text indices or Dart UTF-16 code-unit indices. For example the
`hello` in `😊 hello` spans `(2, 7)`. To extract it, use
`String.fromCharCodes(text.runes.toList().sublist(2, 7))`.
Word IDs restart at zero for each input sequence; use `sequenceIndex` for pair
word/character lookups. Inserted template and padding tokens have null word/sequence IDs; added-token
strings matched within input text retain their input alignment.

Original word IDs are retained during left truncation. This differs from HF
0.23.2's early-truncation optimization in one documented boundary case; the
fixture includes both the observed HF output and HF's full-encoding/post-process
reference result. This package does not claim universal HF pipeline compatibility.

```sh
RUN_HF_NETWORK_TESTS=1 dart test test/huggingface_network_test.dart
```

Network failures or changed model bytes fail this opt-in suite. The default
`dart test` runs the offline regression suite without downloading models.

PowerShell:

```powershell
$env:RUN_HF_NETWORK_TESTS = '1'
dart test test/huggingface_network_test.dart
Remove-Item Env:RUN_HF_NETWORK_TESTS
```

## License

The Dart package is MIT licensed. Test fixtures retain their upstream terms,
including Apache-2.0 and KLUE CC-BY-SA-4.0; see the
[fixture attribution](test/fixtures/huggingface/README.md). Fixtures and generation
tools are excluded from the published package.

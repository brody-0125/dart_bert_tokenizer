# Changelog

## 1.3.0

- Add pre-tokenized word-list encoding for single inputs, pairs, batches and
  parallel batches. Reuse the configured normalization, AddedToken extraction,
  pre-tokenizer, WordPiece, truncation, template and padding pipeline.
- Preserve input list indices as word IDs, including empty-item gaps. Offsets
  remain Unicode code-point positions within each item. AddedToken matching
  never crosses item boundaries.
- Add optional `wordIndex` filters to `charToToken` and `charToWord`. Existing
  calls retain first-match behavior; pair lookups also use `sequenceIndex`.
- Snapshot nested word lists and tokenizer settings for parallel encoding.
- Add a runnable NER alignment example, 552 pinned model pre-tokenized cases,
  and 40 synthetic HF pipeline/AddedToken cases. Preserve 1.2.0 HF edge policies.
- Validate 1,900 offline tests and 1,340 network tests. No runtime dependencies.

## 1.2.0

- Add immutable `AddedToken` definitions, `addTokens()` and `addSpecialTokens()`
  for both JSON and vocab.txt tokenizers. Preserve existing IDs and return new
  or changed registration counts; ignore identical definitions and empty input.
- Pin HF word-character ranges across Dart VM Unicode versions (including 3.10.7).
- Support Unicode single-word boundaries, left/right whitespace absorption,
  and raw-then-normalized matching with original code-point alignment.
- Keep added tokens separate from base WordPiece tries; refresh matching/decode
  caches on definition updates. Filter special IDs independently of spelling.
- Isolate registration state between tokenizer instances and snapshot settings
  for in-flight parallel batches.
- Use lowest ID for normalized-pattern ties; refresh stale decode spellings;
  allocate sparse-vocabulary IDs above the maximum to avoid collisions. These
  documented edge cases intentionally differ from HF 0.23.2.
- Promote AraBERT v02 to a successful pinned network fixture. Add HF registration,
  flag-combination, pair/batch and Unicode-boundary regression fixtures.
- Validate 1,301 offline tests and 788 network tests. No new runtime dependencies.

## 1.1.0

- Add `tokenizer.json` loading support for HuggingFace tokenizer files
  - `WordPieceTokenizer.fromTokenizerJson()` - async file loading
  - `WordPieceTokenizer.fromTokenizerJsonSync()` - sync file loading
  - `WordPieceTokenizer.fromTokenizerJsonString()` - load from JSON string
- Add `Vocabulary.fromMap()` factory for token-to-ID map construction
- Automatically extract normalizer, post-processor, and added tokens from JSON
- Support optional `configOverride`; it replaces exposed settings (including
  omitted-field defaults) and uses the legacy CLS/SEP template.
- Add pinned Hugging Face network fixtures for uncased, cased, multilingual BERT
  and MiniLM, plus reproducible offline goldens from tokenizers 0.23.2.
- Extend fixtures with KLUE Korean BERT/RoBERTa, Google/HFL Chinese BERT,
  Arabic BERT, Turkish BERT and MuRIL; verify AraBERT/IndicBERT rejection
  boundaries. Pin original JSON or vocabulary bytes and document conversion
  settings, license provenance and Japanese morphology limitations.
- Preserve original Unicode code-point offsets through normalization; fix
  non-BMP lowercasing, canonical accent removal and CJK alignment.
- Keep word IDs local to each input sequence and preserve them during truncation.
- Truncate content before adding special tokens; apply padding after pair
  construction and retain JSON settings in parallel batches.
- Honor JSON decoder cleanup, null components, exact added tokens, supported
  templates, and serialized padding/truncation settings. Reject unsupported
  pipeline components and added-token flags explicitly.
- Validate invalid lengths, worker counts, vocabulary IDs and impossible pair
  truncation requests. Support custom unknown and padding metadata.
- Keep legacy vocab.txt decoding; JSON WordPiece cleanup produces punctuation
  without preceding spaces. Offsets now consistently refer to original Unicode
  code points rather than normalized UTF-16 positions.
- Document the HF 0.23.2 early-left-truncation word-ID discrepancy; retain correct
  original word IDs and verify against HF's post-process reference path.
- Add Linux/Windows, minimum SDK/stable, analysis and network fixture CI.
  The release suite contains 1,143 offline tests and 707 network tests across
  eleven successful model pipelines and two unsupported-pipeline boundaries.
- Document vocabulary-derived fixture limits, template-defined type IDs and
  upstream fixture licenses. Remove unqualified performance estimates.

## 1.0.2

- Add project configuration files (.gitignore).
- Update .pubignore for cleaner package distribution.
  (Restored from the published 1.0.2 archive.)

## 1.0.1

- Add comprehensive dartdoc documentation to all public API elements
- Document library, classes, methods, and properties following Effective Dart guidelines
- Improve pub.dev documentation score (target: 20%+ API documentation)

## 1.0.0

- Initial release
- Pure Dart implementation of BERT WordPiece tokenizer
- Initial HuggingFace compatibility claim (superseded by the explicit supported
  pipeline scope and HF discrepancy documented in 1.1.0).
- Memory-efficient typed arrays (Int32List, Uint8List)
- Single text and sentence pair encoding
- Batch encoding (sequential and parallel with Isolates)
- Padding and truncation support
- Offset mapping (char-to-token, token-to-char, word-to-tokens)
- Vocabulary access and token conversion utilities

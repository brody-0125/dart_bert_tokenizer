# Hugging Face fixtures

`manifest.json` pins thirteen executable model fixtures: twelve successful WordPiece
pipelines and one explicitly unsupported pipelines. Repository revision and the
SHA-256 of the exact source bytes are recorded. `source_file` defaults to
`tokenizer.json`; three models instead use `vocab.txt`.

Original creators and source links are identified by `repository` in the
manifest. Google BERT, MiniLM and HFL fixtures retain Apache-2.0 terms (LICENSE).
KLUE BERT/RoBERTa vocabulary subsets are modified from the KLUE team's models:
only entries used by the oracle are retained; they remain CC-BY-SA-4.0
(LICENSE-KLUE and the upstream license link in the manifest). They are not
relicensed under this package's license. IndicBERT and Turkish BERT report MIT
in their model cards. Arabic model metadata does not specify a license; the
manifest records that limitation rather than inventing one. Fixtures are
excluded from the published Dart package.

The oracle is Python `tokenizers==0.23.2`, not a Transformers wrapper. Generate
from the repository root (or pass the script's absolute path):

```sh
python -m pip install -r scripts/requirements-fixtures.txt
python scripts/generate_hf_fixtures.py
python scripts/generate_added_token_fixtures.py
python scripts/generate_unicode_word_boundaries.py
```

Models are downloaded to `.dart_tool/hf-fixtures/`; cache hits still verify the
hash. Generated goldens are checked in and never regenerated during Dart tests.
The generator also retains only vocabulary entries used in the expected outputs
and verifies that each reduced tokenizer gives exactly the full model's results.
These small, offline pipelines are not usable replacement model vocabularies.

The default test suite compares every encoding field plus decode with/without
special tokens. Eight-item batches exercise real worker isolates, not just the
small-batch fallback. Synthetic cases cover normalization, null components,
post-processors and truncation boundaries.

Run the actual downloaded models explicitly:

```sh
RUN_HF_NETWORK_TESTS=1 dart test test/huggingface_network_test.dart
```

PowerShell:

```powershell
$env:RUN_HF_NETWORK_TESTS = '1'
dart test test/huggingface_network_test.dart
```

When enabled, HTTP failures, timeouts, hash mismatches and parser errors fail the
suite. CI runs the network job separately from offline tests.

## HF 0.23.2 left-truncation word IDs

The early-tokenization optimization in HF 0.23.2 removes leading pre-tokenized
splits and then enumerates the remainder from zero. Original offsets survive,
but original word IDs can change. For `zero one two three`, keeping the final
word yields token `three`, offset `(13, 18)` and word ID `0` through the tokenizer
without a post-processor, or `2` with two BERT special tokens. Encoding the full
input and calling `Encoding.truncate()` yields the original word ID `3`.

Dart retains original input word indices. The affected synthetic fixture keeps
the unmodified direct HF result in `encode_expected`, and separately generates
`expected` using HF's untruncated `encode` followed by `post_process`. The
generator asserts that only `word_ids` differ; no assertion is skipped. All
network-model goldens use direct `Tokenizer.encode`/`encode_batch` results.

Source: [v0.23.2 pre_tokenizer.rs](https://github.com/huggingface/tokenizers/blob/v0.23.2/tokenizers/src/tokenizer/pre_tokenizer.rs),
`tokenize_with_limit` and `into_encoding`. This is an observed discrepancy, not
a claim that an upstream issue has been acknowledged or fixed.

## Unicode normalization data

`scripts/generate_unicode_data.py` generates canonical NFD-without-Mn mappings
using Python's `unicodedata`; the generated file records its Unicode version.
Hangul decomposition is algorithmic. The table deliberately does not perform
compatibility transliteration of characters such as ø, ł or œ.

## Language-specific model coverage

- KLUE BERT/RoBERTa and Google/HFL Chinese BERT: official tokenizer JSON,
  Korean NFC/NFD, mixed Han/Latin text, simplified/traditional Chinese,
  supplementary Han characters and emoji. KLUE RoBERTa verifies distinct
  sequence IDs even when both inputs have type ID zero.
- asafaya Arabic BERT, dbmdz Turkish BERT and Google MuRIL: official vocab.txt,
  converted by the pinned `BertWordPieceTokenizer` with the manifest's explicit
  options. Turkish `do_lower_case=false` and MuRIL's explicit lowercase/accent
  settings follow tokenizer_config.json; Arabic lowercasing follows its model
  card. Remaining builder settings use the pinned HF defaults. These fixtures
  validate that stated pipeline, not an unverified Transformers AutoTokenizer
  equivalence. Network tests insert the full downloaded vocabulary into that
  generated pipeline and check against the same offline oracle.
- AraBERT v02: official JSON now succeeds, including normalized/single-word
  `[بريد]`, `[مستخدم]`, `[رابط]` and both decode modes. This does not include the
  separately recommended ArabertPreprocessor.
- IndicBERTv2: the actual Whitespace configuration must produce a specific
  FormatException offline and with the original downloaded file.
- Tohoku Japanese v3 and LINE Japanese DistilBERT remain outside executable
  coverage: their documented inference pipelines require MeCab/UniDic, with
  WordPiece and SentencePiece respectively. Loading just a vocabulary is not
  a valid substitute for testing those models.

Each successful added model runs singles, pairs, both padding/truncation
directions, all three pair truncation strategies, batch and isolate paths.
Padding explicitly uses the model's [PAD] ID (KLUE RoBERTa uses 1). Pair
truncation lengths are chosen from actual token counts so only_first/only_second
remain valid for language-specific vocabularies. Language cases include Arabic
diacritics, Turkish I/İ/ı/i, Indic scripts, combining accents and control chars.

The release suite currently has 1,300 offline tests and 788 opt-in network tests.
Both paths register model cases through `test/hf_fixture_support.dart`, including
the same error-message checks for unsupported pipelines and full-field
comparisons for sequential and parallel batches.

## AddedToken oracle and deliberate differences (1.2.0)

`added_tokens.golden.json` records dynamic registration return counts, all eight
encoding fields, decode modes and independent JSON reload results. It includes
all 32 flag combinations, Unicode inputs, mutation and pair/batch interactions.
Direct HF output is retained as `hf_expected` when Dart intentionally filters a
normalized special ID or refreshes a stale decode cache after an option change.
The expected decoder strings are generated using HF's decoder applied to the
current normalized definitions, with historical special IDs removed when needed.

Different registered spellings can normalize to the same pattern. HF 0.23.2 can
choose different IDs across runs. Dart chooses the lowest ID deterministically;
this has a separate contract test rather than an arbitrary HF golden. Sparse
model vocabularies also allocate after the maximum occupied ID rather than HF's
count-based rule. Existing IDs must never be overwritten.

`unicode_word_boundaries.json` contains ranges from probing every Unicode scalar
with HF's actual single_word matcher (null normalizer/pre-tokenizer). Tests check
both sides of each range boundary through the public encoding API, including
join controls, combining marks and supplementary-plane characters. The same
ranges generate `lib/src/unicode_word_data.dart` for runtime matching. This pins
word boundaries across VM versions: Dart 3.13.3 Unicode properties matched HF
over all scalars, but Dart 3.10.7 differed (first observed at U+0897).

# Hugging Face fixtures

`manifest.json` pins four public WordPiece models by repository, commit SHA and
the SHA-256 of the exact tokenizer.json bytes. The model metadata identifies
their Apache-2.0 license; see LICENSE and the source repositories in the manifest.

The oracle is Python `tokenizers==0.23.2`, not a Transformers wrapper. Generate
from the repository root (or pass the script's absolute path):

```sh
python -m pip install -r scripts/requirements-fixtures.txt
python scripts/generate_hf_fixtures.py
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

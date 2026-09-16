"""Probe HF 0.23.2 Unicode single_word boundaries over every Unicode scalar."""
import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Tokenizer, models

assert tokenizers.__version__ == '0.23.2'
tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
tokenizer.add_tokens([AddedToken('<T>', single_word=True, normalized=False)])
ranges = []
start = previous = None
for block in range(0, 0x110000, 4096):
    points = [cp for cp in range(block, min(block + 4096, 0x110000))
              if not 0xD800 <= cp <= 0xDFFF]
    encodings = tokenizer.encode_batch(['<T>' + chr(cp) for cp in points],
                                      add_special_tokens=False)
    for cp, encoding in zip(points, encodings):
        if encoding.ids == [0]:
            if start is None:
                start = cp
            elif previous != cp - 1:
                ranges.append([start, previous])
                start = cp
            previous = cp
        elif start is not None:
            ranges.append([start, previous])
            start = None
if start is not None:
    ranges.append([start, previous])
output = Path(__file__).resolve().parents[1] / 'test/fixtures/huggingface/unicode_word_boundaries.json'
output.write_text(json.dumps({'oracle': tokenizers.__version__, 'ranges': ranges}), encoding='utf-8')
print(len(ranges), 'Unicode word ranges')

"""Regenerate checked-in goldens using the pinned HF oracle and model files.

Run from any directory after installing requirements-fixtures.txt.
Downloaded models stay in .dart_tool/hf-fixtures, outside the package sources.
"""
import hashlib
from copy import deepcopy
import json
from pathlib import Path
from urllib.request import urlopen

import tokenizers
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "test/fixtures/huggingface"
CACHE = ROOT / ".dart_tool/hf-fixtures"
FIELDS = ("tokens", "ids", "type_ids", "attention_mask", "special_tokens_mask",
          "offsets", "word_ids", "sequence_ids")
TEXTS = ["Hello, world!", "", " \t\n", "Hello   world", "Hello\tworld\nagain",
         "café naïve résumé", "cafe\u0301", "ø ł œ Ð Þ", "你好世界 hello",
         "안녕하세요 세계", "Hello 😊 world", "a\x00b\u200dc", "[MASK]Hello[SEP]",
         "I'm happy. Don't worry!", "a" * 101, "İ 𐐀"]


def golden_json(value, level=0):
    """Keep token arrays on one line, but objects and cases reviewable."""
    indent = '  ' * (level + 1)
    closing = '  ' * level
    if isinstance(value, dict):
        items = [json.dumps(k) + ': ' + golden_json(v, level + 1) for k, v in value.items()]
        return '{\n' + ',\n'.join(indent + item for item in items) + '\n' + closing + '}'
    if isinstance(value, list) and any(isinstance(item, dict) for item in value):
        return '[\n' + ',\n'.join(indent + golden_json(item, level + 1) for item in value) + '\n' + closing + ']'
    return json.dumps(value, ensure_ascii=False)


def expected(t, e):
    result = {f: getattr(e, f) for f in FIELDS}
    result["decoded"] = t.decode(e.ids, skip_special_tokens=True)
    result["decoded_with_special_tokens"] = t.decode(e.ids, skip_special_tokens=False)
    return result


def cases(raw):
    result = []
    for i, text in enumerate(TEXTS):
        for special in (True, False):
            t = Tokenizer.from_str(raw)
            result.append({"name": f"single-{i}-special-{special}", "input": text,
                           "add_special_tokens": special,
                           "expected": expected(t, t.encode(text, add_special_tokens=special))})
    for direction in ("right", "left"):
        for strategy in ("longest_first", "only_first", "only_second"):
            t = Tokenizer.from_str(raw)
            t.enable_truncation(10, direction=direction, strategy=strategy)
            t.enable_padding(length=12, direction=direction)
            a, b = "Hello world again today", "tokenization is interesting today"
            result.append({"name": f"pair-{direction}-{strategy}", "input": a, "pair": b,
                           "truncation": {"max_length": 10, "direction": direction, "strategy": strategy},
                           "padding": {"length": 12, "direction": direction},
                           "expected": expected(t, t.encode(a, b))})
    for a, b in [("tokenization", "Hello world"), ("", ""), ("你好", "😊 hi")]:
        t = Tokenizer.from_str(raw)
        result.append({"name": f"pair-{a}-{b}", "input": a, "pair": b,
                       "expected": expected(t, t.encode(a, b))})
    for direction in ("left", "right"):
        t = Tokenizer.from_str(raw)
        t.enable_padding(direction=direction, pad_to_multiple_of=8)
        texts = ["Hello", "tokenization", "", "你好", "😊 hi", "café", "hello world", "[MASK]"]
        result.append({"name": f"batch-{direction}", "batch": texts,
                       "padding": {"direction": direction, "pad_to_multiple_of": 8},
                       "expected": [expected(t, e) for e in t.encode_batch(texts)]})
        pairs = [(text, 'hello world') for text in texts]
        t.enable_truncation(8, direction=direction)
        result.append({'name': f'pair-batch-{direction}', 'pair_batch': pairs,
                       'padding': {'direction': direction, 'pad_to_multiple_of': 8},
                       'truncation': {'max_length': 8, 'direction': direction, 'strategy': 'longest_first'},
                       'expected': [expected(t, e) for e in t.encode_batch(pairs)]})
    return result


def main():
    assert tokenizers.__version__ == "0.23.2", tokenizers.__version__
    CACHE.mkdir(parents=True, exist_ok=True)
    for model in json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8-sig")):
        path = CACHE / (model["name"] + ".json")
        if not path.exists():
            url = f'https://huggingface.co/{model["repository"]}/resolve/{model["revision"]}/tokenizer.json'
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == model["sha256"], model["name"]
        raw = data.decode("utf-8")
        pipeline = json.loads(raw)
        print(model["name"], {k: pipeline.get(k) for k in ("normalizer", "pre_tokenizer", "post_processor", "decoder", "padding", "truncation")})
        output = {"oracle": "tokenizers==0.23.2", "model": model, "cases": cases(raw)}
        (FIXTURES / (model["name"] + ".golden.json")).write_text(
            golden_json(output) + "\n", encoding="utf-8")
        # Retain exactly the vocabulary entries used by the oracle results.
        # Verify the reduced pipeline still produces the same golden values.
        used = set()
        for case in output['cases']:
            encodings = case['expected'] if 'batch' in case or 'pair_batch' in case else [case['expected']]
            for encoding in encodings:
                used.update(encoding['tokens'])
        used.update(entry['content'] for entry in pipeline['added_tokens'])
        pipeline['model']['vocab'] = {token: id for token, id in pipeline['model']['vocab'].items() if token in used}
        compact = json.dumps(pipeline, ensure_ascii=False)
        assert cases(compact) == output['cases'], model['name']
        (FIXTURES / (model['name'] + '.reduced.json')).write_text(compact + '\n', encoding='utf-8')
    generate_synthetic()


def generate_synthetic():
    base = json.loads((CACHE / 'bert-base-uncased.json').read_text(encoding='utf-8'))
    words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'hello', 'world',
             'a', 'b', 'ab', '##b', 'cafe', 'café', 'e', '##x', '😊', '𐐨', ',', '!',
             'ø', 'ł', 'œ', 'þ', 'ð', 'i', '##i']
    base['model']['vocab'] = {word: i for i, word in enumerate(words)}
    for token in base['added_tokens']:
        token['id'] = words.index(token['content'])
    for token in base['post_processor']['special_tokens'].values():
        token['ids'] = [words.index(token['tokens'][0])]
    result = []

    pipelines = []

    def add(name, pipeline, text, pair=None, special=True, padding=None, truncation=None):
        if pipeline not in pipelines:
            pipelines.append(deepcopy(pipeline))
        raw = json.dumps(pipeline, ensure_ascii=False)
        t = Tokenizer.from_str(raw)
        if padding:
            t.enable_padding(**padding)
        if truncation:
            t.enable_truncation(**truncation)
        e = t.encode(text, pair, add_special_tokens=special)
        fixture = {'name': name, 'tokenizer_index': pipelines.index(pipeline), 'input': text,
                       'pair': pair, 'add_special_tokens': special,
                       'padding': padding, 'truncation': truncation, 'expected': expected(t, e)}
        if truncation and pair is None and truncation.get('direction') == 'left':
            # HF 0.23.2's early left-truncation path renumbers surviving words.
            # Preserve its observed output and separately generate the reference
            # via HF post_process, which retains original input word indices.
            full = Tokenizer.from_str(raw)
            full.no_truncation()
            full.no_padding()
            a = full.encode(text, add_special_tokens=False)
            b = None if pair is None else full.encode(pair, add_special_tokens=False)
            reference = expected(t, t.post_process(a, b, add_special_tokens=special))
            if reference != fixture['expected']:
                assert {k: v for k, v in reference.items() if k != 'word_ids'} == {
                    k: v for k, v in fixture['expected'].items() if k != 'word_ids'}, (name, {k: (reference[k], fixture['expected'][k]) for k in reference if reference[k] != fixture['expected'][k]})
                fixture['encode_expected'] = fixture['expected']
                fixture['oracle_path'] = 'untruncated encode -> HF post_process; retains original word IDs'
                fixture['expected'] = reference
        result.append(fixture)

    for text in ['a\x00b', '😊 hello', '𐐀 hello', 'cafe\u0301 hello', 'café hello',
                 'ø ł œ Þ Ð', '你好hello', '[MASK]hello[SEP]', 'İ hello']:
        add('unicode-' + repr(text), base, text)
    for component in ['normalizer', 'pre_tokenizer', 'post_processor', 'decoder']:
        pipeline = deepcopy(base)
        pipeline[component] = None
        add('null-' + component, pipeline, 'café Hello, world!', 'a b')
    pipeline = deepcopy(base)
    pipeline['normalizer']['clean_text'] = False
    add('clean-text-false', pipeline, 'a\x00b\u200d world')
    pipeline = deepcopy(base)
    pipeline['normalizer']['lowercase'] = False
    pipeline['normalizer']['strip_accents'] = False
    add('no-normalization', pipeline, 'café 😊')
    pipeline = deepcopy(base)
    pipeline['model']['unk_token'] = 'missing'
    pipeline['model']['vocab']['missing'] = 30
    add('custom-unknown', pipeline, 'unrecognized')
    pipeline = deepcopy(base)
    pipeline['post_processor'] = {'type': 'BertProcessing', 'sep': ['[SEP]', 3], 'cls': ['[CLS]', 2]}
    add('bert-processing', pipeline, 'hello', 'world')
    pipeline = deepcopy(base)
    for part in pipeline['post_processor']['pair']:
        if 'Sequence' in part: part['Sequence']['type_id'] = 0
        if 'SpecialToken' in part: part['SpecialToken']['type_id'] = 0
    add('sequence-id-independent-of-type-id', pipeline, 'hello', 'world')
    pipeline = deepcopy(base)
    pipeline['padding'] = {'strategy': {'Fixed': 8}, 'direction': 'Left',
        'pad_to_multiple_of': None, 'pad_id': 42, 'pad_type_id': 7, 'pad_token': '<pad>'}
    pipeline['model']['vocab']['<pad>'] = 42
    pipeline['added_tokens'].append({'id': 42, 'content': '<pad>', 'special': True,
        'normalized': False, 'single_word': False, 'lstrip': False, 'rstrip': False})
    add('custom-padding-metadata', pipeline, 'hello')
    pipeline = deepcopy(base)
    pipeline['model']['continuing_subword_prefix'] = '@@'
    pipeline['model']['vocab']['@@b'] = 31
    pipeline['decoder']['prefix'] = '@@'
    add('custom-subword-prefix', pipeline, 'abb')
    pipeline = deepcopy(base)
    pipeline['model']['vocab']['##😊'] = 31
    pipeline['model']['max_input_chars_per_word'] = 2
    add('word-limit-counts-codepoints', pipeline, '😊😊 😊😊😊')
    pipeline = deepcopy(base)
    pipeline['added_tokens'].append({'id': 31, 'content': '<tag>', 'special': True,
        'normalized': False, 'single_word': False, 'lstrip': False, 'rstrip': False})
    add('custom-added-token', pipeline, 'hello<tag>world')
    for direction in ['left', 'right']:
        for length in [2, 3, 4, 5]:
            add(f'single-truncate-{direction}-{length}', base, 'hello world a b',
                truncation={'max_length': length, 'direction': direction})
        for a in ['hello', 'hello world', 'hello world a']:
            for b in ['hello', 'hello world', 'hello world a']:
                for length in [3, 4, 5]:
                    add(f'pair-truncate-{direction}-{a}-{b}-{length}', base, a, b,
                        truncation={'max_length': length, 'direction': direction})
    add('truncation-without-specials', base, 'hello world a', 'a b world', special=False,
        truncation={'max_length': 4})
    (FIXTURES / 'synthetic.golden.json').write_text(
        golden_json({'oracle': 'tokenizers==0.23.2', 'pipelines': pipelines, 'cases': result}) + '\n', encoding='utf-8')


if __name__ == "__main__":
    main()

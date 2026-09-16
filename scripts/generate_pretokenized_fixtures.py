"""Synthetic pre-tokenized pipeline interactions, pinned to HF 0.23.2."""
from copy import deepcopy
from itertools import product
import json
import tokenizers
from tokenizers import Tokenizer, AddedToken
from generate_hf_fixtures import FIXTURES, expected, golden_json

assert tokenizers.__version__ == '0.23.2'
base = json.loads((FIXTURES/'bert-base-uncased.reduced.json').read_text(encoding='utf-8'))
vocab = base['model']['vocab']
base['model']['vocab'] = {word:i for i,word in enumerate(sorted(vocab,key=vocab.get))}
vocab = base['model']['vocab']
for added in base['added_tokens']: added['id'] = vocab[added['content']]
for special in base['post_processor']['special_tokens'].values():
    special['ids'] = [vocab[word] for word in special['tokens']]
rows = []

def add(name, pipeline, words, pair=None, additions=None):
    t = Tokenizer.from_str(json.dumps(pipeline))
    additions = additions or []
    t.add_tokens([AddedToken(**a) for a in additions])
    e = t.encode(words, pair, is_pretokenized=True)
    observed = expected(t,e)
    reference = deepcopy(observed)
    specials = {i for i,a in t.get_added_tokens_decoder().items() if a.special}
    reference['decoded'] = t.decode([i for i in e.ids if i not in specials],skip_special_tokens=False)
    rows.append({'name': name, 'pipeline': pipeline, 'additions': additions,
        'fixture': {'name': name, 'pretokenized': True, 'input': words,
                    'pair': pair, 'expected': reference}})
    if reference != observed:
        rows[-1]['hf_expected'] = observed
        rows[-1]['difference'] = 'Preserve 1.2.0 special-ID filtering policy for normalized AddedTokens.'

for flags in product((False, True), repeat=5):
    additions = [{'content': '<X>', **dict(zip(('single_word','lstrip','rstrip','normalized','special'),flags))}]
    add('added-flags-'+''.join(map(str,map(int,flags))), base,
        ['a', '  <X>  ', 'b', 'a<X>b', '<x>', ''], additions=additions)
for component in ('normalizer','pre_tokenizer','post_processor','decoder'):
    pipeline = deepcopy(base)
    pipeline[component] = None
    add('null-'+component, pipeline, ['Hello,', 'CAFÉ', 'a b', '😀'], ['world',''])
pipeline = deepcopy(base)
pipeline['post_processor'] = {'type':'BertProcessing', 'cls':['[CLS]',vocab['[CLS]']], 'sep':['[SEP]',vocab['[SEP]']]}
add('bert-processing', pipeline, ['Hello,'], ['world'])
pipeline = deepcopy(base)
for part in pipeline['post_processor']['pair']:
    for spec in part.values(): spec['type_id'] = 0
add('same-type-ids', pipeline, ['Hello,'], ['world'])
add('normalized-added', base, ['CAFÉ','Cafe\u0301','<X>',''], additions=[{'content':'CAFÉ','normalized':True}])
add('cross-item-added', base, ['hello','world','hello world'], additions=[{'content':'hello world'}])
(FIXTURES/'pretokenized.golden.json').write_text(golden_json({
    'oracle':'tokenizers==0.23.2', 'cases':rows})+'\n',encoding='utf-8')
print(f'{len(rows)} synthetic pre-tokenized cases generated')

"""Generate AddedToken registration/matching goldens with tokenizers 0.23.2.

Every case keeps direct HF outputs. The documented normalized-special decode
exception additionally records the ID-based filtering expected from Dart.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import tokenizers
from tokenizers import Tokenizer, AddedToken, models, normalizers, pre_tokenizers, decoders, processors
from generate_hf_fixtures import golden_json, expected

ROOT = Path(__file__).resolve().parents[1]

def base():
    t = Tokenizer(models.WordPiece({'[UNK]':0,'hello':1,'world':2,'cat':3,'cafe':4,'!':5}, unk_token='[UNK]'))
    t.normalizer = normalizers.BertNormalizer(lowercase=True)
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.decoder = decoders.WordPiece()
    return t

cases=[]
def add(name, additions, text, normalizer='default'):
    t=base()
    if normalizer is None: t.normalizer=None
    elif isinstance(normalizer, dict): t.normalizer=normalizers.BertNormalizer(**normalizer)
    case={'name':name,'initial':json.loads(t.to_str()),'input':text,'steps':[]}
    historical_special = set()
    for method, items in additions:
        count=getattr(t,method)([AddedToken(**x) if isinstance(x,dict) else x for x in items])
        pipeline=json.loads(t.to_str())
        e=t.encode(text)
        raw=expected(t,e)
        dart=deepcopy(raw)
        special={x['id'] for x in pipeline['added_tokens'] if x['special']}
        historical_special.update(special)
        # Recompute from current definitions: HF can keep a stale normalized
        # decode cache after normalized=true is changed to false.
        canonical={id:token for token,id in pipeline['model']['vocab'].items()}
        for token in pipeline['added_tokens']:
            canonical[token['id']]=(t.normalizer.normalize_str(token['content'])
                if token['normalized'] and t.normalizer else token['content'])
        for skip,key in [(True,'decoded'),(False,'decoded_with_special_tokens')]:
            pieces=[canonical[i] for i in e.ids if not skip or i not in historical_special]
            dart[key]=t.decoder.decode(pieces) if t.decoder else ' '.join(pieces)
        step={'method':method,'tokens':items,'count':count,'expected':dart,'pipeline':pipeline}
        if dart != raw:
            step['hf_expected']=raw
            step['difference']='Filter special IDs and refresh decode spellings after definition changes; see fixture README.'
        # JSON reload may differ from historical registration state (HF keeps
        # special status after demotion). Preserve and test that separately.
        loaded=Tokenizer.from_str(t.to_str())
        re=loaded.encode(text)
        reload=expected(loaded,re)
        reload['decoded']=loaded.decode([i for i in re.ids if i not in special])
        step['reloaded_expected']=reload
        case['steps'].append(step)
    cases.append(case)

for suffix in ['', 's','_','1','é','中','한','\u0301','\u200c','\u200d','😊','-','\u203f']:
    add('boundary-'+repr(suffix), [('add_tokens',[dict(content='zz',single_word=True,normalized=False)])],f'zz{suffix} {suffix}zz')
for single,left,right,normalized,special in product([False,True],repeat=5):
    opts=dict(content='<X>',single_word=single,lstrip=left,rstrip=right,normalized=normalized,special=special)
    add(f'flags-{single}-{left}-{right}-{normalized}-{special}', [('add_tokens',[opts])], 'hello\t\u00a0<X>\u2003\nworld <X>! a<X>b')
for name,content,text,normalizer in [
    ('accent','CAFÉ','CAFÉ cafe cafe\u0301','default'),
    ('dotted-i','İ','İ i I',dict(lowercase=True,strip_accents=False)),
    ('astral','𐐀','𐐀𐐨 hello','default'),
    ('hangul','각','각 각',dict(lowercase=False,strip_accents=True)),
    ('cjk','中','中中文','default'),
    ('null','CAFÉ','CAFÉ cafe',None),
    ('empty-normalized','\u0301','x\u0301','default'),
    ('controls','ab','a\x00b a\u200db','default'),
]:
    add('normalized-'+name,[('add_tokens',[dict(content=content,normalized=True)])],text,normalizer)
add('raw-before-normalized',[('add_tokens',[dict(content='HELLO',normalized=False),dict(content='hello world',normalized=True)])],'HELLO world')
add('leftmost-longest',[('add_tokens',[dict(content='ab',normalized=False),dict(content='abc',normalized=False),dict(content='bc',normalized=False)])],'abc ab')
add('boundary-failed-longest',[('add_tokens',[dict(content='cat',single_word=True,normalized=False),dict(content='ca',normalized=False)])],'cats')
add('adjacent-whitespace',[('add_tokens',[dict(content='<A>',rstrip=True,normalized=False),dict(content='<B>',lstrip=True,normalized=False)])],'<A>   <B>')
add('registration',[('add_tokens',['cat','cat','','new']),('add_tokens',['new']),('add_tokens',[dict(content='new',single_word=True)]),('add_special_tokens',['new'])],'cat new')
add('special-demotion',[('add_special_tokens',['<X>']),('add_tokens',[dict(content='<X>',special=False,normalized=False)])],'<X>')
add('normalize-special',[('add_tokens',[dict(content='CAFÉ',normalized=True,special=True)])],'café')
add('change-normalized',[('add_tokens',[dict(content='CAFÉ',normalized=True)]),('add_tokens',[dict(content='CAFÉ',normalized=False)])],'CAFÉ café')
interactions=[]
for direction,strategy in product(['left','right'],['longest_first','only_first','only_second']):
    t=base()
    t.add_special_tokens(['[CLS]','[SEP]'])
    t.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1',special_tokens=[('[CLS]',6),('[SEP]',7)])
    initial=json.loads(t.to_str())
    additions=[dict(content='<X>',normalized=True,lstrip=True,rstrip=True),
               dict(content='zz',normalized=False,single_word=True)]
    t.add_tokens([AddedToken(**x) for x in additions])
    t.enable_truncation(9,direction=direction,strategy=strategy)
    t.enable_padding(length=12,direction=direction)
    a,b='hello <X> zz world hello','world zz <X> hello'
    fixture={'name':f'dynamic-pair-{direction}-{strategy}','input':a,'pair':b,
        'padding':{'length':12,'direction':direction},
        'truncation':{'max_length':9,'direction':direction,'strategy':strategy},
        'expected':expected(t,t.encode(a,b))}
    interactions.append({'initial':initial,'additions':additions,'fixture':fixture})
    pairs=[(a,b),(b,a),('',''),('hello','<X>'),(a,''),('',b),('zz!','hello'),('hello','world')]
    if strategy == 'longest_first':
        batch=dict(fixture,name=f'dynamic-pair-batch-{direction}',pair_batch=pairs,
                   expected=[expected(t,e) for e in t.encode_batch(pairs)])
        interactions.append({'initial':initial,'additions':additions,'fixture':batch})
assert tokenizers.__version__ == '0.23.2'
(ROOT/'test/fixtures/huggingface/added_tokens.golden.json').write_text(
    golden_json({'oracle':'tokenizers==0.23.2','cases':cases,'interactions':interactions})+'\n',encoding='utf-8')
print(len(cases),'AddedToken cases')

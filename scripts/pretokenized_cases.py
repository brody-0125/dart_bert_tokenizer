"""HF oracle cases shared by full and reduced model generation."""
from tokenizers import Tokenizer


def cases(raw, expected):
    results = []
    inputs = [[], [''], ['', 'hello', ' ', '\x00', 'world', ''],
              ['Hello,', 'world!'], ['hello world', 'next'],
              ['CAFÉ', 'Cafe\u0301', 'İ', '😀hello', '中文'],
              ['unaffordable', 'playing', 'a'*100, 'a'*101],
              ['대한민국', '서울에서', '일합니다.'], ['北京市', '欢迎你'],
              ['東京', 'で', '働く', 'Café'], ['مرحبا', 'بالعالم!'],
              ['İstanbul', 'IĞDIR', 'ıslak'], ['नमस्ते', 'தமிழ்', 'తెలుగు'],
              ['[CLS]', '[MASK]hello', '[SEP]'], ['가', '가', 'a\u200db', '𠀀😀']]

    def result(t, e):
        value = expected(t, e)
        value['alignment'] = [
            [seq, word, e.word_to_tokens(word, seq), e.word_to_chars(word, seq)]
            for seq in sorted({s for s in e.sequence_ids if s is not None}) for word in range(8)]
        return value

    for i, words in enumerate(inputs):
        for special in (True, False):
            t = Tokenizer.from_str(raw)
            results.append({'name': f'pretokenized-single-{i}-{special}',
                'pretokenized': True, 'input': words, 'add_special_tokens': special,
                'expected': result(t, t.encode(words, is_pretokenized=True, add_special_tokens=special))})
    for a, b in [([], []), ([], ['world']), (['hello'], []),
                 (['Hello,', '', 'world'], ['world!', 'hello'])]:
        t = Tokenizer.from_str(raw)
        results.append({'name': f'pretokenized-pair-{len(results)}', 'pretokenized': True,
            'input': a, 'pair': b, 'expected': result(t, t.encode(a, b, is_pretokenized=True))})
    for direction in ('left', 'right'):
        for strategy in ('longest_first', 'only_first', 'only_second'):
            t = Tokenizer.from_str(raw)
            a, b = ['hello', 'unaffordable', 'world!'], ['hello', 'world!', 'again']
            sizes = [len(t.encode(x, is_pretokenized=True, add_special_tokens=False)) for x in (a,b)]
            limit = max(sizes) + t.num_special_tokens_to_add(True) + 1
            truncation = {'max_length': limit, 'direction': direction, 'strategy': strategy}
            t.enable_truncation(**truncation)
            results.append({'name': f'pretokenized-truncate-{direction}-{strategy}', 'pretokenized': True,
                'input': a, 'pair': b, 'truncation': truncation,
                'expected': result(t, t.encode(a,b,is_pretokenized=True))})
        t = Tokenizer.from_str(raw)
        truncation = {'max_length': 5, 'direction': direction}
        t.enable_truncation(**truncation)
        words = ['hello', 'unaffordable', 'world!']
        results.append({'name': f'pretokenized-single-truncate-{direction}', 'pretokenized': True,
            'input': words, 'truncation': truncation,
            'expected': result(t, t.encode(words,is_pretokenized=True))})
        for pair in (False, True):
            t = Tokenizer.from_str(raw)
            padding = {'direction': direction, 'pad_to_multiple_of': 8}
            t.enable_padding(**padding, pad_id=t.token_to_id('[PAD]'))
            batch = inputs[:8]
            if pair: batch = [(a, ['hello','world']) for a in batch]
            results.append({'name': f'pretokenized-batch-{direction}-{pair}', 'pretokenized': True,
                'pair_batch' if pair else 'batch': batch, 'padding': padding,
                'expected': [result(t,e) for e in t.encode_batch(batch,is_pretokenized=True)]})
    return results

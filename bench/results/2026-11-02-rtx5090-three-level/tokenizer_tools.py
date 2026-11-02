import json,os,pathlib,sys
BASE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path(os.environ.get('STRATA_BENCH_REPO',str(BASE.parents[2])))
ASSETS=pathlib.Path(os.environ.get('STRATA_BENCH_ASSETS',str(ROOT/'models')))
TOKENIZERS={'q36':ASSETS/'qwen36/tokenizer','q38':ASSETS/'pack-iq2_xs/tokenizer'}

def load_tokenizer(directory):
    sys.path.insert(0,str(ROOT/'tools'))
    from strata_tokenizer import Tokenizer
    vocab=json.loads((directory/'vocab.json').read_text());tokens=[None]*len(vocab)
    for token,i in vocab.items():tokens[i]=token
    cfg=json.loads((directory/'tokenizer.json').read_text())
    merges=(directory/'merges.txt').read_text().split('\n');types=json.loads((directory/'token_type.json').read_text())
    tok=Tokenizer(tokens,merges,types,cfg.get('pre') or 'qwen35',cfg.get('special_ids') or None)
    return tok,cfg

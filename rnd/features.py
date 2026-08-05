"""Extract and cache frozen-BERT CLS features for AAPD.

The RND derivation assumes the trainable part is a single linear head on fixed
features, which matches CoMAL's --freeze_bert setting. We therefore cache h(x) once
and do all active-learning arithmetic on top of it.
"""
import json
import os
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer


def load_split(data_dir, split, maxlength=256):
    path = os.path.join(data_dir, 'clf_%s_data_%d.json' % (split, maxlength))
    return json.load(open(path))


def load_label2id(data_dir):
    freq = json.load(open(os.path.join(data_dir, 'label_freq.json')))
    return {name: i for i, (name, _) in enumerate(freq)}, [c for _, c in freq]


def encode(data, label2id, tokenizer, maxlength):
    ids, masks, types = [], [], []
    Y = torch.zeros(len(data), len(label2id))
    for i, item in enumerate(data):
        x = [tokenizer.cls_token_id] + item['input_ids'][:maxlength - 2] + [tokenizer.sep_token_id]
        m = [1] * len(x)
        x = x + [tokenizer.pad_token_id] * (maxlength - len(x))
        m = m + [0] * (maxlength - len(m))
        ids.append(x)
        masks.append(m)
        types.append([0] * maxlength)
        for name in set(item['label']) & set(label2id):
            Y[i, label2id[name]] = 1.0
    return (torch.tensor(ids), torch.tensor(types), torch.tensor(masks)), Y


@torch.no_grad()
def extract(data, label2id, bert_path, maxlength=256, batch_size=32, normalize=True):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    cfg = BertConfig.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path, config=cfg).eval()
    (ids, types, masks), Y = encode(data, label2id, tokenizer, maxlength)
    feats = []
    for i in range(0, len(ids), batch_size):
        out = model(input_ids=ids[i:i + batch_size],
                    token_type_ids=types[i:i + batch_size],
                    attention_mask=masks[i:i + batch_size])
        feats.append(out[1])
    H = torch.cat(feats).double()
    if normalize:
        H = F.normalize(H, dim=-1)
    return H, Y.double()


def cached(cache_path, build):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    obj = build()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(obj, cache_path)
    return obj

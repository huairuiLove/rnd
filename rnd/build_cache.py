"""Cache frozen BERT features for the AAPD subsets A1 needs."""
import os
import time

import torch
torch.set_num_threads(os.cpu_count())
from rnd.features import load_split, load_label2id, extract

DATA = './data/aapd_54'
BERT = './bert/bert-base-uncased'
OUT = './rnd/cache'
LIMITS = {'train': 6000, 'val': 2000}


def main():
    os.makedirs(OUT, exist_ok=True)
    label2id, freq = load_label2id(DATA)
    torch.save({'label2id': label2id, 'freq': freq}, os.path.join(OUT, 'labels.pt'))
    for split, lim in LIMITS.items():
        path = os.path.join(OUT, '%s_%d.pt' % (split, lim))
        if os.path.exists(path):
            print('skip', split, flush=True)
            continue
        data = load_split(DATA, split)[:lim]
        t = time.time()
        H, Y = extract(data, label2id, BERT, batch_size=32)
        torch.save({'H': H, 'Y': Y}, path)
        print('%s: %s feats, %.2f pos/sample, %.1fs' % (
            split, tuple(H.shape), Y.sum(1).mean().item(), time.time() - t), flush=True)
    print('done')


if __name__ == '__main__':
    main()

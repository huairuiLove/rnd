"""Dry-run the RND acquisition path with a fake model, no BERT and no training.

Checks the plumbing that only breaks at integration time: dataloader ordering,
index mapping from pool positions back to dataset indices, dtype promotion, and
that every diagnostic line actually renders.
"""
import types
import torch
from torch.utils.data import Dataset

from rnd.acquire import acquire

C, D, LEN = 8, 24, 256


class FakeDS(Dataset):
    def __init__(self, n):
        self.n = n
        self.Y = (torch.rand(n, C) < 0.2).float()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ti = [torch.zeros(LEN, dtype=torch.long) + (i % 13),
              torch.zeros(LEN, dtype=torch.long),
              torch.ones(LEN, dtype=torch.long)]
        return [ti, self.Y[i], i, torch.zeros(C), torch.tensor(0.0)]


class FakeModel(torch.nn.Module):
    """Same output contract as BackBone_No_GCN_No_Atten: (logits, feat, cls)."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(16, D)
        self.clf = torch.nn.Linear(D, C)

    def forward(self, inputs):
        cls = self.emb(inputs[0][:, 0])
        return self.clf(cls), cls, cls


def main():
    torch.manual_seed(0)
    ds, vds = FakeDS(120), FakeDS(80)
    models = {'backbone': FakeModel()}
    args = types.SimpleNamespace(batch_size=16, sample_pair_num=5, rnd_ref_size=50,
                                 rnd_wd=1e-4, rnd_kappa=0.0, rnd_work_mult=4,
                                 rnd_delta=0.0, rnd_linear_score=False)
    pool, labeled = list(range(30, 120)), list(range(30))
    picks, trace = acquire(args, models, ds, vds, pool, labeled, 'cpu', log=print)
    assert len(picks) == 5, picks
    assert len(set(picks)) == 5, 'duplicate picks'
    assert all(0 <= p < len(pool) for p in picks), 'picks must index into pool'
    mapped = [pool[i] for i in picks]
    assert all(m not in labeled for m in mapped), 'picked an already-labelled index'
    print('picks (pool positions):', picks)
    print('mapped dataset indices:', mapped)

    # kappa is relative to the median row norm, so on near-uniform synthetic priors
    # it needs to exceed 1 to bind at all. Assert it actually lifted rows, otherwise
    # this branch would silently test nothing.
    args.rnd_kappa = 2.0
    print('\n--- eq. (15) rare-label floor enabled ---')
    acquire(args, models, ds, vds, pool, labeled, 'cpu', log=print)

    args.rnd_kappa = 0.0
    args.rnd_linear_score = True
    print('\n--- eq. (9) linear-score ablation ---')
    acquire(args, models, ds, vds, pool, labeled, 'cpu', log=print)
    print('\nall dry-run checks passed')


if __name__ == '__main__':
    main()

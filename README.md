# CoMAL
##### Official implementation of CoMAL
-----
## Run the code

./scripts/[rcv/aapd/jd]/****.sh

our.sh for our method CoMAL

## AAPD data

The AAPD experiments in the paper use 55,840 examples, 54 labels, and an
average of about 2.41 labels per example. The original files included in
`data/aapd` are a different 37,464-example, 145-label arXiv dataset and do not
match Table 1 of the paper.

The matching raw AAPD release is stored in `data/aapd_54/raw`. Validate and
rebuild the paper split and BERT-tokenized classifier files with:

```bash
python aapd_data_process.py --validate-only
python aapd_data_process.py --force
```

The preprocessing concatenates the standard 53,840/1,000/1,000 release,
shuffles with seed 1, and writes a 45,840/5,000/5,000 train/validation/test
split. The paper reports these sizes but not the exact split indices; the seed
and algorithm are recorded in `data/aapd_54/split_manifest.json`.

The AAPD run scripts use `data/aapd_54`. When a validation JSON is present,
checkpoint selection uses it and the final cycle metric is computed on the
test JSON.

python ./main.py \
--data_dir ./data/aapd_54 \
--maxlength 256 \
--batch_size 64 \
--unlab_batch_size 64 \
--total_patience 6 \
--save_path ./model_saved/aapd \
--lr 1e-4 \
--gpuid '0' \
--cycles 51 \
--try_id 1 \
--seed 1 \
--method_type rnd \
--init_example_num 100 \
--well_init_lower_bound 1 \
--sample_pair_num 100 \
--well_init \
--freeze_bert \
--freeze_layer_num 9 \
--rnd_ref_size 2000 \
--rnd_wd 1e-4 \
--rnd_kappa 0.0 \
--rnd_work_mult 10

# ablation knobs (see doc section 14):
# --rnd_delta_rel 1e-3   # damping relative to the Fisher scale (tuning knob, not derived)
# --rnd_normalize        # L2-normalise features and refit the head (coordinate-system ablation)

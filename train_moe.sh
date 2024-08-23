python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'VarNetBoundMOEL1OL' \
    --model.num_cascades 12 \
    --model.overlap_window 6 \
    --model.l1_alpha 1.0 \
    --data 'SliceDataModule' \
    --data.root '/Data'
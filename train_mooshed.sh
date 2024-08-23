python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model.num_cascades 12 \
    --model 'VarNetSharedMooshedOL' \
    --data 'SliceDataModule' \
    --data.root '/Data' \

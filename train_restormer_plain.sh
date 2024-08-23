python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'FreezedVarNetRestormerOL' \
    --model.varnet_path 'checkpoints/8cascades-plain.pt' \
    --model.varnet_with_grappa False \
    --model.with_grappa False \
    --model.restormer_dim 16 \
    --data 'SliceGrappaDataModule' \
    --data.root '/Data' \

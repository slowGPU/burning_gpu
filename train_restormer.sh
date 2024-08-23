python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'FreezedVarNetRestormerOL' \
    --model.varnet_path 'checkpoints/18cascades-alpha1.0-9517.pt' \
    --model.with_grappa False \
    --data 'SliceGrappaDataModule' \
    --data.root '/Data' \

python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'FreezedVarNetCascadedGazeOL' \
    --model.varnet_path 'checkpoints/8cascades-plain.pt' \
    --model.varnet_with_grappa False \
    --model.with_grappa False \
    --data 'SliceGrappaDataModule' \
    --data.root '/Data' \

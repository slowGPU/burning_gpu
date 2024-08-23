python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'FreezedVarNetNAFNetOL' \
    --model.nafnet_width 32 \
    --model.with_grappa False \
    --model.varnet_path 'checkpoints/Toy.pt' \
    --data 'SliceGrappaDataModule' \
    --data.root '/Data' \

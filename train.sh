python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'FreezedVarNetNAFNetOL' \
    --model.varnet_path 'checkpoints/VarNetLogisticUnetSensFix.pt' \
    --data 'SliceGrappaDataModule' \
    --data.root '/home/Data' \
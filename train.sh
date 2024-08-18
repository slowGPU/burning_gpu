python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'VarNetLogisticBoundOL' \
    --data 'SliceDataModule' \
    --data.root '/home/Data' \

# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'VarNetLogisticMambaOL' \
#     --data 'SliceDataModule' \
#     --data.root '/home/Data' \

# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'FreezedVarNetNAFNetOL' \
#     --model.varnet_path 'checkpoints/VarNetLogisticUnetSensFix.pt' \
#     --data 'SliceGrappaDataModule' \
#     --data.root '/home/Data' \
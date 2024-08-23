# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'VarNetLogisticBoundOL' \
#     --data 'SliceDataModule' \
#     --data.root '/home/Data' \

python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'FreezedVarNetNBNetOL' \
    --model.varnet_path 'checkpoints/12cascades-nafnet-9682.pt' \
    --optimizer Adam \
    --optimizer.lr=0.0002 \
    --optimizer.weight_decay=1e-8 \
    --data 'SliceGrappaDataModule' \
    --data.root '/home/Data' \

# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'FreezedVarNetNAFNetOL' \
#     --model.varnet_path 'checkpoints/VarNetLogisticUnetSensFix.pt' \
#     --data 'SliceGrappaDataModule' \
#     --data.root '/home/Data' \
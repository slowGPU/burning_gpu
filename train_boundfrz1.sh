# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'VarNetLogisticBoundOL' \
#     --data 'SliceDataModule' \
#     --data.root '/home/Data' \

python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'VarNetFreezedSensNAFNetL1OL' \
    --model.num_cascades 8 \
    --model.nafnet_width 64 \
    --model.sens_net_path 'checkpoints/sens_net/12cascades-bound.pt' \
    --model.l1_lambda 0.1 \
    --data 'SliceGrappaDataModule' \
    --data.root '/Data' \

# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'FreezedVarNetNAFNetOL' \
#     --model.varnet_path 'checkpoints/VarNetLogisticUnetSensFix.pt' \
#     --data 'SliceGrappaDataModule' \
#     --data.root '/home/Data' \
# python main.py fit \
# 	--trainer.max_epochs 15 \
# 	--seed_everything 42 \
# 	--model 'VarNetLogisticUnetSensFixOL' \
# 	--data 'SliceDataModule' \
# 	--data.root '/home/Data' \
# 	--model.num_cascades 30

# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'VarNetLogisticBoundOL' \
#     --data 'SliceDataModule' \
#     --data.root '/home/Data' \

# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'VarNetLogisticBoundFullOL' \
#     --data 'SliceGrappaDataModule' \
#     --data.root '/Data' \

# python main.py fit \
#     --trainer.max_epochs 15 \
#     --seed_everything 42 \
#     --model 'FreezedVarNetNAFNetOL' \
#     --model.varnet_path 'checkpoints/VarNetLogisticUnetSensFix.pt' \
#     --data 'SliceGrappaDataModule' \
#     --data.root '/home/Data' \
    # --model 'FreezedVarNetNAFNetOL' \
    # --model.varnet_path 'checkpoints/30cascades.pth' \
    # --data 'SliceGrappaDataModule' \
    # --data.root '/home/Data'

python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'VarNetFreezedSensToyNAFNetOL' \
    --data 'SliceDataModule' \
    --data.root '/home/Data' \
    --model.sens_net_path 'checkpoints/good_sens_net.pth' \
    --data.batch_size 1

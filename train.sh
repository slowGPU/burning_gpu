python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'VarNetFreezedSensNAFNetOL' \
    --model.sens_net_path 'checkpoints/sens_net/VarNetLogisticUnetSensFix.pt' \
    --model.num_cascades 10 \
    --model.nafnet_width 16 \
    --data 'SliceGrappaDataModule' \
    --data.root '/home/Data' \
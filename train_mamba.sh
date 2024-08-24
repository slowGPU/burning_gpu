python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'VarNetMambaOL' \
    --data 'SliceDataModule' \
    --data.root '/home/Data'
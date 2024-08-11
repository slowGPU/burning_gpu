python main.py fit \
    --trainer.max_epochs 15 \
    --seed_everything 42 \
    --model 'VarNetLogisticSensOL' \
    --model.num_cascades 4 \
    --data.root '/home/Data' \
  
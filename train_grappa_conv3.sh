python main.py fit \
	--trainer.max_epochs 15 \
	--seed_everything 42 \
	--model 'GrappaConv3NAFNetOL' \
	--data 'SliceGrappaDataModule' \
	--data.root '/home/Data' \

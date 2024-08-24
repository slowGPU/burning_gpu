python main.py fit \
	--trainer.max_epochs 15 \
	--seed_everything 42 \
	--model 'GrappaNAFNetMSEOL' \
	--data 'SliceGrappaDataModule' \
	--data.root '/home/Data' \

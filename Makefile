train:
	cd nanoGPT && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_gnn/train_adapt_gpt_config.py --model gpt && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_gnn/train_adapt_gpt_config.py --model llama
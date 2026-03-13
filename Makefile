train:
	cd nanoGPT && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_feather/train_adapt_gpt_config.py --model gpt && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_feather/train_adapt_gpt_config.py --model llama && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_netlsd/train_adapt_gpt_config.py --model gpt && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_netlsd/train_adapt_gpt_config.py --model llama && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_gnn/train_adapt_gpt_config.py --model gpt && \
	python train_pad_gemb_ar_eval.py --train_config_path data/10_nodes_gnn/train_adapt_gpt_config.py --model llama
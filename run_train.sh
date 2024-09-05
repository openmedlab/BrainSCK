python run_brainsck_train.py \
--dataset_type adhd --gpu_id 0 --max_epochs 100 \
--pretrain_weight ./checkpoints/hcp_pretrain/hcp_epoch10.pth \      ### HCP pretrain model weight
--train_list train.csv \                                            ### path to train list scv
--valid_list valid.csv \                                            ### path to valid list scv
--save_model_path ./checkpoints/adhd_model/                         ### path to save model
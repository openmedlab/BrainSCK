python run_brainsck_test.py \
--dataset_type adhd --gpu_id 0 \
--test_list ../../BrainSCK/local_data/adhd/test.csv \           ### path to test list scv
--load_model_path ./checkpoints/adhd_model/best_model.pth       ### path to test model
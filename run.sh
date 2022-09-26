gdown https://drive.google.com/uc?id=1_zEVEjABc34QztDH8oIPQv-xUMXc_alq --output datasets/omniglot_data.npy
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dagan.py datasets/omniglot_data.npy final_omniglot_generator.pt --save_checkpoint_path omniglot_checkpoint.pt --iters 50
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_omniglot_classifier.py final_omniglot_generator.pt
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_dagan.py datasets/omniglot_data.npy final_omniglot_generator.pt --save_checkpoint_path omniglot_checkpoint.pt --iters 250
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_omniglot_classifier.py final_omniglot_generator.pt

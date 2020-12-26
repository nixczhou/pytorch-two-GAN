python combine_A_and_B.py --fold_A phase1/A/ --fold_B phase1/B/ --fold_AB basketball_seg_detection/train_phase1/


python combine_A_and_B.py --fold_A phase2/A/ --fold_B phase2/B/ --fold_AB basketball_seg_detection/train_phase2/


python train_two_pix2pix.py --dataroot ./datasets/basketball_seg_detection --name basketball_seg_detection_pix2pix --model two_pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode two_aligned --no_lsgan --norm batch --pool_size 0 --output_nc 1 --phase1 train_phase_1 --phase2 train_phase_2 --save_epoch_freq 2

python test_two_pix2pix.py --dataroot ./datasets/basketball_seg_detection --which_direction AtoB --model two_pix2pix --name basketball_seg_detection_pix2pix --output_nc 1 --dataset_mode aligned --which_model_netG unet_256 --norm batch --how_many 186 --loadSize 256
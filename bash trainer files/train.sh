#!/bin/bash
#SBATCH -A rohit_1202
#SBATCH -c 18
#SBATCH --gres=gpu:2
#SBATCH --nodelist gnode061
#SBATCH --mem-per-cpu=2G
#SBATCH --time=50:00:00
#SBATCH --mail-user=rohitsatish.official@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=output.txt


# activate conda environment
conda init bash
source activate rohit_1202
echo "conda environment activated"


# step 1================================================================================
echo "-----STEP 1: TASK 1: BDD: starting RAP-FT model TRAINING on CITYSCAPES-----------"
echo "--training with DS-RAP units and DS-BN units in encoder and single decoder head for CS. using ERFNet architecture--------"
python /home2/rohit_1202/Rohit_Reprod/MDIL-SS-Reproduction/MDIL-SS-main/train_RAPFT_step1.py --savedir "Checkpoints_2/IDD" --num-epochs 150 --batch-size 6 --state "/home2/rohit_1202/Rohit_Reprod/MDIL-SS-Reproduction/MDIL-SS-main/trained_models/erfnet_encoder_pretrained.pth.tar" --num-classes 27 --current_task=0 --dataset='IDD'
echo "-----done-------"


# step 2 - can take 30-40 hours on 2 Nvidia GeForce GTX 1080 Ti==================
echo "-----STEP 2: TASK 2: BDD->IDD: starting RAPFT-KLD (OURS), KLD between {cs_old, cs_curr}, training model on BDD. lambdac=0.1------------"
python /home2/rohit_1202/Rohit_Reprod/MDIL-SS-Reproduction/MDIL-SS-main/train_new_task_step2.py --savedir “Checkpoints_2/BDD_Step2” --num-epochs 150 --model-name-suffix='ours-IDD1-BDD2' --batch-size 6 --state "/home2/rohit_1202/Rohit_Reprod/MDIL-SS-Reproduction/Checkpoints_2/IDD/checkpoint_IDD_erfnet_RA_parallel_150_6RAP_FT_step1.pth.tar" --dataset='BDD' --dataset_old='IDD' --num-classes 27 20 --current_task=1 --nb_tasks=2 --num-classes-old 27
echo "--done---"
#!/bin/bash
#SBATCH --job-name=vikky_anya_front
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:30:00
conda activate three
cd ~/git/threestudio/
NAME=anya_front
CONFIG=2xl_artic3d
MODEL=xl
ELEV=5.0
WANDB=true
PROJECT=zero123_artic3d_comp
python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=ARTIC3D/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev_dass0 data.default_elevation_deg=${ELEV} system.loggers.wandb.enable=${WANDB} system.loggers.wandb.project=${PROJECT} system.loggers.wandb.name=${NAME}_${CONFIG}config0dass_${MODEL}model_${ELEV}elev # system.freq.guidance_eval=10 trainer.val_check_interval=25

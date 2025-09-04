#################### Supervised only by Multimodal Labels (CMC) ####################
nohup python -u main_tune.py --dataset SIMS --is_pseudo >is_pseudo_sims.txt 2>&1 &
nohup python -u main_tune.py --dataset SIMS-v2 --is_pseudo >is_pseudo_simsv2.txt 2>&1 &
nohup python -u main_tune.py --dataset MOSI --is_pseudo >is_pseudo_mosi.txt 2>&1 &
nohup python -u main_tune.py --dataset MOSEI --is_pseudo >is_pseudo_mosei.txt 2>&1 &
#################### Supervised by Unimodal Labels (CMC-GT) ########################
nohup python -u main_tune.py --dataset SIMS >sims.txt 2>&1 &
nohup python -u main_tune.py --dataset SIMS-v2 >simsv2.txt 2>&1 &
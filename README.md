![Python 3.10](https://img.shields.io/badge/python-3.10-green)
![Pytorch 2.7](https://img.shields.io/badge/pytorch-2.7-orange)

>Codes for **Calibrating Multimodal Consensus for Emotion Recognition**.

## Usage
### Clone the repository
    git clone https://github.com/gw-zhong/CMC.git
### Download the datasets
+ [CH-SIMS](https://github.com/thuiar/MMSA)
+ [CH-SIMS v2](https://github.com/thuiar/MMSA)
### Preparation
Set the ```data_path``` and the ```model_path``` correctly in ```main.py```.
### Single Training
#### Unimodal Pretraining
```python
python main.py --dataset SIMS --transformer_layers 5 --nhead 4 --out_dropout 0.4 --is_pseudo
python main.py --dataset SIMS-v2 --transformer_layers 4 --nhead 2 --out_dropout 0.3 --is_pseudo
python main.py --dataset MOSI --transformer_layers 2 --nhead 4 --out_dropout 0.5 --is_pseudo
python main.py --dataset MOSEI --transformer_layers 2 --nhead 4 --out_dropout 0.0 --is_pseudo
 ```
Or use ground truth unimodal label (CMC-GT):
```python
python main.py --dataset SIMS --transformer_layers 1 --nhead 2 --out_dropout 0.1 --finetune
python main.py --dataset SIMS-v2 --transformer_layers 4 --nhead 8 --out_dropout 0.1 --finetune
 ```
#### Multimodal Finetuning
```python
python main.py --dataset SIMS --transformer_layers 5 --nhead 4 --out_dropout 0.4 --is_pseudo --finetune --pretrained_model
python main.py --dataset SIMS-v2 --transformer_layers 4 --nhead 2 --out_dropout 0.3 --is_pseudo --finetune --pretrained_model
python main.py --dataset MOSI --transformer_layers 2 --nhead 4 --out_dropout 0.5 --is_pseudo --finetune --pretrained_model
python main.py --dataset MOSEI --transformer_layers 2 --nhead 4 --out_dropout 0.0 --is_pseudo --finetune --pretrained_model
 ```
Or use ground truth unimodal label (CMC-GT):
```python
python main.py --dataset SIMS --transformer_layers 1 --nhead 2 --out_dropout 0.1 --finetune --pretrained_model
python main.py --dataset SIMS-v2 --transformer_layers 4 --nhead 8 --out_dropout 0.1 --finetune --pretrained_model
 ```
### Hyperparameter tuning
#### Quick Start
 ```bash
 bash script.sh
  ```
#### Normal Way
 ```python
python main_tune.py --dataset [SIMS/SIMS-v2/MOSI/MOSEI] [--is_pseudo]
 ```
**Note:**

with `--is_pseudo`: training the **CMC** model;

without `--is_pseudo`: training the **CMC-GT** model (currently only supporting `SIMS`/`SIMS-v2`).
### Reproduction
To facilitate the reproduction of the results in the paper, we have also uploaded the corresponding model weights:
- [BaiduYun Disk](https://pan.baidu.com/s/1Gr4VqRzVJAmLxrDAsVmWkw) `code: 2rtn`
## Contact
If you have any question, feel free to contact me through [gwzhong@zju.edu.cn](gwzhong@zju.edu.cn).
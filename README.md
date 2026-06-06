# MARGA: Multi-agent Relative Geometric-aware Attention with Entropy-sequential Decision for Team Orienteering Problem


## 1. Environment Setup

```bash
conda create -n MARGA python=3.10
conda activate MARGA
pip install -r requirements.txt
```

* matplotlib==3.6.3
* numba==0.56.4
* scipy==1.10.0
* tensorboard==2.11.0
* tensorboard_logger==0.1.0
* torch==1.13.1
* tqdm==4.64.1




## 2. Dataset

### Generate Data for Validation and Test

Create validation and test datasets for the desired graph sizes, reward distributions, and tour length limits:

```bash
python generate_data.py --name test --seed 1234 --graph_sizes 20 20 20 35 35 35 50 50 50 75 75 75 100 100 100 --max_length 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5

python generate_data.py --name val --seed 4321 --graph_sizes 20 20 20 35 35 35 50 50 50 75 75 75 100 100 100 --max_length 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5 1.5 2 2.5
```


## 3. Train a Model

Use `run.py` to train a model for TOP 

```bash
python run.py \
  --graph_size 20 --data_distribution const \
  --num_agents 2 --max_length 2.0 \
  --val_dataset data/1depots/const/20/val_seed4321_L2.0.pkl 
```

Checkpoints and logs are saved to `outputs/<problem>_<distribution><size>/<run_name>/`.

## 4. Evaluate with Pretrained Weights


Mixed-training Model :

```bash
python eval.py data/1depots/const/20/test_seed1234_L2.0.pkl \
  --model pretrained/Mixed-training/top_mixed20/MARGA/epoch-99.pt \
  --num_agents 2 --num_depots 1
```

Specific-training Model :

```bash
python eval.py data/1depots/const/20/test_seed1234_L2.0.pkl \
  --model pretrained/Specific-training/const_20/MARGA/epoch-99.pt \
  --num_agents 2 --num_depots 1
```

## 5. Visualise a Solution

Render a solution predicted by a trained or pretrained model using `visualize.py`:

```bash
python visualize.py \
  --model pretrained/Specific-training/const_20/MARGA/epoch-99.pt \
  --graph_size 20 --data_distribution const \
  --num_agents 2 --max_length 2.0 \
  --dataset_file data/1depots/const/20/test_seed1234_L2.0.pkl \
  --instance_idx 0 --output outputs/example_tour.png
```

### Other options and help
```bash
python run.py -h
python eval.py -h
python visualize.py -h
```

## Acknowledgements
* AM ： 
[wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) 

* TOP-Former : 
[danifuertes/top-former](https://github.com/danifuertes/top-former)

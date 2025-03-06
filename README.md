# β-GNN: A Robust Ensemble Approach Against Graph Structure Perturbation

This repository includes the implementation to reproduce the results stated in paper "β-GNN: A Robust Ensemble Approach Against Graph Structure Perturbation". It introduces a very straight-forward approach to tackle edge perturbations towards GNNs, using a weighting factor and an ensemble of GNNs and MLP.
The data used in this implementation is fetched from [GARNET](https://proceedings.mlr.press/v198/deng22a.html) paper and its [repository](https://github.com/cornell-zhang/GARNET).

## Recreate the Python Environment

Follow these steps to create and activate a Conda environment with Python 3.12.3 and install dependencies from `requirements.txt`.

```
conda create -n my_env python=3.12.3 -y
conda activate my_env
conda install pip -y
pip install -r requirements.txt
```

## Reproduce ALL Results for GNNs (with or without β-GNN) and Compared Models

To reproduce the results given in the paper, you can run the ```bash run_all_experiments.sh``` script. Since this will iteratively run all experiments with the tuned hyperparameters, It will take a while.

If you would like to train the models one by one and do experiment with different parameters, you can run ```python train.py --args [ARGS]``` with the arguments you can find in the script.

Similarly, you can reproduce the results for **ALL** experiments for the compared models by running ```bash run_benchmarks.sh```. Or if you would like to save time and run one by one, you can run 
```python benchmark.py``` with proper arguments.

To visualize the track of the β values, one should first run and save the beta values for corresponding experiments. Then, by modifying the paths in ```beta_visualize.py```, you can visualize the results for learned β values.

## Citation
If you refer to β-GNN in your research, please cite our [paper](https://doi.org/10.1145/3721146.3721949).

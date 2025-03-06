import torch

from models import *

from utils import preprocess_args, results_dataframe, model_select, logger, beta_logger
from data_loader import load_planetoid, adj_to_edges, load_dataset, mask_to_index, convert_to_adjacency_matrix
import pandas as pd

# from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

import argparse


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item()


def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main(args):

    log_results = list()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.save_df:
        # Initialize an empty DataFrame to store the best results
        columns = ['node_idx', 'prediction', 'ground_truth', 'set_type']
        best_results_df = pd.DataFrame(columns=columns)

    dataset = load_dataset(args)

    # dataset = load_planetoid('Cora')[0] # clean version for Planetoid
    # print(dataset[0])

    # data = dataset[0].to(device) # clean version for Planetoid

    num_features = dataset.x.shape[1]
    num_classes = len(torch.unique(dataset.y))
    print(f'num features {num_features}, num classes {num_classes}')

    data = dataset.to(device)

    model = model_select(num_features, num_classes, device, args)

    for run in range(args.runs):
        model.reset_parameters()
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.model == 'CombinedModel' or args.model == 'CombinedModelVector':
            optimizer = torch.optim.Adam([
                {'params': model.gcn_model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},  # Learning rate for GCN
                {'params': model.node_feature_model.parameters(), 'lr': args.mlp_lr, 'weight_decay': args.mlp_weight_decay},  # Learning rate for MLP
                {'params': [model.beta], 'lr': 0.01, 'weight_decay': args.weight_decay}  # Learning rate for beta
            ])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        criterion = torch.nn.CrossEntropyLoss()

        # Initialize variables to store the best metrics
        # best_test_acc = 0
        best_val_acc = 0
        best_metrics = {
            'epoch': 0,
            'loss': 0,
            'train_acc': 0,
            'val_acc': 0,
            'test_acc': 0,
            'beta': None
        }
        betas = []
        for epoch in range(args.epochs):
            loss = train(model, data, optimizer, criterion)
            train_acc, val_acc, test_acc = test(model, data)

            # Check if the current validation accuracy is the best so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc  # Record test accuracy of best validation model
                best_metrics['epoch'] = epoch
                best_metrics['loss'] = loss
                best_metrics['train_acc'] = train_acc
                best_metrics['val_acc'] = val_acc
                best_metrics['test_acc'] = test_acc
                if args.model == 'CombinedModel':
                    best_metrics['beta'] = model.beta.data.clone()
                elif args.model == 'InterLayer':
                    best_metrics['beta1'] = model.beta1.data.clone()
                    best_metrics['beta2'] = model.beta2.data.clone()

                if args.save_df:
                    results_df = results_dataframe(model, data)

            if args.model in ['CombinedModel', 'CombinedModelVector']:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}, Beta: {model.beta.data}')
            elif args.model in ['InterLayer']:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}, Beta 1: {model.beta1.data}, Beta 2: {model.beta2.data}')
            else:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            betas.append(model.beta.data.clone())

        # Print the best metrics after training
        if args.model in ['CombinedModel', 'CombinedModelVector']:
            print(f'Best Test Acc: {best_metrics["test_acc"]:.5f} at Epoch: {best_metrics["epoch"]:03d}, '
                  f'Loss: {best_metrics["loss"]:.5f}, Train: {best_metrics["train_acc"]:.5f}, '
                  f'Val: {best_metrics["val_acc"]:.5f}, Beta: {best_metrics["beta"]}')
        elif args.model in ['InterLayer']:
            print(f'Best Test Acc: {best_metrics["test_acc"]:.5f} at Epoch: {best_metrics["epoch"]:03d}, '
                  f'Loss: {best_metrics["loss"]:.5f}, Train: {best_metrics["train_acc"]:.5f}, '
                  f'Val: {best_metrics["val_acc"]:.5f}, Beta 1: {best_metrics["beta1"]}, Beta 2: {best_metrics["beta2"]}')
        else:
            print(f'Best Test Acc: {best_metrics["test_acc"]:.5f} at Epoch: {best_metrics["epoch"]:03d}, '
                  f'Loss: {best_metrics["loss"]:.5f}, Train: {best_metrics["train_acc"]:.5f}, '
                  f'Val: {best_metrics["val_acc"]:.5f}')
        beta_logger(args, betas)
        log_results.append(best_metrics)

        if args.save_df:
            best_results_df = pd.concat([best_results_df, results_df], ignore_index=True)
            best_results_df.to_csv(f'results/{args.model}_{args.dataset}_{args.attack}_{args.ptb_rate}_results.csv',
                                   index=False)
            print(best_results_df.head())
    logger(args, log_results)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--mlp_lr', type=float, default=0.01, help='learning rate for the node feature model (MLP)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout rate, default is 0.5')
    parser.add_argument('--mlp_dropout_rate', type=float, default=0.5, help='dropout rate for node feature model (MLP), default is 0.5')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--mlp_weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--hidden_dim', type=int, default=16, help='hidden dimension size')
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'NodeFeatureModel', 'CombinedModel', 'CombinedModelVector',
                                 'InterLayer', 'GPRGNN'], help='model name')
    parser.add_argument('--baseline', type=str, default='GCN', help='Baseline GNN model name for the combined model',
                        choices=['GCN', 'GPRGNN', 'GAT'])
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads for GAT model')
    # data loader arguments
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel'],
                        help='choose graph dataset')
    parser.add_argument('--perturbed', action='store_true',
                        help='use adversarial graph as input')
    parser.add_argument('--attack', type=str, default='meta',
                        choices=['nettack', 'meta', 'grbcd'],
                        help='used to choose attack method and test nodes')
    parser.add_argument('--ptb_rate', type=float, default=.2,
                        help='adversarial perturbation budget:\
                                suggest to use 0.2 for meta attack, 5.0 for nettack attack, 0.5 for grbcd attack')
    parser.add_argument('--runs', type=int, default=1, help='number of runs to measure mean and deviation')
    parser.add_argument('--save_df', action='store_true', help='Whether to save results in dataframe format') # save
    # outputs optionally

    # Parse GPRGNN arguments

    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])

    args = parser.parse_args()

    # args = preprocess_args(args)

    main(args)
    print(args)

import argparse
from utils import benchmark_logger

from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.defense import GCN, GCNSVD, GCNJaccard, RGCN, ProGNN
from models import ModifiedGCNJaccard
from deeprobust.graph.utils import preprocess

from data_loader import load_dataset_adj, edge2adj

from scipy.sparse import csr_matrix, issparse


import torch
import numpy as np


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available")
    # device = torch.device("cpu")

    adj_mtx, features, labels, idx_train, idx_val, idx_test = load_dataset_adj(args)
    # adj_mtx, features, labels, idx_train, idx_val, idx_test = adj_mtx.to(device), features.to(device), labels.to(device), idx_train.to(device), idx_val.to(device), idx_test.to(device)

    num_features = features.shape[1]

    # adj, features, labels = preprocess(data.adj_mtx, data.features, data.labels, preprocess_adj=False, sparse=True, device=device)
    model = GCN(nfeat=num_features, nhid=64, nclass=labels.max().item() + 1, dropout=0.5, lr=args.lr,
                weight_decay=args.weight_decay, device=device)

    test_results = list()
    if args.only_gcn:
        model = model.to(device)
        perturbed_adj, features, labels = preprocess(adj_mtx, features, labels, preprocess_adj=False, sparse=True, device=device)
        for run in range(args.runs):
            model.fit(features, perturbed_adj, labels, idx_train, idx_val)  # train on clean graph with earlystopping
            test_acc = model.test(idx_test)
            test_results.append(test_acc)

    elif args.method == 'GCNJaccard':
        print(f"{args.method} method is being trained")
        if args.dataset in ['cora', 'pubmed']:
            model = GCNJaccard(nfeat=features.shape[1],
                               nhid=32,
                               nclass=labels.max().item() + 1,
                               dropout=0.5,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               device=device).to(device)
            for run in range(args.runs):
                model.fit(features, adj_mtx, labels, idx_train, idx_val, train_iters=args.epochs, threshold=0.03)
                test_acc = model.test(idx_test)
                test_results.append(test_acc)
        elif args.dataset in ['squirrel', 'chameleon']:
            model = ModifiedGCNJaccard(nfeat=features.shape[1],
                               nhid=32,
                               nclass=labels.max().item() + 1,
                               dropout=0.5,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               device=device).to(device)
            for run in range(args.runs):
                model.fit(features, adj_mtx, labels, idx_train, idx_val, train_iters=args.epochs, threshold=0.03)
                test_acc = model.test(idx_test)
                test_results.append(test_acc)

    elif args.method == 'GCNSVD':
        print(f"{args.method} method is being trained")
        model = GCNSVD(nfeat=features.shape[1],
                       nhid=32,
                       nclass=labels.max().item() + 1,
                       dropout=0.5,
                       lr=args.lr,
                       weight_decay=args.weight_decay,
                       device=device).to(device)

        for run in range(args.runs):
            model.fit(features, adj_mtx, labels, idx_train, idx_val, train_iters=args.epochs, k=50)
            test_acc = model.test(idx_test)
            test_results.append(test_acc)

    elif args.method == 'RGCN':
        print(f"{args.method} method is being trained")
        model = RGCN(nnodes=adj_mtx.shape[0],
                     nfeat=features.shape[1],
                     nclass=labels.max() + 1,
                     dropout=0.6,
                     lr=args.lr,
                     nhid=32,
                     device=device).to(device)

        for run in range(args.runs):
            model.fit(csr_matrix(features), adj_mtx, labels, idx_train, idx_val, train_iters=args.epochs)
            test_acc = model.test(idx_test)
            test_results.append(test_acc)

    elif args.method == 'prognn':
        perturbed_adj, features, labels = preprocess(adj_mtx, features, labels, preprocess_adj=False, device=device)
        for run in range(args.runs):
            prognn = ProGNN(model, args, device=device)
            prognn.fit(features, perturbed_adj.to(device), labels, idx_train, idx_val)
            test_acc = prognn.test(features, labels, idx_test)
            test_results.append(test_acc)

    benchmark_logger(args, test_results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--only_gcn', action='store_true',
                        help='test the performance of gcn without other components')
    parser.add_argument('--method', type=str, default='prognn', choices=['prognn', 'GCNJaccard', 'GCNSVD', 'RGCN'], help='graph defence method')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
    parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
    parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
    parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
    parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
    parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')


    parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--symmetric', action='store_true',
                        help='how adj would be estimated')

    parser.add_argument('--runs', type=int, default=1, help='number of runs to measure mean and deviation')

    args = parser.parse_args()

    print(f"Arguments are: \n {args}")
    main(args)

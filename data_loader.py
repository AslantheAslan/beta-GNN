import argparse
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.loader import NeighborLoader

from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data

import numpy as np
import pickle
from scipy.sparse import load_npz, csr_matrix
from deeprobust.graph.data import Dataset, PrePtbDataset

import argparse

import pandas as pd

import pyarrow
import fastparquet

import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import os
import sys

pd.set_option('display.max_columns', 100)


def adj_to_edges(adj):
    # Convert the numpy array to a scipy sparse matrix
    adj_matrix_sparse = csr_matrix(adj)
    # Convert the adjacency matrix to edge_index format
    edge_index, _ = from_scipy_sparse_matrix(adj_matrix_sparse)
    return edge_index


def convert_to_adjacency_matrix(edge_list, num_vertices):
    # Initialize the adjacency matrix with all zeros
    adj_mat = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]

    # Iterate through all the edges
    for edge in edge_list:
        # Get the source and destination nodes
        source = edge[0]
        destination = edge[1]

        # Mark the edge between source and destination in adjacency matrix
        adj_mat[source][destination] = 1

        #  Mark the edge between destination and source in adjacency matrix
        # (since the graph is bidirectional)
        adj_mat[destination][source] = 1

    return torch.tensor(adj_mat, dtype=torch.long)


def index_to_mask(index, size):
    """"
    To convert idx_train to train_mask
    """
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def mask_to_index(mask):
    """
    Convert a mask (boolean tensor) back to indices.
    """
    return torch.nonzero(mask, as_tuple=False).squeeze()


def load_planetoid(dataset_name):
    # For this example, we'll use the Cora dataset from Planetoid
    dataset = Planetoid(root='~/data/' + dataset_name, name=dataset_name)
    return dataset


def load_dataset(args):
    ## load dataset

    """

    Fetched from https://github.com/cornell-zhang/GARNET/blob/main/main.py

    """

    dataset = args.dataset
    if dataset in ['chameleon', 'squirrel']:
        with open(f'data/GARNET_data/{dataset}_data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        features = data["features"]
        labels = data["labels"]
        idx_train = data["idx_train"]
        idx_val = data["idx_val"]
        idx_test = data["idx_test"]
        if args.perturbed:
            adj_mtx = load_npz(f'data/GARNET_data/{dataset}_perturbed_{args.ptb_rate}.npz')
        else:
            adj_mtx = load_npz(f'data/GARNET_data/{dataset}.npz')
        if args.attack == 'nettack':
            idx_test = np.load(f"data/GARNET_data/{dataset}_idx_test.npy")

    else:
        data = Dataset(root='./data/GARNET_data/', name=dataset, setting='prognn')
        labels = data.labels
        features = data.features.todense()
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        perturbed_data = PrePtbDataset(root='./data/GARNET_data/',
                                       name=dataset,
                                       attack_method=args.attack,
                                       ptb_rate=args.ptb_rate)
        if args.perturbed:
            adj_mtx = perturbed_data.adj
        else:
            adj_mtx = data.adj
        if args.attack == 'nettack':
            idx_test = perturbed_data.target_nodes

    if args.dataset == "chameleon" and args.attack == "nettack":
        embedding_symmetric = True
    else:
        embedding_symmetric = False
    edge_index = adj_to_edges(adj_mtx)
    train_mask = index_to_mask(idx_train, adj_mtx.shape[0])
    val_mask = index_to_mask(idx_val, adj_mtx.shape[0])
    test_mask = index_to_mask(idx_test, adj_mtx.shape[0])
    print(f"IDX TEST LEN: {len(idx_test)}")
    print(f"TEST MASK: {len(test_mask)}")
    data = Data(
        x=torch.from_numpy(features),
        edge_index=edge_index,
        y=torch.from_numpy(labels).long(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    print(
        f'For dataset {args.dataset}, the perturbation is {args.perturbed}. Under {args.attack}, with {args.ptb_rate} perturbation rate, the edge size is {len(edge_index[0])}')
    return data


def load_dataset_adj(args):
    ## load dataset_adj

    """

    Fetched from https://github.com/cornell-zhang/GARNET/blob/main/main.py

    """

    dataset = args.dataset
    if dataset in ['chameleon', 'squirrel']:
        with open(f'data/GARNET_data/{dataset}_data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        features = data["features"]
        labels = data["labels"]
        idx_train = data["idx_train"]
        idx_val = data["idx_val"]
        idx_test = data["idx_test"]
        if args.perturbed:
            adj_mtx = load_npz(f'data/GARNET_data/{dataset}_perturbed_{args.ptb_rate}.npz')
        else:
            adj_mtx = load_npz(f'data/GARNET_data/{dataset}.npz')
        if args.attack == 'nettack':
            idx_test = np.load(f"data/GARNET_data/{dataset}_idx_test.npy")

    else:
        data = Dataset(root='./data/GARNET_data/', name=dataset, setting='prognn')
        labels = data.labels
        features = data.features.todense()
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        perturbed_data = PrePtbDataset(root='./data/GARNET_data/',
                                       name=dataset,
                                       attack_method=args.attack,
                                       ptb_rate=args.ptb_rate)
        if args.perturbed:
            adj_mtx = perturbed_data.adj
        else:
            adj_mtx = data.adj
        if args.attack == 'nettack':
            idx_test = perturbed_data.target_nodes

    if args.dataset == "chameleon" and args.attack == "nettack":
        embedding_symmetric = True
    else:
        embedding_symmetric = False

    # data = Data(
    #     features=features,
    #     adj_mtx=adj_mtx,
    #     labels=labels,
    #     idx_train=idx_train,
    #     idx_val=idx_val,
    #     idx_test=idx_test
    # )

    return adj_mtx, features, labels, idx_train, idx_val, idx_test


def load_batches(args):
    data = load_dataset(args)
    n_smpls_1hop = args.n_smpls_1hop
    n_smpls_2hop = args.n_smpls_2hop
    print(n_smpls_1hop, n_smpls_2hop)
    # Create loaders for train, val, and test sets
    train_loader = NeighborLoader(
        data,
        num_neighbors=[n_smpls_1hop, n_smpls_2hop],  # 2-hop neighborhood
        batch_size=256,  # number of seed nodes to start the sampling from
        input_nodes=data.train_mask,
        shuffle=True
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[n_smpls_1hop, n_smpls_2hop],
        batch_size=256,
        input_nodes=data.val_mask,
        shuffle=False
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=[n_smpls_1hop, n_smpls_2hop],
        batch_size=1,
        input_nodes=data.test_mask,
        shuffle=False
    )

    return data, train_loader, val_loader, test_loader


def edge2adj(edge_list) -> torch.tensor:
    # Separate senders and receivers
    senders = edge_list[0]
    receivers = edge_list[1]

    # Determine the number of nodes (assuming nodes are zero-indexed)
    num_nodes = max(senders.max(), receivers.max()) + 1

    # Create an empty adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)

    # Populate the adjacency matrix
    adj_matrix[senders, receivers] = 1

    return adj_matrix


def download_CIDDS() -> None:
    url = 'https://www.hs-coburg.de/wp-content/uploads/2024/11/CIDDS-001.zip'
    # Define the target folder and ZIP file name
    target_folder = r'./data/CIDDS'  # Windows

    # Check if the dataset is already downloaded
    if os.path.exists(target_folder) and any(os.scandir(target_folder)):
        print(f"Dataset already exists in {target_folder}. Skipping download.")
    else:
        os.makedirs(target_folder, exist_ok=True)  # Ensure folder exists

        # Download the ZIP file with progress bar
        print("Downloading dataset...")
        with urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB per read
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            zip_data = BytesIO()
            while chunk := response.read(block_size):
                zip_data.write(chunk)
                progress_bar.update(len(chunk))

            progress_bar.close()

        # Extract ZIP file
        print("\nExtracting dataset...")
        with ZipFile(zip_data) as zfile:
            zfile.extractall(target_folder)

        print(f"Files extracted to: {target_folder}")

    print("Process completed!")


def preprocess_CIDDS() -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sketches insightful figures for CIDDS. Saves train, val, and test splits as parquet files. Adopted from
    https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python/blob/main/Chapter16/chapter16
    .ipynb
    :return: df_train, df_val, df_test
    """

    df = pd.read_csv("./data/CIDDS/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv")
    print("RAW DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    # Drop unnecessary colums
    df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos', 'class', 'attackID', 'attackDescription'])
    df['attackType'] = df['attackType'].replace('---', 'benign')
    df['Date first seen'] = pd.to_datetime(df['Date first seen'])
    print("NEW DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    # Count labels
    count_labels = df['attackType'].value_counts() / len(df) * 100
    print(count_labels)
    plt.pie(count_labels[:3], labels=df['attackType'].unique()[:3], autopct='%.0f%%')
    plt.savefig("count_label", dpi=300, bbox_inches='tight')

    # Feature Distribution
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
    df['Duration'].hist(ax=ax1)
    ax1.set_xlabel("Duration")
    df['Packets'].hist(ax=ax2)
    ax2.set_xlabel("Number of packets")
    pd.to_numeric(df['Bytes'], errors='coerce').hist(ax=ax3)
    ax3.set_xlabel("Number of bytes")
    plt.savefig("feature_distribution", dpi=300, bbox_inches='tight')

    # Preprocessing the CIDDS-001 dataset

    df['weekday'] = df['Date first seen'].dt.weekday
    df = pd.get_dummies(df, columns=['weekday']).rename(columns={'weekday_0': 'Monday',
                                                                 'weekday_1': 'Tuesday',
                                                                 'weekday_2': 'Wednesday',
                                                                 'weekday_3': 'Thursday',
                                                                 'weekday_4': 'Friday',
                                                                 'weekday_5': 'Saturday',
                                                                 'weekday_6': 'Sunday',
                                                                 })

    df['daytime'] = (df['Date first seen'].dt.second + df['Date first seen'].dt.minute * 60 + df[
        'Date first seen'].dt.hour * 60 * 60) / (24 * 60 * 60)

    def one_hot_flags(input):
        return [1 if char1 == char2 else 0 for char1, char2 in zip('APRSF', input[1:])]

    df = df.reset_index(drop=True)
    ohe_flags = one_hot_flags(df['Flags'].to_numpy())
    ohe_flags = df['Flags'].apply(one_hot_flags).to_list()
    df[['ACK', 'PSH', 'RST', 'SYN', 'FIN']] = pd.DataFrame(ohe_flags, columns=['ACK', 'PSH', 'RST', 'SYN', 'FIN'])
    df = df.drop(columns=['Date first seen', 'Flags'])
    print("PREPROCESSED DATAFRAME SHAPE: ", df.shape)
    print(df.head())

    temp = pd.DataFrame()
    temp['SrcIP'] = df['Src IP Addr'].astype(str)
    temp['SrcIP'][~temp['SrcIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp['SrcIP'].str.split('.', expand=True).rename(columns={2: 'ipsrc3', 3: 'ipsrc4'}).astype(int)[
        ['ipsrc3', 'ipsrc4']]
    temp['ipsrc'] = temp['ipsrc3'].apply(lambda x: format(x, "b").zfill(8)) + temp['ipsrc4'].apply(
        lambda x: format(x, "b").zfill(8))
    df = df.join(temp['ipsrc'].str.split('', expand=True)
                 .drop(columns=[0, 17])
                 .rename(columns=dict(enumerate([f'ipsrc_{i}' for i in range(17)])))
                 .astype('int32'))
    print(df.head(5))

    temp = pd.DataFrame()
    temp['DstIP'] = df['Dst IP Addr'].astype(str)
    temp['DstIP'][~temp['DstIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp['DstIP'].str.split('.', expand=True).rename(columns={2: 'ipdst3', 3: 'ipdst4'}).astype(int)[
        ['ipdst3', 'ipdst4']]
    temp['ipdst'] = temp['ipdst3'].apply(lambda x: format(x, "b").zfill(8)) \
                    + temp['ipdst4'].apply(lambda x: format(x, "b").zfill(8))
    df = df.join(temp['ipdst'].str.split('', expand=True)
                 .drop(columns=[0, 17])
                 .rename(columns=dict(enumerate([f'ipdst_{i}' for i in range(17)])))
                 .astype('int32'))
    print(df.head(5))

    m_index = df[pd.to_numeric(df['Bytes'], errors='coerce').isnull() == True].index
    df['Bytes'].loc[m_index] = df['Bytes'].loc[m_index].apply(lambda x: 10e6 * float(x.strip().split()[0]))
    df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce', downcast='integer')

    df = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Proto', 'attackType'])
    print(df.head(5))

    labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']
    df_train, df_test = train_test_split(df, random_state=0, test_size=0.2, stratify=df[labels])
    df_val, df_test = train_test_split(df_test, random_state=0, test_size=0.5, stratify=df_test[labels])

    scaler = PowerTransformer()
    df_train[['Duration', 'Packets', 'Bytes']] = scaler.fit_transform(df_train[['Duration', 'Packets', 'Bytes']])
    df_val[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_val[['Duration', 'Packets', 'Bytes']])
    df_test[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_test[['Duration', 'Packets', 'Bytes']])

    df_train.to_parquet('./data/CIDDS/df_train.parquet')
    df_val.to_parquet('./data/CIDDS/df_val.parquet')
    df_test.to_parquet('./data/CIDDS/df_test.parquet')

    print(df_train[df_train['benign'] == 1].head())

    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
    df_train['Duration'].hist(ax=ax1)
    ax1.set_xlabel("Duration")
    df_train['Packets'].hist(ax=ax2)
    ax2.set_xlabel("Number of packets")
    df_train['Bytes'].hist(ax=ax3)
    ax3.set_xlabel("Number of bytes")
    plt.savefig("scaled_feature_distribution", dpi=300, bbox_inches='tight')

    return df_train, df_val, df_test


if __name__ == '__main__':
    # dataset = load_planetoid('Cora')
    # print(dataset[0])
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
    args = parser.parse_args()
    print(args)
    dataset = load_dataset(args)
    # print(type(dataset.edge_index[0]), type(dataset.x), type(dataset.y), type(dataset.train_mask), type(dataset.val_mask), type(dataset.test_mask))
    #
    # planetoid = load_planetoid('Cora')
    # print(type(planetoid.data.edge_index[0]), type(planetoid.data.x), type(planetoid.data.y), type(planetoid.data.train_mask), type(planetoid.data.val_mask), type(planetoid.data.test_mask))
    # print(planetoid)
    # print(planetoid[0])
    #
    # num_trues_dataset = torch.sum(dataset.train_mask).item()
    # num_trues_planetoid = torch.sum(planetoid.data.train_mask).item()
    #
    # print(len(dataset.train_mask), len(planetoid.data.train_mask))
    # print(num_trues_dataset, num_trues_planetoid)
    #
    #
    # print(dataset.train_mask)
    # print(planetoid.data.train_mask)

    npz_dataset = np.load(f'data/GARNET_data/{args.dataset}.npz')
    print(npz_dataset.files)
    print(
        f'Attr_shape\n {npz_dataset['attr_shape']}\n\n, Attr_indptr\n {npz_dataset['attr_indptr']}\n\n, Adj_data\n {npz_dataset['adj_data']}\n\n, Adj_shape\n {npz_dataset['adj_shape']}\n\n, Attr_data\n {npz_dataset['attr_data']}\n\n')

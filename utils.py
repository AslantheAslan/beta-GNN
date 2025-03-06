import yaml
import pandas as pd
from torch import where

import csv
import statistics

from models import *


def preprocess_args(args):
    """

    Utility functions for reading and writing YAML files.

    Copied from https://github.com/cornell-zhang/GARNET/blob/main/utils.py#L106

    """

    arg_data = args.__dict__
    if args.perturbed:
        config_file = \
            f"configs/{args.dataset}/{args.backbone}/{args.attack}/perturbed_{args.ptb_rate}.yaml"
    else:
        config_file = \
            f"configs/{args.dataset}/{args.backbone}/{args.attack}/clean.yaml"
    with open(config_file) as file:
        yaml_data = yaml.safe_load(file)
    for arg, value in arg_data.items():
        if value is None:
            continue
        if value in ['True', 'False']:
            yaml_data[arg] = value == 'True'
        else:
            yaml_data[arg] = value
    args.__dict__ = yaml_data
    if args.full_distortion:
        if args.dataset == "chameleon" and args.attack == "meta":
            args.gamma = 0.3
        elif args.dataset == "squirrel":
            args.gamma = 0.08
    return args


def logger(args, best_dict):
    if args.perturbed:
        filename = './logs_new/' + args.dataset + '_' + args.model + '_' + args.baseline + '_' + args.attack + '_' + str(
            args.ptb_rate) + '.log'
    else:
        filename = './logs_new/' + args.dataset + '_' + args.model + '_' + args.baseline + '_' + 'clean' + '_' + args.attack + '_' + '.log'

    if len(best_dict) > 1:
        # Extract 'test_acc' values for statistical calculation
        test_acc_values = [d["test_acc"] for d in best_dict]

        # Calculate the mean and standard deviation
        mean_test_acc = statistics.mean(test_acc_values)
        std_dev_test_acc = statistics.stdev(test_acc_values)
    else:
        mean_test_acc = best_dict[0]["test_acc"]
        std_dev_test_acc = 0

    # Write to the CSV file
    with open(filename, 'w', newline='') as csvfile:
        # Determine the fieldnames from the keys of the first dictionary
        fieldnames = best_dict[0].keys()

        # Create a CSV writer
        writer = csv.writer(csvfile)

        # Write the args as the first row
        writer.writerow([args])  # Note: args is a single value

        # Write the header row based on the dictionary keys
        writer.writerow(fieldnames)

        # Write the actual data rows from the dictionaries
        for d in best_dict:
            writer.writerow(d.values())
        # Write the mean and standard deviation in the last row
        # This is a new row where the 'test_acc' statistics are written under the appropriate column
        summary_row = [''] * (len(fieldnames) - 1)  # Empty cells for other columns
        summary_row.append(f"Mean: {mean_test_acc:.4f}, Std Dev: {std_dev_test_acc:.4f}")

        writer.writerow(summary_row)

    print(f"Data has been written to {filename}")


def beta_logger(args, betas):
    if args.perturbed:
        filename = './betas/' + args.dataset + '_' + args.model + '_' + args.baseline + '_' + args.attack + '_' + str(
            args.ptb_rate) + '.log'
    else:
        filename = './betas/' + args.dataset + '_' + args.model + '_' + args.baseline + '_' + 'clean' + '_' + args.attack + '_' + '.log'

    with open(filename, "w") as file:
        for item in betas:
            file.write(f"{item}\n")


def benchmark_logger(args, test_acc_list):
    if args.perturbed:
        filename = './logs_benchmark/' + args.dataset + '_' + args.method + '_' + args.attack + '_' + str(
            args.ptb_rate) + '.log'
    else:
        filename = './logs_benchmark/' + args.dataset + '_' + args.method + '_' + 'clean' + '_' + args.attack + '_' + '.log'

    mean_test_acc = statistics.mean(test_acc_list)
    std_dev_test_acc = statistics.stdev(test_acc_list)

    test_acc_list.append(f"Avg Test Acc:{mean_test_acc:.4f}, Std Dev: {std_dev_test_acc:.6f}")

    with open(filename, "w") as file:
        for item in test_acc_list:
            file.write(f"{item}\n")


def results_dataframe(model, data) -> pd.DataFrame:
    # Capture predictions, ground truth, and set type
    logits = model(data)
    for mask_type in ['train_mask', 'val_mask', 'test_mask']:
        mask = getattr(data, mask_type)
        predictions = logits[mask].max(1)[1]
        ground_truth = data.y[mask]
        node_indices = where(mask)[0].cpu().numpy()

        set_type = mask_type.replace('_mask', '')  # convert mask type to set type (train/val/test)

        # Create a DataFrame with the current best results
        results = pd.DataFrame({
            'node_idx': node_indices,
            'prediction': predictions.cpu().numpy(),
            'ground_truth': ground_truth.cpu().numpy(),
            'set_type': set_type
        })
    return results


def model_select(num_features, num_classes, device, args):
    if args.model == 'GCN':
        model = GCN(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                    dropout_rate=args.dropout_rate).to(device)
    elif args.model == 'GAT':
        model = GAT(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                    heads=args.heads, dropout_rate=args.dropout_rate).to(device)
    elif args.model == 'GPRGNN':
        model = GPRGNN(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                       dropout_rate=args.dropout_rate, args=args).to(device)
    elif args.model == 'NodeFeatureModel':
        model = NodeFeatureModel(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                                 dropout_rate=args.dropout_rate).to(device)

    elif args.model == 'CombinedModel':
        if args.baseline == 'GCN':
            gnn_model = GCN(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                            dropout_rate=args.dropout_rate).to(device)
        elif args.baseline == 'GPRGNN':
            gnn_model = GPRGNN(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                               dropout_rate=args.dropout_rate, args=args).to(device)
        elif args.baseline == 'GAT':
            gnn_model = GAT(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                            heads=args.heads, dropout_rate=args.dropout_rate).to(device)
        else:
            return None
        feat_model = NodeFeatureModel(in_channels=num_features, hidden_channels=args.hidden_dim,
                                      out_channels=num_classes, dropout_rate=args.mlp_dropout_rate).to(device)
        model = CombinedModel(gcn_model=gnn_model, node_feature_model=feat_model, out_channels=num_classes).to(device)

    elif args.model == 'CombinedModelVector':
        if args.baseline == 'GCN':
            gnn_model = GCN(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                            dropout_rate=args.dropout_rate).to(device)
        elif args.baseline == 'GPRGNN':
            gnn_model = GPRGNN(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                               dropout_rate=args.dropout_rate, args=args).to(device)
        else:
            return None
        feat_model = NodeFeatureModel(in_channels=num_features, hidden_channels=args.hidden_dim,
                                      out_channels=num_classes, dropout_rate=args.mlp_dropout_rate).to(device)
        model = CombinedModelVector(gcn_model=gnn_model, node_feature_model=feat_model, out_channels=num_classes).to(
            device)
    elif args.model == 'InterLayer':
        model = InterLayer(in_channels=num_features, hidden_channels=args.hidden_dim, out_channels=num_classes,
                           dropout_rate=args.dropout_rate).to(device)
    else:
        print('invalid model')
        return None

    return model

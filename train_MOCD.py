import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import copy
import utils
from multi_data import Multi_view_data, Multi_view_OMix_data
from loadMatData import load_data, generate_partition
from evaluation_metrics import EvaluationMetrics
from utils import reassign_labels, special_train_test_split
from evaluation_metrics import compute_all_scores
from model import MOCD



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

np.set_printoptions(precision=4, suppress=True)


def build_graphs(X, neighbor):
    A = []
    for v in range(len(X)):
        X_v = torch.tensor(X[v]).T
        A_v, _ = utils.build_CAN(X_v, neighbor)
        A.append(A_v)
    return A


def train_and_evaluate(args):
    def valid(model, loader, device):
        # Set the model to evaluation mode
        model.eval()
        prob = []
        resp = []
        label = []
        for indexes, (X, Y) in enumerate(loader):
            A_test = build_graphs(X, neighbor=args.neighbor)
            for v in range(num_views):
                X[v] = X[v].to(device)
                A_test[v] = A_test[v].to(device)
            Y = Y.to(device)
            with torch.no_grad():
                evidences, res = model(X, A_test)
                _, Y_pre = torch.max(res, dim=1)
                prob.append(torch.softmax(res, 1).cpu().numpy())
                resp.append(res.cpu().numpy())
                label.append(Y.cpu().numpy())
        label = np.concatenate(label)
        prob = np.concatenate(prob)
        resp = np.concatenate(resp)
        return resp, prob, label

    def hsic_loss(X, Y, sigma=1.0):
        """
        Computes the Hilbert-Schmidt Independence Criterion (HSIC)
        :param X: Feature matrix with shape (n, d)
        :param Y: Feature matrix with shape (n, d)
        :param sigma: Bandwidth for Gaussian kernel
        :return: HSIC value
        """
        n = X.size(0)  # Number of samples
        # Centering matrix H
        H = (torch.eye(n) - (1.0 / n) * torch.ones((n, n))).to(device)

        # Compute Gaussian kernel matrices K_X and K_Y
        K_X = torch.exp(-torch.cdist(X, X, p=2) ** 2 / (2 * sigma ** 2))
        K_Y = torch.exp(-torch.cdist(Y, Y, p=2) ** 2 / (2 * sigma ** 2))

        # Calculate HSIC
        HSIC_value = torch.trace(K_X @ H @ K_Y @ H) / ((n - 1) ** 2)
        return HSIC_value

    model = MOCD(num_views, dims, num_classes, gamma=args.gamma)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.CrossEntropyLoss(reduction='none')
    model.to(device)
    model.train()
    best_valid_ccr = 0.0
    loss_history = []
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for (X, Y, x_mix, Y_mix, lambdas, u) in train_loader:

            A_train = build_graphs(X, neighbor=args.neighbor)
            for v in range(num_views):
                x_mix[v] = x_mix[v].to(device)
                Y_mix[v] = Y_mix[v].to(device)
                lambdas[v] = lambdas[v].to(device)
                u[v] = u[v].to(device)
                X[v] = X[v].to(device)
                A_train[v] = A_train[v].to(device)
            Y = Y.to(device)

            evidences, res = model(X, A_train)
            sup_loss = sum(criterion(evidences[v], Y) for v in range(num_views))

            agu_evi = model.agument(x_mix)
            loss_Oen = 0
            for view in range(num_views):
                mix_mask = (1 - lambdas[view]) * (1 - u[view]) + u[view] / 4
                loss_Oen += torch.mean(criterion2(agu_evi[view], Y_mix[view]) * mix_mask)
                loss_Oen += torch.mean(criterion2(agu_evi[view], Y) * mix_mask)

                random_targets = [torch.LongTensor(Y.shape[0]).random_(c, c + 1).to(device) for c in range(num_classes)]
                loss_Oen += sum(torch.mean(criterion2(agu_evi[view], target) * u[view]) for target in
                                random_targets) / num_classes / 2


            loss_hsic = 0
            for view in range(0, num_views):
                loss_hsic += hsic_loss(agu_evi[view], res)

            loss = sup_loss + loss_Oen * args.alpha + loss_hsic * args.beta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss)


        resp, softmax_scores, valid_truth_label = valid(model, valid_loader, device)

        softmax_ccr, softmax_fpr, softmax_ccrs = EvaluationMetrics.ccr_at_fpr(np.array(valid_truth_label),
                                                                              softmax_scores,
                                                                              unk_label=args.unseen_label_index, fpr_values=fpr_values)


        if best_valid_ccr < softmax_ccrs[-2]:
            best_valid_ccr = softmax_ccrs[-2]
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Loss for Epoch {epoch}: {epoch_loss}')


    model.load_state_dict(best_model_wts)
    test_Z, softmax_scores, test_truth_label = valid(model, test_loader, device)
    test_truth_label = np.array(test_truth_label)
    compute_all_scores(args, test_truth_label, softmax_scores,fpr_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--fix_seed", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default is 42.")

    parser.add_argument('--dataset', type=str, default='BBCNews', help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--device', type=str, default='2')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train [default: 500]')
    parser.add_argument('--unseen_label_index', type=int, default=-100)

    parser.add_argument('--training_rate', type=float, default= 0.1)
    parser.add_argument('--valid_rate', type=float, default= 0.1)

    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate')
    parser.add_argument('--openness', type=float, default=0.1, metavar='openness')
    parser.add_argument('--alpha', type=float, default=1, metavar='G', help='parameter for loss function')
    parser.add_argument('--beta', type=float, default=1, metavar='G', help='parameter for loss function')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--neighbor', type=int, default=25, metavar='N',
                        help='number of neighbors for graph construction')
    parser.add_argument('--lamb', type=float, default=1)
    parser.add_argument('--mix_alpha', type=float, default=1.5)

    args = parser.parse_args()
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    dataset_dict = {1: 'BBCNews', 2: 'Caltech20', 3: 'Hdigit', 4: 'Iaprtc12', 5: 'NUSWIDE-OBJ', 6: "VGGFace2"}
    select_dataset = [1, 2, 3, 4, 5, 6]
    fpr_values = [ 0.01, 0.05, 0.1, 0.5, 1]


    args.file = './res/openness_res.txt'

    for ii in select_dataset:

        if args.fix_seed:
            set_seed(args.seed)

        args.data = dataset_dict[ii]
        features, labels = load_data(args.data, "./data/")
        num_samples = features[0].shape[0]
        num_views = len(features)
        dims = [[x.shape[1]] for x in features]
        n_classes = len(np.unique(labels))

        open2 = (1 - args.openness) * (1 - args.openness)
        args.unseen_num = round((1 - open2 / (2 - open2)) * n_classes)
        print("unseen_num:%d" % args.unseen_num)

        original_num_classes = len(np.unique(labels))
        seen_labels = list(range(original_num_classes - args.unseen_num))
        y_true = reassign_labels(labels, seen_labels, args.unseen_label_index)

        num_classes = len(np.unique(y_true)) - 1
        NCLASSES = num_classes

        train_indices, test_valid_indices = special_train_test_split(y_true, args.unseen_label_index,
                                                                     test_size=1 - args.training_rate)
        valid_indices, test_indices = generate_partition(y_true[test_valid_indices], test_valid_indices,
                                                         args.valid_rate / (1 - args.training_rate))

        train_loader = torch.utils.data.DataLoader(
            Multi_view_OMix_data(num_views, train_indices, features, y_true),
            batch_size=args.batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            Multi_view_data(num_views, test_indices, features, y_true), batch_size=args.batch_size,
            shuffle=False)
        valid_loader = torch.utils.data.DataLoader(
            Multi_view_data(num_views, valid_indices, features, y_true), batch_size=args.batch_size,
            shuffle=False)

        train_and_evaluate(args)

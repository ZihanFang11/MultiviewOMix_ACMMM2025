
import torch


import numpy as np

def count_each_class_num(labels):
    '''
    Count the number of samples in each class.
    :param labels: A list or numpy array of class labels.
    :return: A dictionary with class labels as keys and counts as values.
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict:
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_partition(labels, indices, ratio):
    '''
    Partition the data into labeled and unlabeled sets based on a given ratio.
    :param labels: List of labels.
    :param indices: Indices corresponding to the labels.
    :param ratio: Ratio of data to be labeled.
    :return: Indices for labeled and unlabeled data.
    '''
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {}  # number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num:
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # at least one sample

    p_labeled = []
    p_unlabeled = []
    for idx in range(len(labels)):
        if labeled_each_class_num[labels[idx]] > 0:
            labeled_each_class_num[labels[idx]] -= 1
            p_labeled.append(indices[idx])
            total_num -= 1
        else:
            p_unlabeled.append(indices[idx])
    return p_labeled, p_unlabeled

def reassign_labels(y, seen_labels, unseen_label_index):
    '''
    Reassign labels to ensure continuity and handle unseen labels.
    :param y: Original labels.
    :param seen_labels: Labels considered as "seen" in the training data.
    :param unseen_label_index: Index to assign for unseen labels.
    :return: Array of reassigned labels.
    '''
    if isinstance(y, list):
        y = np.array(y)

    old_new_label_dict = {old_label: new_label for new_label, old_label in enumerate(seen_labels)}

    def convert_label(old_label):
        return old_new_label_dict.get(old_label, unseen_label_index)

    new_y = [convert_label(label) for label in y]
    return np.array(new_y)

def special_train_test_split(y, unseen_label_index, test_size):
    '''
    Split the dataset into training and testing sets, handling seen and unseen labels.
    :param y: Labels array.
    :param unseen_label_index: Index used for unseen labels.
    :param test_size: Proportion of the test set.
    :return: Indices for training and testing sets.
    '''
    if isinstance(y, list):
        y = np.array(y)

    seen_indices = np.where(y != unseen_label_index)[0]
    unseen_indices = np.where(y == unseen_label_index)[0]

    seen_train_indices, seen_test_indices = generate_partition(y[seen_indices], seen_indices, 1 - test_size)

    train_indices = seen_train_indices
    test_indices = np.concatenate([seen_test_indices, unseen_indices])
    return train_indices, test_indices


def distance(X, Y, square=True):
    '''
    Compute the squared Euclidean distance between each pair of vectors.
    :param X: Matrix X.
    :param Y: Matrix Y.
    :param square: Boolean to return squared distances or not.
    :return: Distance matrix.
    '''
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0) ** 2
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0) ** 2
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()  # Ensures non-negative distances
    if not square:
        result = torch.sqrt(result)
    return result

def build_CAN(X, num_neighbors, links=0):
    '''
    Build the Clustering-with-Adaptive-Neighbors (CAN) graph.
    :param X: Data matrix.
    :param num_neighbors: Number of neighbors to consider for graph construction.
    :param links: Additional links to add to the graph (optional).
    :return: Tuple of weights matrix and raw weights matrix.
    '''
    size = X.shape[1]
    num_neighbors = min(num_neighbors, size - 1)
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    return weights, raw_weights
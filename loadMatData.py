import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize



def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict
def generate_partition(labels,ind, ratio=0.1):
    each_class_num = count_each_class_num(labels)
    # number of labeled samples for each class
    labeled_each_class_num = {}
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(ind[idx])
            total_num -= 1
        else:
            p_unlabeled.append(ind[idx])
    return p_labeled, p_unlabeled

def load_data(dataset, path="./data/", ):

    feature_list = []
    data = sio.loadmat(path + dataset + '.mat')
    features = data['X']
    for i in range(features.shape[1]):
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        feature_list.append(feature)
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    return feature_list, labels



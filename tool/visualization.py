import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.ticker as mtick

VALID_FEATURE_VISUALIZATION_METHODS = ['TSNE', 'PCA']


def plot_sample_cv2(names, images, scores_dict, ground_truths, save_folder=None, norm_by_sample=False):
    # Get the number of samples
    total_number = len(images)

    scores = copy.deepcopy(scores_dict)
    # Normalize anomalies
    for k, v in scores.items():
        if not norm_by_sample:
            max_value = np.max(v)
            min_value = np.min(v)

            scores[k] = (scores[k] - min_value) / max_value * 255
            scores[k] = scores[k].astype(np.uint8)
        else:
            for i in range(len(scores[k])):
                scores[k][i] = (scores[k][i] - np.min(scores[k][i])) / np.max(scores[k][i]) * 255
                scores[k][i] = scores[k][i].astype(np.uint8)

    # Draw ground truths
    mask_images = []
    for idx in range(total_number):
        ground_truth = ground_truths[idx]
        mask_image = images[idx].copy()
        mask_image[ground_truth > 0.5] = (0, 0, 255)
        mask_images.append(mask_image)

    # Save images
    for idx in range(total_number):
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_ori.jpg'), images[idx])
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_gt.jpg'), mask_images[idx])

        for key in scores:
            heat_map = cv2.applyColorMap(scores[key][idx], cv2.COLORMAP_JET)
            vis_map = cv2.addWeighted(heat_map, 0.5, images[idx], 0.5, 0)
            cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_{key}.jpg'), vis_map)


def plot_anomaly_score_distributions(scores_dict, ground_truths_list, save_folder, class_name):
    ground_truths = np.stack(ground_truths_list, axis=0)

    N_COUNT = 100000

    for k, v in scores_dict.items():
        layer_score = np.stack(v, axis=0)
        normal_score = layer_score[ground_truths == 0]
        abnormal_score = layer_score[ground_truths != 0]

        plt.clf()
        plt.figure(figsize=(4, 3))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        sns.histplot(np.random.choice(normal_score, N_COUNT), color="green", bins=50, label='${d(p_n)}$',
                     stat='probability', alpha=.75)
        sns.histplot(np.random.choice(abnormal_score, N_COUNT), color="red", bins=50, label='${d(p_a)}$',
                     stat='probability', alpha=.75)

        plt.xlim([0, 3])

        save_path = os.path.join(save_folder, f'distributions_{class_name}_{k}.jpg')

        plt.savefig(save_path, bbox_inches='tight', dpi=300)


def visualize_feature(features, labels, legends, n_components=3, method='TSNE'):
    assert method in VALID_FEATURE_VISUALIZATION_METHODS
    assert n_components in [2, 3]

    if method == 'TSNE':
        model = TSNE(n_components=n_components)
    elif method == 'PCA':
        model = PCA(n_components=n_components)
    else:
        raise NotImplementedError

    feat_proj = model.fit_transform(features)

    if n_components == 2:
        ax = scatter_2d(feat_proj, labels)
    elif n_components == 3:
        ax = scatter_3d(feat_proj, labels)
    else:
        raise NotImplementedError

    plt.legend(legends)
    plt.axis('off')


def scatter_3d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes(projection='3d')

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter3D(feat_proj[label == l, 0],
                      feat_proj[label == l, 1],
                      feat_proj[label == l, 2], s=5)

    return ax1


def scatter_2d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes()

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter(feat_proj[label == l, 0],
                    feat_proj[label == l, 1], s=5)

    return ax1

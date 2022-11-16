import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix


def sklearn_metrics(gt, pred, save_name=None):
    acc = accuracy_score(gt, pred)
    p = precision_score(gt, pred, average='weighted')
    r = recall_score(gt, pred, average='weighted')
    f1 = f1_score(gt, pred, average='weighted')
    cm = confusion_matrix(gt, pred, normalize=None)
    normalized_cm = confusion_matrix(gt, pred, normalize='true')

    print('acc =', acc)
    print('f1 =', f1)
    print('precision =', p)
    print('recall =', r)
    # print("Confusion matrix")
    # print(cm)

    if save_name:
        base_name = os.path.splitext(save_name)[0]
        nor_save_name = base_name + '_normalized'
        plot_cm(cm, base_name)
        plot_cm(normalized_cm, nor_save_name, False)

def plot_cm(cm, save_name, annot=True):
    fig, ax = plt.subplots(figsize=(16,16), dpi=100)
    sns.heatmap(
        cm, cmap='Blues', square=True, annot=annot, fmt='g', ax=ax
    )
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.yaxis.set_ticklabels(range(cm.shape[0]))
    ax.xaxis.set_ticklabels(range(cm.shape[0]))
    plt.savefig('{}.png'.format(save_name))

def main():
    file_path = sys.argv[1]
    with open(file_path, 'r') as f:
        content = f.readlines()

    gt = []
    dt = []
    for line in content:
        new_line = line.replace('\n', '')
        gt_, dt_ = map(int, new_line.split(' ')[1:])
        gt.append(gt_)
        dt.append(dt_)

    save_name = os.path.split(file_path[-1])
    save_name = os.path.splitext(save_name)[0].replace('pred_', '')
    sklearn_metrics(gt, dt, save_name=None)

# coding: utf-8

import pdb
import mne
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from mne.viz import tight_layout
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

'''
# This script is to load epochs data.
'''

'''
# Function: set environment variables.
# Output: epochs_dir, dir of epochs fif files.
'''

root_dir = os.path.join('D:\\', 'RSVP_MEG_experiment')
epochs_dir = os.path.join(root_dir, 'epochs_saver', 'epochs')
result_dir = os.path.join(root_dir, 'extract_p300')

'''
# Function: load epochs.
# Output: epochs, loaded and concatenated epochs data.
# Output: epochs_data, data from epochs.
# Output: epochs_label, label from epochs.
'''

labels = None
epochs_data = None
epochs_list = []
for i in [5, 7, 9]:
    '''
    # Function: Reading epochs from -epo.fif.
    # Output: epochs, resampled epochs.
    '''
    epo_path = os.path.join(epochs_dir, 'meg_mxl_epochs_%d-epo.fif' % i)

    epochs = mne.read_epochs(epo_path, verbose=True)
    epochs.crop(tmin=0.0, tmax=0.8)

    # Attention!!!
    # This may cause poor alignment between epochs.
    # But this is necessary for concatenate_epochs.
    if epochs_list.__len__() != 0:
        epochs.info['dev_head_t'] = epochs_list[0].info['dev_head_t']
    epochs_list.append(epochs)

    '''
    # Function: Preparing dataset for MVPA.
    # Output: labels, labels of each trail.
    # Output: epochs_data, data of each trail.
    '''
    if labels is None:
        labels = epochs.events[:, -1]
        epochs_data = epochs.get_data()
    else:
        labels = np.concatenate([labels, epochs.events[:, -1]])
        epochs_data = np.concatenate([epochs_data, epochs.get_data()], 0)

epochs = mne.concatenate_epochs(epochs_list)
epochs_data = epochs.get_data()
epochs_label = epochs.events[:, -1]

'''
# Function: init pipelines and necessary components.
# Output: pca_pipeline: pipeline perform PCA.
# Output: xdawn_pipeline: pipeline perform Xdawn.
# Output: cv: Cross validation instance.
# Output: lr: LogisticRegression instance.
# Output: tsne: TSNE instance.
'''

pca_pipeline = make_pipeline(mne.decoding.Vectorizer(),
                             PCA(n_components=6),
                             MinMaxScaler())

xdawn_pipeline = make_pipeline(mne.preprocessing.Xdawn(n_components=6),
                               mne.decoding.Vectorizer(),
                               MinMaxScaler())

cv = StratifiedKFold(n_splits=10,
                     shuffle=True)

lr = LogisticRegression(penalty='l1',
                        solver='liblinear',
                        class_weight='balanced',
                        verbose=1)

svm = SVC(class_weight='balanced',
          gamma='scale',
          C=1,
          verbose=1)

tsne = TSNE(n_components=3,
            init='random',
            verbose=1)

'''
# Function: MVPA demo.
# Output: [no output].
'''


def report_MVPA_results(labels, preds,
                        target_names=['odd', 'norm'], title=''):
    print(title + 'reporting:')
    report = classification_report(labels, preds, target_names=target_names)
    print(report)
    with open(os.path.join(result_dir, 'accs_%s.txt' % title), 'w') as f:
        f.writelines(report)

    # Normalized confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    fig = plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title + 'Normalized Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


preds_xdawn = np.empty(len(labels))
for train, test in cv.split(epochs, labels):
    svm.fit(xdawn_pipeline.fit_transform(epochs[train]), labels[train])
    preds_xdawn[test] = svm.predict(xdawn_pipeline.transform(epochs[test]))

preds_pca = np.empty(len(labels))
for train, test in cv.split(epochs, labels):
    svm.fit(pca_pipeline.fit_transform(epochs_data[train]), labels[train])
    preds_pca[test] = svm.predict(pca_pipeline.transform(epochs_data[test]))

report_MVPA_results(labels, preds_xdawn, title='xdawn_svm')
report_MVPA_results(labels, preds_pca, title='pca_svm')
plt.show()

stophere
'''
# Function: pca, xdawn, tsne data and then plot manifold.
# Output: [no output].
'''


def plot3D_norm_odd(norms, odds, title='--'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(norms[:, 0], norms[:, 1], norms[:, 2], c='gray')
    ax.scatter(odds[:, 0], odds[:, 1], odds[:, 2], c='red')
    ax.set_title(title)


pca_tsne_data = tsne.fit_transform(pca_pipeline.fit_transform(epochs_data))
xdawn_tsne_data = tsne.fit_transform(xdawn_pipeline.fit_transform(epochs))

norms = pca_tsne_data[epochs_label == 2, :]
odds = pca_tsne_data[epochs_label == 1, :]
plot3D_norm_odd(norms, odds, title='pca_tsne')

norms = xdawn_tsne_data[epochs_label == 2, :]
odds = xdawn_tsne_data[epochs_label == 1, :]
plot3D_norm_odd(norms, odds, title='xdawn_tsne')

plt.show()


# evokedn = epochs['norm'].average()
# evokedn.plot(spatial_colors=True, show=False)

# evokedo = epochs['odd'].average()
# evokedo.plot(spatial_colors=True, show=False)

# plt.show()

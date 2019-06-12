# coding: utf-8

import mne
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report
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
# Output: xdawn_pipeline, pipeline perform Xdawn.
# Output: cv, Cross validation instance.
# Output: svm, SVM classifier instance.
'''

xdawn_pipeline = make_pipeline(mne.preprocessing.Xdawn(n_components=6),
                               mne.decoding.Vectorizer(),
                               MinMaxScaler())

cv = StratifiedKFold(n_splits=10,
                     shuffle=True)

svm = SVC(class_weight='balanced',
          gamma='scale',
          C=1,
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


preds_xdawn = np.empty(len(labels))
for train, test in cv.split(epochs, labels):
    svm.fit(xdawn_pipeline.fit_transform(epochs[train]), labels[train])
    preds_xdawn[test] = svm.predict(xdawn_pipeline.transform(epochs[test]))

report_MVPA_results(labels, preds_xdawn, title='xdawn_svm')

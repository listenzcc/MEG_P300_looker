# coding: utf-8

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import time
from matplotlib.backends.backend_pdf import PdfPages

'''
# This script is to load epochs data.
'''

'''
# Function: set environment variables.
# Output: epochs_dir, dir of epochs fif files.
'''
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
root_dir = os.path.join('D:\\', 'RSVP_experiment')
epochs_dir = os.path.join(root_dir, 'epochs_saver', 'epochs_0.1-7')
result_dir = os.path.join(root_dir, 'extract_p300')
pdf_path = os.path.join(result_dir, 'ts_joint_MEG_shift_%s.pdf' % time_stamp)


'''
# Function: load epochs.
# Output: epochs, loaded and concatenated epochs data.
# Output: epochs_data, data from epochs.
# Output: epochs_label, label from epochs.
'''
tmin, tmax = 0, 1
labels = None
epochs_data = None
epochs_list = []
for i in range(4, 12):  # [5, 7, 9]:
    '''
    # Function: Reading epochs from -epo.fif.
    # Output: epochs, resampled epochs.
    '''
    epo_path = os.path.join(epochs_dir, 'meg_mxl_epochs_%d-eposhift.fif' % i)

    epochs = mne.read_epochs(epo_path, verbose=True)
    epochs.crop(tmin=tmin, tmax=tmax)

    # Attention!!!
    # This may cause poor alignment between epochs.
    # But this is necessary for concatenate_epochs.
    print(epochs.info['dev_head_t'])
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

'''
# Function: plot evoked, joint.
'''
figures = []

freqs = np.linspace(0.1, 7, 20)
n_cycles = freqs / 2.
tfr_morlet = mne.time_frequency.tfr_morlet

for id in ['odd', 'norm']:
    evoked = epochs[id].average()
    fig = evoked.plot(spatial_colors=True, show=False)
    fig.suptitle(id)
    figures.append(fig)

    power, itc = tfr_morlet(epochs[id], freqs=freqs, n_cycles=n_cycles,
                            use_fft=True, return_itc=True, decim=1, n_jobs=12)
    fig = power.plot_joint(mode='mean', tmin=tmin, tmax=tmax, show=False)
    fig.suptitle(id)
    figures.append(fig)

print('Saving into pdf.')
with PdfPages(pdf_path) as pp:
    for fig in figures:
        pp.savefig(fig)

# plt.show()
plt.close('all')

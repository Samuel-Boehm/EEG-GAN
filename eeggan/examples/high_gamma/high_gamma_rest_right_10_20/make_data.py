#  Authors:	Kay Hartmann <kg.hartma@gmail.com>
#  			Samuel Böhm <samuel-boehm@web.de>
import sys
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')
from collections import OrderedDict
from typing import Tuple, List
import numpy as np
from eeggan.cuda import init_cuda

from eeggan.examples.high_gamma.make_data import make_dataset_for_subj, make_deep4_for_subj

FS = 512.
CLASSDICT_REST_RIGHT_HAND = OrderedDict([('rest', 3), ('right_hand', 1)])
SEGMENT_IVAL = (0.5, 2.25)
INPUT_LENGTH = int((SEGMENT_IVAL[1] - SEGMENT_IVAL[0]) * 512)
N_PROGRESSIVE_STAGES = 6
N_DEEP4 = 10
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4',
            'P8', 'O1', 'O2', 'M1', 'M2']
SUBJ_INDECES = np.arange(1, 15)
DATAPATH = '/home/boehms/eeg-gan/EEG-GAN/Data'
MODELPATH = '/home/boehms/eeg-gan/EEG-GAN/Data/Models'
SUBJ_ID = 1
RUN_ALL = False


def run(subj_ind: int = SUBJ_ID,
        dataset_path: str = DATAPATH,
        deep4_path: str = MODELPATH,
        channels: List[str] = CHANNELS,
        classdict: OrderedDict = CLASSDICT_REST_RIGHT_HAND,
        fs: float = FS,
        interval_times: Tuple[float, float] = SEGMENT_IVAL,
        n_progressive: int = N_PROGRESSIVE_STAGES,
        n_deep: int = N_DEEP4):
    init_cuda()  # activate cuda

    make_dataset_for_subj(subj_ind=subj_ind, dataset_path=dataset_path,
                          channels=channels, classdict=classdict,
                          fs=fs, interval_times=interval_times, verbose='DEBUG')

    # make_deep4_for_subj(subj_ind=subj_ind, dataset_path=dataset_path, deep4_path=deep4_path,
    #                     n_progressive=n_progressive, n_deep4=n_deep, verbose='INFO')


if __name__ == "__main__":
    if RUN_ALL:
        for i in SUBJ_INDECES:
            run(subj_ind=i)
    else:
        run(subj_ind=SUBJ_ID)

#  Authors:	Kay Hartmann <kg.hartma@gmail.com>
#  			Samuel Böhm <samuel-boehm@web.de>
import sys
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')
from collections import OrderedDict
from typing import Tuple, List
from eeggan.cuda import init_cuda

from eeggan.examples.high_gamma.make_data import make_dataset_for_subj, make_deep4_for_subj

FS = 512.
# CLASSDICT_REST_RIGHT_HAND = OrderedDict([('right_hand', 1), ('left_hand', 2),('rest', 3), ('feet', 4) ])
CLASSDICT_REST_RIGHT_HAND = OrderedDict([('right_hand', 1), ('rest', 3)])
SEGMENT_IVAL = (-0.5, 4.00)
INPUT_LENGTH = int((SEGMENT_IVAL[1] - SEGMENT_IVAL[0]) * FS)
N_PROGRESSIVE_STAGES = 6
N_DEEP4 = 10
CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']
N_EPOCHS = 100
EXPERIMENT = 'Thesis'
DATAPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Data/{EXPERIMENT}'
MODELPATH = f'/home/boehms/eeg-gan/EEG-GAN/Data/Models/{EXPERIMENT}'
NAME = 'full'

def run(name: str = NAME,
        dataset_path: str = DATAPATH,
        deep4_path: str = MODELPATH,
        channels: List[str] = CHANNELS,
        classdict: OrderedDict = CLASSDICT_REST_RIGHT_HAND,
        fs: float = FS,
        interval_times: Tuple[float, float] = SEGMENT_IVAL,
        n_progressive: int = N_PROGRESSIVE_STAGES,
        n_deep: int = N_DEEP4,
        n_epochs = N_EPOCHS):

    init_cuda()  #activate cuda

    make_dataset_for_subj(name=name, dataset_path=dataset_path,
                          channels=channels, classdict=classdict,
                          fs=fs, interval_times=interval_times, verbose='INFO')

    #make_deep4_for_subj(name=name, dataset_path=dataset_path, deep4_path=deep4_path,
    #                    n_progressive=n_progressive, n_deep4=n_deep, verbose='INFO', n_epochs=n_epochs)


if __name__ == "__main__":
    run()

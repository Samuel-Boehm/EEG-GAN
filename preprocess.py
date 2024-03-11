import os

from constants import BASEDIR
from gan.data.preprocess import fetch_and_unpack_schirrmeister2017_moabb_data



if __name__ == '__main__':
    
    MAPPING = {'right': 0, 'rest': 1}
    INTERVAL_TIMES = (-0.5, 2.5)
    FS = 256
    CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
            'T7','C3','Cz','C4','T8','P7','P3',
            'Pz','P4','P8','O1','O2','M1','M2']
    
    # Check if data folder exists:
    if not os.path.exists(os.path.join(BASEDIR, 'data')):
        os.makedirs(os.path.join(BASEDIR, 'data'))

    ds = fetch_and_unpack_schirrmeister2017_moabb_data(CHANNELS, INTERVAL_TIMES, FS, MAPPING)

    ds.save(os.path.join(BASEDIR, 'data', 'clinical'))
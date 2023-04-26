#  Author: Samuel Boehm <samuel-boehm@web.de>
import sys
# setting path
sys.path.append('/home/boehms/eeg-gan/EEG-GAN/EEG-GAN')

from eeggan.examples.high_gamma.braindecode_hack import IntermediateOutputWrapper
from eeggan.training.handlers.metrics import WassersteinMetric, InceptionMetric, ClassificationMetric, FrechetMetric
from eeggan.data.dataset import Data
from eeggan.examples.high_gamma.make_data import load_deeps4
from eeggan.cuda import to_cuda, init_cuda
from eeggan.data.preprocess.resample import downsample
from utils import create_balanced_datasets

from torch import Tensor
import pandas as pd
import numpy as np
from torch import Tensor
import argparse
import os
import joblib
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--gan_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Results/Thesis/')
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--stage', type=int, required=True)
parser.add_argument('--subject', type=int, required=False)
parser.add_argument('--data_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Data/Thesis/')
parser.add_argument('--deep4_path', type=str, required=False, default='/home/boehms/eeg-gan/EEG-GAN/Data/Models/Thesis/')
parser.add_argument('--downsample', type=bool, required=False, default=False)

args = parser.parse_args()

batchsize = 3000

init_cuda()

gan_path = os.path.join(args.gan_path, args.name)

# Change the Batch output class to have a meaningfull name instead of epoch. 
class BatchOutput:
    def __init__(self, batch_real: Data[Tensor], batch_fake: Data[Tensor], i_epoch: str = 'name'):
        self.i_epoch = i_epoch
        self.batch_real = batch_real
        self.batch_fake = batch_fake

def calculate_metrics(real, fake, stage, data_name, deep4_path, batchsize):

    idx = np.arange(batchsize)

    np.random.shuffle(idx)
    
    idx = np.split(idx, 2)

    realVSreal = BatchOutput(Data(real.X[idx[0]], real.y[idx[0]], real.y_onehot[idx[0]]),
                            Data(real.X[idx[1]], real.y[idx[1]], real.y_onehot[idx[1]]))
    
    realVSfake = BatchOutput(real, fake)

    fakeVSfake = BatchOutput(Data(fake.X[idx[0]], fake.y[idx[0]], fake.y_onehot[idx[0]]),
                            Data(fake.X[idx[1]], fake.y[idx[1]], fake.y_onehot[idx[1]]))
    
    batches = [realVSreal, realVSfake, fakeVSfake]
    batch_names = ['real vs real', 'real vs fake', 'fake vs fake']
    
    # load trained deep4s for stage
 
    deep4s = load_deeps4(data_name, stage-1, deep4_path)

    select_modules = ['pool_4', 'softmax']

    deep4s = [to_cuda(IntermediateOutputWrapper(select_modules, deep4)) for deep4 in deep4s]

    # initiate metrics
    metric_wasserstein = WassersteinMetric(100, np.prod(real.X.shape[1:]).item(), tb_writer=None)
    
    sample_factor = 2 ** (6 - stage)
    metric_inception = InceptionMetric(deep4s, sample_factor, tb_writer=None)
    metric_frechet = FrechetMetric(deep4s, sample_factor, tb_writer=None)
    metric_classification = ClassificationMetric(deep4s, sample_factor, tb_writer=None)


    
    metrics = [metric_wasserstein, metric_inception, metric_classification, metric_frechet]
    metric_names = ['SWD', 'IS', 'Class', 'FID']
    
    df = pd.DataFrame()
    
    for i, batch in enumerate(batches):
        batch.i_epoch = batch_names[i]
        print(args.name, ' running ',  batch_names[i])
        for metric in metrics:
            metric.update(batch)
    
    for i, metric in enumerate(metrics):
        df[metric_names[i]] = list(zip(*metric.values))[1]
        
    df.index = batch_names

    df[['IS', 'IS_std']] = pd.DataFrame(df['IS'].tolist(), index=df.index)
    df[['SWD', 'SWD_std']] = pd.DataFrame(df['SWD'].tolist(), index=df.index)
    df[['Class', 'Class_std']] = pd.DataFrame(df['Class'].tolist(), index=df.index)
    df[['FID', 'FID_std']] = pd.DataFrame(df['FID'].tolist(), index=df.index)
    return df

if __name__ == '__main__':

    if args.name == 'torch_cGAN': 
        real = joblib.load(os.path.join(args.data_path, f'tGAN_real_stage{args.stage}.dataset'))
        fake = joblib.load(os.path.join(args.data_path, f'tGAN_fake_stage{args.stage}.dataset'))

    else:
        real, fake = create_balanced_datasets(gan_path, args.data_path, args.data_name, args.stage, None, False, batchsize*2)
    
    if args.downsample == True:
        print('data downsampled from ', real.X.shape)
        real.X = downsample(real.X, 2, axis=2)
        fake.X = downsample(fake.X, 2, axis=2)
        print('to ', real.X.shape)

    df = calculate_metrics(real, fake, args.stage, args.data_name, args.deep4_path, batchsize*2)
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv(f'/home/boehms/eeg-gan/EEG-GAN/EEG-GAN/analysis_scripts/outputs/{args.name}_stage{args.stage}.csv')
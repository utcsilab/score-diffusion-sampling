#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from dotmap import DotMap
import torch, sys, os, json, argparse
sys.path.append('.')

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)
args = DotMap(json.load(open(parser.parse_args().config_path)))

from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest, NCSNv2Deeper, NCSNv2

from annealedLangevin import ald
from utils            import *

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.model.gpu);

# Target weights - replace with target model
contents = torch.load(args.sampling.target_model)

# Extract config
config = contents['config']
config.model.depth = args.model.depth
config.sampling = args.sampling
config.sampling.sigma = 0.
config.model.num_classes = args.model.num_classes
config.sampling.num_steps = config.model.num_classes - config.sampling.sigma_offset

# Range of SNR, test channels and hyper-parameters
config.sampling.noise_range = 10 ** (-torch.tensor(config.sampling.snr_range) / 10.)

# Get a model
if config.model.depth == 'large':
    diffuser = NCSNv2Deepest(config)
elif config.model.depth == 'medium':
    diffuser = NCSNv2Deeper(config)
elif config.model.depth == 'low':
    diffuser = NCSNv2(config)

if len(config.training.sigmas) > 0:
    diffuser.sigmas = config.training.sigmas

config.training.sigmas = diffuser.sigmas
diffuser = diffuser.cuda()
# !!! Load weights
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()

if not config.sampling.step_size or config.sampling.prior_sampling == 1:
    # Choose the core step size (epsilon) according to [Song '20]
    candidate_steps = np.logspace(-11, -7, 10000)
    step_criterion  = np.zeros((len(candidate_steps)))
    gamma_rate      = 1 / config.model.sigma_rate
    for idx, step in enumerate(candidate_steps):
        sigma_squared   = config.model.sigma_end ** 2
        one_minus_ratio = (1 - step / sigma_squared) ** 2
        big_ratio       = 2 * step /\
            (sigma_squared - sigma_squared * one_minus_ratio)
        
        # Criterion
        step_criterion[idx] = one_minus_ratio ** config.sampling.steps_each * \
            (gamma_rate ** 2 - big_ratio) + big_ratio
        
    best_idx        = np.argmin(np.abs(step_criterion - 1.))
    fixed_step_size = candidate_steps[best_idx]
    config.sampling.step_size    = torch.tensor(fixed_step_size)

config.data = args.data
config.training = args.training

# Global results
result_dir = './results/' + config.data.file + '_' + config.data.dataloader + '/' + config.sampling.target_model.split("/")[-2]

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

forward_model = None
Y_adj, oracle, forward_operator, adjoint_operator = None, None, None, None

print('Dataset: ' + config.data.file)
print('Dataloader: ' + config.data.dataloader)
print('\nStep Size: ' + str(np.float64(config.sampling.step_size)) + '\n') 
best_images = []

if config.sampling.prior_sampling == 0:
    print('Forward Class: ' + config.sampling.forward_class)
    forward_model = globals()[config.sampling.forward_class]()
    Y_adj, oracle, forward_operator, adjoint_operator = forward_model.DataLoader(config)
    config.sampling.channels = oracle.shape[0]
    init_val_X = torch.randn_like(oracle).cuda()

else:
    real = torch.randn(config.sampling.channels, config.sampling.oracle_shape[0], config.sampling.oracle_shape[1], dtype = torch.float)
    imag = torch.randn(config.sampling.channels, config.sampling.oracle_shape[0], config.sampling.oracle_shape[1], dtype = torch.float)
    init_val_X = torch.complex(real, imag).cuda()
    config.sampling.noise_range = [1]
    config.sampling.noise_boost = 1
    oracle = init_val_X.clone()
    Y_adj = torch.zeros(len(config.sampling.noise_range))
    config.sampling.sampling_file = 'prior'

# For each SNR value
for snr_idx, config.sampling.local_noise in tqdm(enumerate(config.sampling.noise_range)):
    # Starting with random noise and running Annealed Langevin Dynamics
    print('\n\nSampling for SNR Level ' + str(snr_idx) + ': ' + str(config.sampling.snr_range[snr_idx]))
    current = init_val_X.clone()
    best_images.append(ald(diffuser, config, Y_adj[snr_idx], oracle, current, forward_operator, adjoint_operator))
        
torch.cuda.empty_cache()

# Save results to file based on noise
save_dict = {'snr_range': config.sampling.snr_range,
            'config': config,
            'oracle_H': oracle,
            'best_images': best_images}

torch.save(save_dict, result_dir + '/' + config.sampling.sampling_file + '.pt')
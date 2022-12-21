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

from loaders          import *
from annealedLangevin import ald
from utils            import MulticoilForwardMRI

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

# Sometimes
torch.backends.cudnn.benchmark = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);

# Target weights - replace with target model
target_weights = args.target_model
contents = torch.load(target_weights)

# Extract config
config = contents['config']
config.sampling.sigma = 0. # Nothing here
config.data.sampling_path = args.sampling_path
config.data.sampling_file = args.sampling_file

# !!! 'Beta' in paper
config.noise_boost = args.noise_boost
config.sigma_offset = args.sigma_offset
config.dc_boost = args.dc_boost
config.model.step_size = args.step_size
config.num_steps = config.model.num_classes - config.sigma_offset
config.sampling.steps_each = 4

# Range of SNR, test channels and hyper-parameters
snr_range          = np.array(args.snr_range)
noise_range        = 10 ** (-snr_range / 10.)
config.model.K     = args.K 

# Get a model
if args.depth == 'large':
    diffuser = NCSNv2Deepest(config)
elif args.depth == 'medium':
    diffuser = NCSNv2Deeper(config)
elif args.depth == 'low':
    diffuser = NCSNv2(config)

diffuser = diffuser.cuda()
# !!! Load weights
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()

if config.model.step_size == 0:
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
    config.model.step_size    = fixed_step_size

# Global results
result_dir = './results/' + args.sampling_file

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

MRI_model = MulticoilForwardMRI()
Y, oracle, forward_operator, adjoint_operator, norm_operator = MRI_model.DataLoader(config)
adjoint_image = adjoint_operator(Y)

# batch size now changed to 3 for 3 different alpha basises
real = torch.randn(config.model.K, oracle.shape[1], oracle.shape[2], dtype = torch.float)
imag = torch.randn(config.model.K, oracle.shape[1], oracle.shape[2], dtype= torch.float)
init_val_X = torch.complex(real, imag).cuda()

normalize_values = []
for k in range(config.model.K):
    normalize_values.append(torch.quantile(torch.abs(adjoint_image[k,:,:]), 0.95))

normalize_values_tensor = torch.tensor(normalize_values).cuda()
config.inference.norm_operator = normalize_values_tensor[:,None,None]
best_images = []

# For each SNR value
for snr_idx, local_noise in tqdm(enumerate(noise_range)):

    # Starting with random noise
    current = init_val_X.clone()
    config.local_noise = local_noise
    
    # Annealed Langevin Dynamics
    best_images.append(ald(diffuser, config, Y, oracle, current, forward_operator, adjoint_operator, norm_operator))
        
torch.cuda.empty_cache()

# Save results to file based on noise
save_dict = {'snr_range': snr_range,
            'config': config,
            'oracle_H': oracle,
            'best_images': best_images}

torch.save(save_dict, result_dir + '/sigma_begin293_sigma_end0.0007_num_classes2311.0_sigma_rate0.9944_epochs300.0.pt')
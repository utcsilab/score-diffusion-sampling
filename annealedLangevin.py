import torch
import numpy as np
from tqdm import tqdm as tqdm

def ald(diffuser, config, Y, oracle, current, forward_operator, adjoint_operator, norm_operator):
    
    K = config.model.K
    step_images = []
    oracle_log = torch.zeros((config.num_steps, K))
    oracle_flatten = oracle.flatten(start_dim=1)

    min_nrmse_img = torch.zeros((K, current.shape[1], current.shape[2]), dtype=torch.complex64)
    min_nrmse_idx = torch.ones(K) * torch.inf
    min_nrmse = torch.ones(K) * torch.inf
    meas_grad = 0

    with torch.no_grad():
        for step_idx in tqdm(range(config.num_steps)):
            # Compute current step size and noise power
            current_sigma = diffuser.sigmas[step_idx + config.sigma_offset].item()
            config.inference.current_sigma = current_sigma
            
            # Compute alpha
            alpha = config.model.step_size * (current_sigma / config.model.sigma_end) ** 2
                
            # Labels for diffusion model
            labels = torch.ones(current.shape[0]).cuda() * (step_idx + config.sigma_offset)
            labels = labels.long()
            
            # For each step spent at that noise level
            for inner_idx in range(config.sampling.steps_each):
                # Compute score
                current_real = torch.view_as_real(current).permute(0, 3, 1, 2)

                # Get score
                score = diffuser(current_real, labels)
                
                # View as complex
                score = torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())
                config.inference.score = score

                if (config.prior_sampling == 1):
                    config.noise_boost = 1            
                
                else:
                    # Compute gradient for measurements in un-normalized space
                    current_norm = current * config.inference.norm_operator
                    H_forw = forward_operator(current_norm) 
                    H_adj = adjoint_operator(H_forw - Y)

                    # Re-normalize gradient to match score model
                    meas_grad = norm_operator(H_adj, config)

                # Annealing noise
                grad_noise = np.sqrt(2 * alpha * config.noise_boost) * torch.randn_like(current) 
                
                # Apply update
                current = current + alpha * (score - (meas_grad / config.dc_boost)) + grad_noise
                current_flatten = current.flatten(start_dim=1)

                # NRMSE
                nrmse = (torch.norm((oracle_flatten - current_flatten), dim=1) / torch.norm(oracle_flatten, dim=1))
                
            # Store Min NRMSE Image
            for i in range(len(nrmse)):
                if (nrmse[i] < min_nrmse[i]):
                    min_nrmse[i] = nrmse[i]
                    min_nrmse_idx[i] = step_idx
                    min_nrmse_img[i] = current[i]
                        
            # Store NRMSE
            oracle_log[step_idx] = nrmse
            
            if step_idx % 100 == 0:
                print('\nStep %d, NRMSE %3f' % (step_idx, nrmse.mean(dim=0)))
                step_images.append(current)
    
    return {'min_nrmse_img': min_nrmse_img, 
            'min_nrmse_idx': min_nrmse_idx,
            'min_nrmse': min_nrmse,
            'step_images': step_images,
            'oracle_log': oracle_log,
            'Y': Y}
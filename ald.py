import numpy as np
import torch
from tqdm import tqdm as tqdm

def annealedLangevin(diffuser, config, Y, oracle, current, forward_operator, adjoint_operator, norm_operator):
    
    K = config.model.num_channels
    step_images = []
    oracle_log = np.zeros((config.num_steps, K))

    min_loss_img = torch.zeros((K, current.shape[1], current.shape[2]), dtype=torch.complex64)
    min_loss_idx = torch.ones(K) * torch.inf
    min_loss = np.ones(K) * np.inf
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
                
                # else:
                #     # Compute gradient for measurements in un-normalized space
                #     current_norm = current * config.inference.norm_operator
                #     H_forw = forward_operator(current_norm) 
                #     H_adj = adjoint_operator(H_forw - Y)

                #     # Re-normalize gradient to match score model
                #     meas_grad = norm_operator(H_adj, config)

                # Annealing noise
                grad_noise = np.sqrt(2 * alpha * config.noise_boost) * torch.randn_like(current) 
                
                # Apply update
                current = current + alpha * (score - (meas_grad / config.dc_boost)) + grad_noise
                # loss = (torch.sum(torch.square(torch.abs(current - oracle)), dim=(-1, -2)) / torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).cpu().numpy()
                loss = torch.zeros(current.shape)
                
            # # Store Min Loss Image
            # for i in range(len(loss)):
            #     if (loss[i] < min_loss[i]):
            #         min_loss[i] = loss[i]
            #         min_loss_idx[i] = step_idx
            #         min_loss_img[i] = current[i]
                        
            # # Store loss
            # oracle_log[step_idx] = loss
            
            if step_idx % 100 == 0:
                step_images.append(current)
    
    return {'min_loss_img': min_loss_img, 
            'min_loss_idx': min_loss_idx,
            'min_loss': min_loss,
            'step_images': step_images,
            'oracle_log': oracle_log,
            'Y': Y}
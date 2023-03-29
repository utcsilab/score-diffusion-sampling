import torch
from tqdm import tqdm as tqdm

def ald(diffuser, config, Y_adj, oracle, current, forward_operator, adjoint_operator):
    step_images = []
    oracle_log = torch.zeros((config.sampling.num_steps, config.sampling.channels))
    min_nrmse_img = torch.zeros((config.sampling.channels, current.shape[1], current.shape[2]), dtype=torch.complex64)
    min_nrmse_idx = 0
    min_nrmse = torch.inf
    meas_grad = 0

    with torch.no_grad():
        for step_idx in tqdm(range(config.sampling.num_steps)):
            # Compute current step size, noise power and alpha
            current_sigma = diffuser.sigmas[step_idx + config.sampling.sigma_offset].item()
            alpha = torch.tensor(config.sampling.step_size * (current_sigma / config.model.sigma_end) ** 2)
                
            # Labels for diffusion model
            labels = torch.ones(current.shape[0]).cuda() * (step_idx + config.sampling.sigma_offset)
            labels = labels.long()
            
            # For each step spent at that noise level
            for inner_idx in range(config.sampling.steps_each):
                # Compute score
                current_real = torch.view_as_real(current).permute(0, 3, 1, 2)
                score = diffuser(current_real, labels)
                score = torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())

                if (config.sampling.prior_sampling == 0):
                    # Compute gradient and normalize
                    H_adj = adjoint_operator(forward_operator(current))
                    meas_grad = config.sampling.dc_boost * (H_adj - Y_adj) / (config.sampling.local_noise/2. + current_sigma ** 2)

                # Annealing noise and update
                grad_noise = torch.sqrt(2 * alpha * config.sampling.noise_boost) * torch.randn_like(current) 
                current = current + alpha * (score - meas_grad) + grad_noise
                nrmse = (torch.sum(torch.square(torch.abs(current - oracle)), dim=(-1, -2)) / torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2)))
                
            # Store Min NRMSE Image
            if (torch.mean(nrmse) < min_nrmse):
                min_nrmse = torch.mean(nrmse)
                min_nrmse_idx = step_idx
                min_nrmse_img = current
                        
            # Store NRMSE
            oracle_log[step_idx] = nrmse
            
            if step_idx % 100 == 0:
                print('\nStep %d, NRMSE %3f' % (step_idx, torch.mean(nrmse)))
                step_images.append(current)
                if (config.sampling.prior_sampling == 0):
                    print('\nMeasurement Grad before Normalization: ' + str(round(float(torch.mean(torch.abs(H_adj - Y_adj))), 2)))
                    print('Measurement Grad after Normalization: ' + str(round(float(torch.mean(torch.abs(meas_grad))), 2)))
                    print('Score Power: ' + str(round(float(torch.mean(torch.abs(score))), 2)))
    
    return {'min_nrmse_img': min_nrmse_img, 
            'min_nrmse_idx': min_nrmse_idx,
            'min_nrmse': min_nrmse,
            'step_images': step_images,
            'oracle_log': oracle_log,
            'Y': Y_adj}
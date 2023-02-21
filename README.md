# Generic Score-Based Generative Model Sampling Pipeline

by [Asad Aali](http://users.ece.utexas.edu/~jtamir/csilab.html) and [Jon Tamir](http://users.ece.utexas.edu/~jtamir/csilab.html), UT CSI Lab.

This repo contains an implementation for the paper [Improved Techniques for Training Score-Based Generative Models](http://arxiv.org/abs/2006.09011). 

by [Yang Song](http://yang-song.github.io/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/), Stanford AI Lab.

-----------------------------------------------------------------------------------------

We streamline the method proposed in [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600). Score-based generative models are flexible neural networks trained to capture the score function of an underlying data distribution—a vector field pointing to directions where the data density increases most rapidly. We present a generic pipeline for performing sampling on pre-trained score-based generative models.

![samples](assets/FastMRI.png)

(**From left to right**: Our samples on FastMRI 384x384)

## Running Experiments

### Dependencies

Run the following to install all necessary python packages for our code.

```bash
pip install -r requirements.txt
```

### Configuration file structure

`ald.py` contains code for running the score-based generative model sampling algorithm. However, we use `sampling.py` to prepare the dataset and confgurations for samplin. `sampling.py` is the file that you should run for sampling:

```
Usage:
python score-diffusion-training/sampling.py --config_path [CONFIG FILE PATH]

Configuration files are kept in `configs/`. Configuration files are structured as:

```bash
├── sampling # sampling configurations
│   └── <dataloader>: specify the dataloder for loaading data for posterior sampling
│   └── <forward_class">: specify the name of forward class in a utils.py file. this forward class will be used for computing forward and adjoint for annealed langevin dynamics algorithm during posterior sampling.
│   └── <sampling_file>: name of the dataset/file for storing results
│   └── <sampling_path>: path where the sampling measurements are kept
│   └── <target_model>: location of the pre-trained model for sampling
│   └── <noise_boost>: power of the additive noise during each annealed langevin dynamics step
│   └── <dc_boost>: controls scaling of the data consistency term in the langevin algorithm
│   └── <sigma_offset>: controls number of initial sigma levels to skip when sampling. skips the highest sigma levels
│   └── <prior_sampling>: specify 1 for prior sampling and 0 for posterior. note that forward class must be provided if posterior sampling
│   └── <image_size>: provide pixels for the image to be generated in the format: [1, 384, 384]
│   └── <snr_range>: specify SNR levels for the undersampled measurements to perform sampling on as a list: example -> [-20, -15, -10, -5, 0, 5]. please note that you must apply the specified noise from SNR list to your undersampled measurements in the utils.py file.
│   └── <steps_each>: specify # of steps for each noise level during the langevin algorithm
│   └── <seed>: sampling seed
```

Samples will be saved in `results/`.

## References

```bib
@article{jalal2021robust,
  title={Robust Compressed Sensing MRI with Deep Generative Priors},
  author={Jalal, Ajil and Arvinte, Marius and Daras, Giannis and Price, Eric and Dimakis, Alexandros G and Tamir, Jonathan I},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```

```bib
@inproceedings{song2020improved,
  author    = {Yang Song and Stefano Ermon},
  editor    = {Hugo Larochelle and
               Marc'Aurelio Ranzato and
               Raia Hadsell and
               Maria{-}Florina Balcan and
               Hsuan{-}Tien Lin},
  title     = {Improved Techniques for Training Score-Based Generative Models},
  booktitle = {Advances in Neural Information Processing Systems 33: Annual Conference
               on Neural Information Processing Systems 2020, NeurIPS 2020, December
               6-12, 2020, virtual},
  year      = {2020}
}
```

```bib
@inproceedings{song2019generative,
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11895--11907},
  year={2019}
}
```

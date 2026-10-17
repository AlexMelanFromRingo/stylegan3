## Alias-Free Generative Adversarial Networks (StyleGAN3)<br><sub>Official PyTorch implementation of the NeurIPS 2021 paper</sub>

![Teaser image](./docs/stylegan3-teaser-1920x1006.png)

**Alias-Free Generative Adversarial Networks**<br>
Tero Karras, Miika Aittala, Samuli Laine, Erik H&auml;rk&ouml;nen, Janne Hellsten, Jaakko Lehtinen, Timo Aila<br>
https://nvlabs.github.io/stylegan3<br>

Abstract: *We observe that despite their hierarchical convolutional nature, the synthesis process of typical generative adversarial networks depends on absolute pixel coordinates in an unhealthy manner. This manifests itself as, e.g., detail appearing to be glued to image coordinates instead of the surfaces of depicted objects. We trace the root cause to careless signal processing that causes aliasing in the generator network. Interpreting all signals in the network as continuous, we derive generally applicable, small architectural changes that guarantee that unwanted information cannot leak into the hierarchical synthesis process. The resulting networks match the FID of StyleGAN2 but differ dramatically in their internal representations, and they are fully equivariant to translation and rotation even at subpixel scales. Our results pave the way for generative models better suited for video and animation.*

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

## About this fork

The official code targets Python 3.8, PyTorch 1.9 and CUDA 11.1 (2021). This fork runs on current versions
(Python 3.12+, PyTorch 2.x, CUDA 12/13, NumPy 2, SciPy 1.18) with a plain `pip install`, without Docker or conda:

- **Setup**: `pip install -r requirements.txt`. The custom CUDA kernels compile against a CUDA toolkit installed with pip
  (no system wide CUDA installation), and when they cannot be compiled the code falls back to native PyTorch
  implementations instead of stopping with a build error.
- **Visualizer**: works again - ported from pyimgui (which no longer installs) to [imgui-bundle](https://github.com/pthom/imgui_bundle),
  with fixes for Wayland / WSLg and separate-process rendering.
- **Fixes** for current libraries: training, FID and the visualizer failed on recent versions of PyTorch, SciPy, matplotlib and Python.
- **Speed**: less memory and time spent on bookkeeping during training, see [docs/modernization.md](./docs/modernization.md).
- **New options**: the relativistic pairing loss of R3GAN (`--loss=rpgan`), the FD<sub>DINOv2</sub> metric (`--metrics=fd_dinov2_50k_full`).

Network pickles are fully compatible in both directions. The details, measurements and the reasons behind each change are in
[docs/modernization.md](./docs/modernization.md).

## Release notes

This repository is an updated version of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch), with several new features:
- Alias-free generator architecture and training configurations (`stylegan3-t`, `stylegan3-r`).
- Tools for interactive visualization (`visualizer.py`), spectral analysis (`avg_spectra.py`), and video generation (`gen_video.py`).
- Equivariance metrics (`eqt50k_int`, `eqt50k_frac`, `eqr50k`).
- General improvements: reduced memory usage, slightly faster training, bug fixes.

Compatibility:
- Compatible with old network pickles created using [stylegan2-ada](https://github.com/NVlabs/stylegan2-ada) and [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).  (Note: running old StyleGAN2 models on StyleGAN3 code will produce the same results as running them on stylegan2-ada/stylegan2-ada-pytorch.  To benefit from the StyleGAN3 architecture, you need to retrain.)
- Supports old StyleGAN2 training configurations, including ADA and transfer learning. See [Training configurations](./docs/configs.md) for details.
- Improved compatibility with Ampere GPUs and newer versions of PyTorch, CuDNN, etc.

## Synthetic image detection

While new generator approaches enable new media synthesis capabilities, they may also present a new challenge for AI forensics algorithms for detection and attribution of synthetic media. In collaboration with digital forensic researchers participating in DARPA's SemaFor program, we curated a synthetic image dataset that allowed the researchers to test and validate the performance of their image detectors in advance of the public release. Please see [here](https://github.com/NVlabs/stylegan3-detector) for more details.

## Additional material

- [Result videos](https://nvlabs-fi-cdn.nvidia.com/stylegan3/videos/)
- [Curated example images](https://nvlabs-fi-cdn.nvidia.com/stylegan3/images/)
- [StyleGAN3 pre-trained models](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan3) for config T (translation equiv.) and config R (translation and rotation equiv.)
  > <sub>Access individual networks via `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/<MODEL>`, where `<MODEL>` is one of:</sub><br>
  > <sub>`stylegan3-t-ffhq-1024x1024.pkl`, `stylegan3-t-ffhqu-1024x1024.pkl`, `stylegan3-t-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan3-r-ffhq-1024x1024.pkl`, `stylegan3-r-ffhqu-1024x1024.pkl`, `stylegan3-r-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan3-t-metfaces-1024x1024.pkl`, `stylegan3-t-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan3-r-metfaces-1024x1024.pkl`, `stylegan3-r-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan3-t-afhqv2-512x512.pkl`</sub><br>
  > <sub>`stylegan3-r-afhqv2-512x512.pkl`</sub><br>
- [StyleGAN2 pre-trained models](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan2) compatible with this codebase
  > <sub>Access individual networks via `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/<MODEL>`, where `<MODEL>` is one of:</sub><br>
  > <sub>`stylegan2-ffhq-1024x1024.pkl`, `stylegan2-ffhq-512x512.pkl`, `stylegan2-ffhq-256x256.pkl`</sub><br>
  > <sub>`stylegan2-ffhqu-1024x1024.pkl`, `stylegan2-ffhqu-256x256.pkl`</sub><br>
  > <sub>`stylegan2-metfaces-1024x1024.pkl`, `stylegan2-metfacesu-1024x1024.pkl`</sub><br>
  > <sub>`stylegan2-afhqv2-512x512.pkl`</sub><br>
  > <sub>`stylegan2-afhqcat-512x512.pkl`, `stylegan2-afhqdog-512x512.pkl`, `stylegan2-afhqwild-512x512.pkl`</sub><br>
  > <sub>`stylegan2-brecahad-512x512.pkl`, `stylegan2-cifar10-32x32.pkl`</sub><br>
  > <sub>`stylegan2-celebahq-256x256.pkl`, `stylegan2-lsundog-256x256.pkl`</sub><br>

## Requirements

* Linux, Windows or WSL2 and an NVIDIA GPU. A single GPU with 8 GB of memory is enough for 256x256 models (see [Training](#training)).
* 64-bit Python 3.9 &ndash; 3.13 and PyTorch 2.1 or later.
* Python libraries: [requirements.txt](./requirements.txt).

```.bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv && source .venv/bin/activate
pip install -r requirements.txt                       # Windows: install the CUDA build of PyTorch first, see https://pytorch.org
```

**Custom CUDA kernels.** Some layers (filtered leaky ReLU, resampling, bias + activation) use custom CUDA kernels that are
compiled on first use and make training about 4x faster. Compiling them needs a C++ compiler (GCC on Linux, Visual Studio
on Windows) and a CUDA toolkit whose major version matches the one PyTorch was built with (`python -c "import torch; print(torch.version.cuda)"`).
On Linux the toolkit can simply be installed with pip, into the same environment - for a PyTorch built with CUDA 13.0:

```.bash
pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-cuda-cccl==13.0.*"
```

It is found automatically, also when an older system toolkit is installed. If the kernels cannot be compiled, a warning
names the reason (and the exact command for your PyTorch version), and native PyTorch implementations of the same ops are used:
everything works, training is slower. Set `STYLEGAN_CUSTOM_OPS=off` to skip the compilation, or `STYLEGAN_CUSTOM_OPS=require`
to fail instead of falling back. On Windows, install [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/)
with the C++ workload and the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) to use the kernels.

* Docker and conda still work ([Dockerfile](./Dockerfile), [environment.yml](./environment.yml)), but are not needed.

See [Troubleshooting](./docs/troubleshooting.md) for help on common installation and run-time problems.

## Getting started

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames or URLs:

```.bash
# Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
python gen_images.py --outdir=out --trunc=1 --seeds=2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

# Render a 4x2 grid of interpolations for seeds 0 through 31.
python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```

Outputs from the above commands are placed under `out/*.png`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.

**Docker**: You can run the above curated image example using Docker as follows:

```.bash
# Build the stylegan3:latest image
docker build --tag stylegan3 .

# Run the gen_images.py script using Docker:
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    stylegan3 \
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \
         --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```

Note: The Docker image requires NVIDIA driver release `r470` or later.

The `docker run` invocation may look daunting, so let's unpack its contents here:

- `--gpus all -it --rm --user $(id -u):$(id -g)`: with all GPUs enabled, run an interactive session with current user's UID/GID to avoid Docker writing files as root.
- ``-v `pwd`:/scratch --workdir /scratch``: mount current running dir (e.g., the top of this git repo on your host machine) to `/scratch` in the container and use that as the current working dir.
- `-e HOME=/scratch`: let PyTorch and StyleGAN3 code know where to cache temporary files such as pre-trained models and custom PyTorch extension build results. Note: if you want more fine-grained control, you can instead set `TORCH_EXTENSIONS_DIR` (for custom extensions build dir) and `DNNLIB_CACHE_DIR` (for pre-trained model download cache). You want these cache dirs to reside on persistent volumes so that their contents are retained across multiple `docker run` invocations.

## Interactive visualization

This release contains an interactive model visualization tool that can be used to explore various characteristics of a trained model.  To start it, run:

```.bash
python visualizer.py
```

<a href="./docs/visualizer_screen0.png"><img alt="Visualizer screenshot" src="./docs/visualizer_screen0_half.png"></img></a>

Pass the path or URL of a network pickle to open it directly: `python visualizer.py ~/models/network-snapshot-001000.pkl`.
The visualizer works on Linux (X11 and Wayland), Windows and WSL2 (WSLg, where it automatically uses the GPU through Mesa's
D3D12 driver instead of software rendering). "Separate process" in the performance panel renders the images in a second
process, which keeps the interface responsive with large networks.

## Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```.python
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation
```

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle contains three networks. `'G'` and `'D'` are instantaneous snapshots taken during training, and `'G_ema'` represents a moving average of the generator weights over several training steps. The networks are regular instances of `torch.nn.Module`, with all of their parameters and buffers placed on the CPU at import and gradient computation disabled by default.

The generator consists of two submodules, `G.mapping` and `G.synthesis`, that can be executed separately. They also support various additional options:

```.python
w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
img = G.synthesis(w, noise_mode='const', force_fp32=True)
```

Please refer to [`gen_images.py`](./gen_images.py) for complete code example.

## Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels. Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information. Alternatively, the folder can also be used directly as a dataset, without running it through `dataset_tool.py` first, but doing so may lead to suboptimal performance.

**FFHQ**: Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as 1024x1024 images and create a zip archive using `dataset_tool.py`:

```.bash
# Original 1024x1024 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-1024x1024.zip

# Scaled down 256x256 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-256x256.zip \
    --resolution=256x256
```

See the [FFHQ README](https://github.com/NVlabs/ffhq-dataset) for information on how to obtain the unaligned FFHQ dataset images. Use the same steps as above to create a ZIP archive for training and validation.

**MetFaces**: Download the [MetFaces dataset](https://github.com/NVlabs/metfaces-dataset) and create a ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/metfaces/images --dest=~/datasets/metfaces-1024x1024.zip
```

See the [MetFaces README](https://github.com/NVlabs/metfaces-dataset) for information on how to obtain the unaligned MetFaces dataset images. Use the same steps as above to create a ZIP archive for training and validation.

**AFHQv2**: Download the [AFHQv2 dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) and create a ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/afhqv2 --dest=~/datasets/afhqv2-512x512.zip
```

Note that the above command creates a single combined dataset using all images of all three classes (cats, dogs, and wild animals), matching the setup used in the StyleGAN3 paper. Alternatively, you can also create a separate dataset for each class:

```.bash
python dataset_tool.py --source=~/downloads/afhqv2/train/cat --dest=~/datasets/afhqv2cat-512x512.zip
python dataset_tool.py --source=~/downloads/afhqv2/train/dog --dest=~/datasets/afhqv2dog-512x512.zip
python dataset_tool.py --source=~/downloads/afhqv2/train/wild --dest=~/datasets/afhqv2wild-512x512.zip
```

## Training

You can train new networks using `train.py`. For example:

```.bash
# Train StyleGAN3-T for AFHQv2 using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \
    --gpus=8 --batch=32 --gamma=8.2 --mirror=1

# Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
    --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

# Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
```

Note that the result quality and training time depend heavily on the exact set of options. The most important ones (`--gpus`, `--batch`, and `--gamma`) must be specified explicitly, and they should be selected with care. See [`python train.py --help`](./docs/train-help.txt) for the full list of options and [Training configurations](./docs/configs.md) for general guidelines &amp; recommendations, along with the expected training speed &amp; memory usage in different scenarios.

### Training on a single GPU

The configurations of the paper use 8 GPUs, but `--gpus` does not change the training dynamics: on one GPU, the same
`--batch` is processed in several passes of `--batch-gpu` images with gradient accumulation. For example, on one 16 GB GPU:

```.bash
# StyleGAN3-T at 256x256, ~65 s/kimg on an RTX 4080 SUPER, 5.6 GB of GPU memory.
python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/mydata-256x256.zip \
    --gpus=1 --batch=16 --batch-gpu=8 --gamma=2 --mirror=1 --snap=10
```

* `--batch-gpu=8` instead of 16 is nearly as fast (66 vs 65 s/kimg above) with half the memory. On Windows and WSL2, running
  close to the memory limit does not fail but slows training down a lot (the driver moves memory to the system RAM),
  so leave some headroom.
* A smaller `--batch` usually needs a larger `--gamma` and/or a lower `--dlr`, see [Training configurations](./docs/configs.md).
* Without the custom CUDA kernels (see [Requirements](#requirements)) training is about 4x slower and needs more memory.

### Relativistic pairing loss (experimental)

`--loss=rpgan` replaces the non-saturating logistic loss with the relativistic pairing loss of
[R3GAN](https://arxiv.org/abs/2501.05441) ("The GAN is dead; long live the GAN!", NeurIPS 2024): every generated image is
scored relative to a real image of the minibatch, and the zero-centered gradient penalty is applied to both the real (R1)
and the generated (R2) images, with the same `--gamma`. The paper proves local convergence for this loss and reports less
mode dropping. Only the loss is taken from R3GAN, the architectures stay those of StyleGAN. It has not been validated on
long training runs with these architectures yet - compare it against the default loss on your data.

The results of each training run are saved to a newly created directory, for example `~/training-runs/00000-stylegan3-t-afhqv2-512x512-gpus8-batch32-gamma8.2`. The training loop exports network pickles (`network-snapshot-<KIMG>.pkl`) and random image grids (`fakes<KIMG>.png`) at regular intervals (controlled by `--snap`). For each exported pickle, it evaluates FID (controlled by `--metrics`) and logs the result in `metric-fid50k_full.jsonl`. It also records various statistics in `training_stats.jsonl`, as well as `*.tfevents` if TensorBoard is installed.

## Quality metrics

By default, `train.py` automatically computes FID for each network pickle exported during training. We recommend inspecting `metric-fid50k_full.jsonl` (or TensorBoard) at regular intervals to monitor the training progress. When desired, the automatic computation can be disabled with `--metrics=none` to speed up the training slightly.

Additional quality metrics can also be computed after the training:

```.bash
# Previous training run: look up options automatically, save result to JSONL file.
python calc_metrics.py --metrics=eqt50k_int,eqr50k \
    --network=~/training-runs/00000-stylegan3-r-mydataset/network-snapshot-000000.pkl

# Pre-trained network pickle: specify dataset explicitly, print result to stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq-1024x1024.zip --mirror=1 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl
```

The first example looks up the training configuration and performs the same operation as if `--metrics=eqt50k_int,eqr50k` had been specified during training. The second example downloads a pre-trained network pickle, in which case the values of `--data` and `--mirror` must be specified explicitly.

Note that the metrics can be quite expensive to compute (up to 1h), and many of them have an additional one-off cost for each new dataset (up to 30min). Also note that the evaluation is done using a different random seed each time, so the results will vary if the same metric is computed multiple times.

Recommended metrics:
* `fid50k_full`: Fr&eacute;chet inception distance<sup>[1]</sup> against the full dataset.
* `kid50k_full`: Kernel inception distance<sup>[2]</sup> against the full dataset.
* `pr50k3_full`: Precision and recall<sup>[3]</sup> againt the full dataset.
* `ppl2_wend`: Perceptual path length<sup>[4]</sup> in W, endpoints, full image.
* `eqt50k_int`: Equivariance<sup>[5]</sup> w.r.t. integer translation (EQ-T).
* `eqt50k_frac`: Equivariance w.r.t. fractional translation (EQ-T<sub>frac</sub>).
* `eqr50k`: Equivariance w.r.t. rotation (EQ-R).
* `fd_dinov2_50k_full`: Fr&eacute;chet distance in the feature space of DINOv2 ViT-L/14<sup>[7]</sup> against the full dataset (`fd_dinov2_10k_full`: with 10k generated images, faster). Agrees much better with human judgments of image quality than FID, which relies on Inception-v3 features. The values are not comparable to FID. Downloads DINOv2 through `torch.hub` on first use (~1.2 GB).

Legacy metrics:
* `fid50k`: Fr&eacute;chet inception distance against 50k real images.
* `kid50k`: Kernel inception distance against 50k real images.
* `pr50k3`: Precision and recall against 50k real images.
* `is50k`: Inception score<sup>[6]</sup> for CIFAR-10.

References:
1. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), Heusel et al. 2017
2. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), Bi&nacute;kowski et al. 2018
3. [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991), Kynk&auml;&auml;nniemi et al. 2019
4. [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), Karras et al. 2018
5. [Alias-Free Generative Adversarial Networks](https://nvlabs.github.io/stylegan3), Karras et al. 2021
6. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), Salimans et al. 2016
7. [Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models](https://arxiv.org/abs/2306.04675), Stein et al. 2023

## Spectral analysis

The easiest way to inspect the spectral properties of a given generator is to use the built-in FFT mode in `visualizer.py`. In addition, you can visualize average 2D power spectra (Appendix A, Figure 15) as follows:

```.bash
# Calculate dataset mean and std, needed in subsequent steps.
python avg_spectra.py stats --source=~/datasets/ffhq-1024x1024.zip

# Calculate average spectrum for the training data.
python avg_spectra.py calc --source=~/datasets/ffhq-1024x1024.zip \
    --dest=tmp/training-data.npz --mean=112.684 --std=69.509

# Calculate average spectrum for a pre-trained generator.
python avg_spectra.py calc \
    --source=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl \
    --dest=tmp/stylegan3-r.npz --mean=112.684 --std=69.509 --num=70000

# Display results.
python avg_spectra.py heatmap tmp/training-data.npz
python avg_spectra.py heatmap tmp/stylegan3-r.npz
python avg_spectra.py slices tmp/training-data.npz tmp/stylegan3-r.npz
```

<a href="./docs/avg_spectra_screen0.png"><img alt="Average spectra screenshot" src="./docs/avg_spectra_screen0_half.png"></img></a>

## License

Copyright &copy; 2021, NVIDIA Corporation & affiliates. All rights reserved.

This work is made available under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

## Citation

```
@inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik H\"ark\"onen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
}
```

## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Acknowledgements

We thank David Luebke, Ming-Yu Liu, Koki Nagano, Tuomas Kynk&auml;&auml;nniemi, and Timo Viitanen for reviewing early drafts and helpful suggestions. Fr&eacute;do Durand for early discussions. Tero Kuosmanen for maintaining our compute infrastructure. AFHQ authors for an updated version of their dataset. Getty Images for the training images in the Beaches dataset. We did not receive external funding or additional revenues for this project.

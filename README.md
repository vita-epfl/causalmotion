# Causal Motion Forecasting

This is an official implementation for the paper

**[Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective](https://arxiv.org/abs/2111.14820)**
<br>
<a href="https://sites.google.com/view/yuejiangliu">Yuejiang Liu</a>,
<a href="https://www.riccardocadei.com">Riccardo Cadei</a>,
<a href="https://people.epfl.ch/jonas.schweizer/?lang=en">Jonas Schweizer</a>,
<a href="https://www.linkedin.com/in/sherwin-bahmani-a2b5691a9">Sherwin Bahmani</a>,
<a href="https://people.epfl.ch/alexandre.alahi/?lang=en/">Alexandre Alahi</a>
<br>

TL;DR: incorporate causal invariance and structure into the design and training of motion forecasting models
* causal formalism of motion forecasting with three groups of latent variables
* causal (invariant) representations to suppress spurious features and promote robust generalization
* causal (modular) structure to approximate a sparse causal graph and facilitate efficient adaptation

<p align="left">
  <img src="docs/thumbnail.png" width="800">
</p>

*The code will be further cleaned and structured by January 2022.*

### Spurious Shifts

Please check out the code in the [spurious](spurious) folder.

### Style Shifts

Please check out the code in the [style](style) folder.

### Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{liu2021causalmotion,
  title={Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective},
  author={Liu, Yuejiang and Cadei, Riccardo and Schweizer, Jonas and Bahmani, Sherwin and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2111.14820},
  year={2021}
}
```

### Developers

Our code is mainly developed by [Riccardo Cadei](https://www.riccardocadei.com) and [Jonas Schweizer](https://people.epfl.ch/jonas.schweizer/?lang=en).

### Acknowledgements

Our code is built upon the public code of the following papers:
* [Human Trajectory Prediction via Counterfactual Analysis](https://github.com/CHENGY12/CausalHTP)
* [STGAT: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction](https://github.com/huang-xx/STGAT)
* [It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction](https://github.com/HarshayuGirase/Human-Path-Prediction)



# HyperAgent [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fszrlee%2FHyperAgent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

Author: [Yingru Li](https://richardli.xyz), [Jiawei Xu](https://github.com/jiawei415), [Zhi-Quan Luo](https://en.wikipedia.org/wiki/Zhi-Quan_Tom_Luo)

Welcome to the official implementation of **Ensemble++**. This repository accompanies our paper [Scalable Exploration via Ensemble++](https://arxiv.org/abs/2407.13195).

## Key Features

- **Fast Incremental Uncertainty Estimation:** Ensures quick updates and reliable uncertainty quantification with logarithmic per-step computational complexity.
- **Scalable Exploration:** Efficiently handles large state-action spaces, facilitating robust and adaptive exploration while matching the regret order of exact Thompson sampling.
- **Integration with GPT Models:** Utilizes the strengths of GPT architectures to enhance decision-making processes in contextual bandits with natural language input.

## Getting Started 

We welcome contributions and feedback from the community to help improve and expand the capabilities of Ensemble++.

### Nonlinear Synthetic Bandits
To run experiments on nonlinear synthetic bandits, use the script [`start_synthetic.sh`](scripts/start_synthetic.sh).

### UCI Dataset Bandits
To run experiments on UCI dataset bandits, use the script [`start_realdata.sh`](scripts/start_realdata.sh).

### Hatespeech Bandits with GPT-2
To run experiments on hatespeech bandits with GPT-2, use the script [`start_llm.sh`](scripts/start_llm.sh).


## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@misc{li2024ensemble++,
      title={Scalable Exploration via Ensemble++}, 
      author={Li, Yingru and Xu, Jiawei and Wang, Baoxiang and Luo, Zhi-Quan},
      year={2024},
      eprint={2407.13195},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.13195},
      note  = {Preprint. Presentation at ICML 2024 Workshops: (1) "Aligning Reinforcement Learning Experimentalists and Theorists"; (2) "Automated Reinforcement Learning: Exploring Meta-Learning, AutoML, and LLMs"},
}
```

```bibtex
@inproceedings{li2024hyperagent,
  title         = {{Q-Star Meets Scalable Posterior Sampling: Bridging Theory and Practice via HyperAgent}},
  author        = {Li, Yingru and Xu, Jiawei and Han, Lei and Luo, Zhi-Quan},
  booktitle     = {Forty-first International Conference on Machine Learning},
  year          = {2024},
  series        = {Proceedings of Machine Learning Research},
  eprint        = {2402.10228},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG}，
  url           = {https://arxiv.org/abs/2402.10228}
}
```

> For large-scale deep RL benchmarking results and details, visit the [szrlee/HyperAgent](https://github.com/szrlee/HyperAgent) repository.
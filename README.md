# Hybrid reinforcement learning with expert state sequences

## About this repository
This repository contains an implementation of the paper [Hybrid Reinforcement Learning with Expert State Sequences](https://aaai.org/Conferences/AAAI-19/wp-content/uploads/2018/11/AAAI-19_Accepted_Papers.pdf). The implementation is built directly on top of [PyTorch implementation of Advantage Actor Critic (A2C)](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr). 


## Dependencies
To get started with the framework, install the following dependencies:
- Python 3.6
- [PyTorch 0.4](https://pytorch.org/get-started/previous-versions/)
- [OpenAI baselines](https://github.com/openai/baselines)


## Training and model configuration 
A2C baseline (A2C agent in the paper): 
```
python main.py --policy-coef 1 --entropy-coef 0.5 --value-loss-coef 0.2 --dual-act-coef 0 --dual-state-coef 0 --dual-sup-coef 0 --dual-emb-coef 0 --log-dir baseline_a2c 
```

Hybrid agent combining A2C and behavior cloning from observation with the proposed action inference:
```
python main.py --policy-coef 1 --entropy-coef 0.5 --value-loss-coef 0.2 --dual-act-coef 2 --dual-sup-coef 2 --dual-emb-coef 0.1 --dual-rank 2 --dual-emb-dim 128 --dual-type dual --log-dir hybrid_dual 
```

Hybrid agent combining A2C and behavior cloning from observation with the MLP-based action inference (Hybrid-MLP agent in the paper):
```
python main.py --policy-coef 1 --entropy-coef 0.5 --value-loss-coef 0.2 --dual-act-coef 2 --dual-sup-coef 2 --dual-emb-coef 0.1 --dual-rank 2 --dual-emb-dim 128 --dual-type mlp --log-dir hybrid_mlp 
```
(dual-rank is used as the number of MLP layers for MLP type action inference model.)


Behavior cloning from observation with the proposed action inference (BC-Dual agent in the paper):
```
python main.py --policy-coef 0 --entropy-coef 0.5 --value-loss-coef 0 --dual-act-coef 2 --dual-sup-coef 2 --dual-emb-coef 0.1 --dual-rank 2 --dual-emb-dim 128 --dual-type dual --log-dir dual_only 
```

Behavior cloning from observation with the MLP-based action inference model (BC-MLP agent in the paper):
```
python main.py --policy-coef 0 --entropy-coef 0.5 --value-loss-coef 0 --dual-act-coef 2 --dual-sup-coef 2 --dual-emb-coef 0.1 --dual-rank 2 --dual-emb-dim 128 --dual-type mlp --log-dir mlp_only 
```

The noise in the expert demonstration can be controlled using arguments ``--demo-eps`` (the non-optimal action ratio) and ``--demo-eta`` (the missing state ratio).


## License
MIT License
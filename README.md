# lux ai competition 
Off policy reinforcement learning approach for Lux ai competition Kaggle
For the details of the competition, please check this page -> [Lux ai](https://www.kaggle.com/c/lux-ai-2021/overview)

# Overview
I tried a off policy reinforcement learning, V-trace/UPGO.
Training from random weight, the score of my agent is ~1200, but
 I think adding more training or other simple modification would improve the score.
Because remaining 1 or half day left when I confirmed the clear convergence of my rl algorithm...

![model](./image/pipeline.png)

* model: based on [image caption](https://arxiv.org/abs/1707.07998), task, encoder and lstm decoder architecture.
* multi agent action space: using auto regressive policy, [arxiv](https://arxiv.org/abs/1708.04782)

$$
\pi_{\theta}(a|s) = \prod_{i}^{N}\pi_{\theta}(a_{u_{i}}|a_{u_{<i}},s), \hspace{0.5cm} s:{\rm state},\  \ a_{u_{i}}: {\rm unit_i\ action}
$$

* Extending handyrl : Extending [handyrl](https://github.com/DeNA/HandyRL) to auto regressive multi agent policy from single agent policy.
Specifically all tensors are replaced like this, (episodes, steps, players, actions) -> (episodes, steps, players, units, actions) and loss is modified for auto regressive style.

* Environment: [Gym environment](https://github.com/glmcdona/LuxPythonEnvGym), easy to use and really fast. But there might be some minor game logic differences from the original one. I fixed it with using 2000 episode replay files, after fixing these points, I confirmed the perfect replication of the original game logic. I will check carefully it again and make a pull request on the gym env repo if it is true.

* Decoder input: 
    * making unique unit order: beginning from min y unit and then find the nearest neighbor unit repeatedly.
    * (x, y, cargo_space, fuel) for each unit. This 4 dim is converted into 32 dim through a linear layer.
    * unlike image cation model, we know the exact place of each unit that's why I directly use the unit position"s feature of the encoder, instead of the pooled feature.
* UPGO is much faster for convergence than V-trace.
* kaggle notebook: [here](), you can check my agent of 1035 epoch(500,000 episodes).
* code/github: I shared it [here](https://github.com/Fkaneko/kaggle_lux_ai).

This is my first reinforcement learning project and there might be some misunderstandings. Please be free to ask me if you find something weird.

# How to run
## environment
* Ubuntu 18.04
* Python with Anaconda/Mamba
* NVIDIA GPUx1

## Code&Package Installation
```bash
# clone project
$PROJECT=kaggle_lux_ai
$CONDA_NAME=lux_ai
git clone https://github.com/Fkaneko/$PROJECT

# install project
cd $PROJECT
conda create -n $CONDA_NAME python=3.7
conda activate $CONDA_NAME
yes | bash install.sh
 ```
## Training
Simply run followings
```python
python handyrl_main.py -t   # for reinforcement learning from random weight
```
Please check the `src/config/handyrl_config.yaml` for the default training configuration.
Also need to check the original [handyrl documentation](https://github.com/DeNA/HandyRL).

## Testing
Please check your agent on kaggle like [this one].
<!-- ```python -->
<!-- python test_agents.py -->
<!-- ``` -->
<!-- Testing enviroment is done with gym env - stable baseline agent class. -->
<!-- Please check these class for run match. -->



# License
## Dependent repositories
* LuxPythonEnvGym: MIT, https://github.com/glmcdona/LuxPythonEnvGym
* HandyRL:MIT, https://github.com/DeNA/HandyRL

##  Code
Apache 2.0

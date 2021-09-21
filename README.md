# Meta-Gradient Reinforcement Learning
Implementation of NeurIPS 2018 paper "Meta-Gradient Reinforcement Learning" <a href="https://proceedings.neurips.cc/paper/2018/file/2715518c875999308842e3455eda2fe3-Paper.pdf">Link</a>

We implement A2C as the base algorithm under CartPole-v1 environment and consider γ (i.e. the discounting factor) as the only hyperparameter to be meta-learned/meta-adapted.

The maximum reward of the CartPole-v1 environment is 500. 
We constrain the number of training steps to be `500` due to computation and time limits. For each run, the training is terminated as soon as the agent obtains rewards exceeding than `475` (`475` is the reward threshold specified by the CartPole-v1 environment). 

We use neural networks to implement the actor and the value network, both of which comprise three fully connected layers with a hidden size of 40 and with ReLU as activation function. Log-softmax operation is used to normalize the sum of the output of the actor network to one. The table below is the hyper-parameters used in our experiment.

The hyperparameter we use are as follows:
| hyperparameter |           |
|----------------------------|----------------------|
| optimizer                  | Adam                 |
| learning rate              | 0.01                 |
| sample size                | 30                   |
| initial gamma              | 0.99                 |
| alpha                      | 0.0001               |

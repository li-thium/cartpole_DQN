# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #
import numpy as np
import gym
import torch.nn as nn
import torch
import torch.nn.functional as F

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool


class Net(nn.Module):
    # 构造只有一个隐含层的网络
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, 2)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #

class dqn:
    # （1）初始化
    def __init__(self, n_states, n_hidden1, n_hidden2, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden1 = n_hidden1  # 隐含层个数
        self.n_hidden2 = n_hidden2  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net()
        # 实例化目标网络
        self.target_q_net = Net()

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    # （3）网络训练
    def update(self, transition_dict):  # 传入经验池中的batch个样本
        # 获取当前时刻的状态 array_shape=[b,4]
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        # 下一时刻的状态 array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        # 根据actions索引在训练网络的输出的第1维度上获取对应索引的q值（state_value）
        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())

        self.count += 1

    def take_action(self, state):  # 使用epsilon-贪婪策略选取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()

        return action

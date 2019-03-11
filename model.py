import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import get_distribution
from utils import init_normc_
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def zero_bias_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.bias.data.fill_(0)

            
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, dual_type, dual_rank=None, dual_emb_dim=None):
        super(Policy, self).__init__()
        self.base = EmbBase(obs_shape[0], action_space)
        if dual_type == 'dual':
            self.dual_model = DualModel(num_states=obs_shape[0] + 1, num_actions=action_space.n,
                                        rank=dual_rank, emb_dim = dual_emb_dim)
        else:
            self.dual_model = DualBaseline(num_states=obs_shape[0] + 1, num_actions=action_space.n,
                                           emb_dim=dual_emb_dim, num_layer=dual_rank)
        self.state_size = self.base.state_size

    def forward(self, inputs, states, masks):
        return self.base(inputs, states, masks)

    def act(self, inputs, states, masks, deterministic=False):
        value, hidden_actor, states = self(inputs, states, masks)
        
        action = self.base.dist.sample(hidden_actor, deterministic=deterministic)

        action_log_probs, dist_entropy, all_log_probs = self.base.dist.logprobs_and_entropy(hidden_actor, action)
        
        return value, action, action_log_probs, states

    def dual_predict(self, states, actions, next_states):
        return self.dual_model(states, actions, next_states)

    def get_value(self, inputs, states, masks):        
        value, _, _ = self(inputs, states, masks)
        return value
    
    def evaluate_actions(self, inputs, states, masks, actions):
        value, hidden_actor, states = self(inputs, states, masks)
        action_log_probs, dist_entropy, all_log_probs = self.base.dist.logprobs_and_entropy(hidden_actor, actions)
        return value, action_log_probs, dist_entropy, states, all_log_probs

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init_normc_(m.weight.data)

class EmbBase(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(EmbBase, self).__init__()
        emb_dim = 500
        self.action_space = action_space
        emb0 = nn.Embedding(num_inputs, emb_dim)
        self.actor = nn.Sequential(
            emb0,
            # nn.Sigmoid(),
            # nn.Linear(emb_dim, emb_dim),
            # nn.Tanh()
        )
        emb1 = nn.Embedding(num_inputs, emb_dim)
        self.critic = nn.Sequential(
            emb1,
            # nn.Sigmoid(),
            # nn.Linear(emb_dim, emb_dim),
            # nn.Tanh()
        )

        self.critic_linear = nn.Linear(emb_dim, 1)
        self.dist = get_distribution(emb_dim, action_space)

        self.train()
        self.reset_parameters()
        emb0.weight.data = torch.eye(emb_dim)
        emb1.weight.data = torch.eye(emb_dim)

    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init_mlp)
        self.apply(zero_bias_init)
        
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)
    
    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor, states

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class DualModel(nn.Module):
    def __init__(self, num_states, num_actions, emb_dim=128, rank=16):
        super(DualModel, self).__init__()
        self.emb_dim = emb_dim
        self.rank = rank
        self.gamma = 1
        print("Dual Model: rank={}, emb_dim={}".format(self.rank, self.emb_dim))
        emb0 = nn.Embedding(num_states, self.emb_dim)
        self.state_emb = nn.Sequential(
            emb0
        )
        emb1 = nn.Embedding(num_states, self.emb_dim)
        self.action_emb = nn.Sequential(
            emb1
        )

        m_components = []
        n_components = []
        for i in range(self.rank):
            m_components.append(nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim,bias=False))
            n_components.append(nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=False))
        self.m_components = ListModule(*m_components)
        self.n_components = ListModule(*n_components)
        self.action_out_linear = nn.Linear(in_features=self.emb_dim, out_features=num_actions, bias=True)
        self.state_out_linear = nn.Linear(in_features=self.emb_dim, out_features=num_states, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

    def forward_states(self, states, actions):
        st_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        sum = 0
        for i in range(self.rank):
            m_cmp = self.m_components[i](act_emb)
            n_cmp = self.n_components[i](st_emb)
            mn = m_cmp * n_cmp
            sum += mn
        return F.log_softmax(self.state_out_linear(self.gamma * st_emb + sum), dim=1), sum

    def forward_actions(self, states, next_states):
        st_emb = self.state_emb(states)
        next_emb = self.state_emb(next_states)
        delta = next_emb - self.gamma * st_emb
        sum = 0
        for i in range(self.rank):
            m_cmp = self.m_components[i](delta)
            n_cmp = self.n_components[i](st_emb)
            mn = m_cmp * n_cmp
            sum += mn
        return F.log_softmax(self.action_out_linear(sum), dim=1), sum

    def forward(self, states, actions, next_states):
        st_out, delta_st = self.forward_states(states, actions)
        act_out, act_emb = self.forward_actions(states, next_states)
        delta_st_label = self.state_emb(next_states) - self.state_emb(states)
        act_emb_label = self.action_emb(actions)
        emb_loss = (act_emb - act_emb_label.detach()).abs().mean() +\
                   (delta_st - delta_st_label.detach()).abs().mean()
        return st_out, act_out, emb_loss



class DualBaseline(nn.Module):
    def __init__(self, num_states, num_actions, emb_dim=128, num_layer=1):
        super(DualBaseline, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        # self.rank = 16
        # self.gamma = 1
        emb0 = nn.Embedding(num_states, self.emb_dim)
        self.state_emb = nn.Sequential(
            emb0
        )
        emb1 = nn.Embedding(num_states, self.emb_dim)
        self.action_emb = nn.Sequential(
            emb1
        )
        emb2 = nn.Embedding(num_states, self.emb_dim)
        self.next_state_emb = nn.Sequential(
            emb2
        )

        self.s_linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim)
        self.a_linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim)
        self.ns_linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim)
        action_linear = []
        next_state_linear = []
        for i in range(self.num_layer):
            action_linear.append(nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim))
            next_state_linear.append(nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim))
        self.action_linear = ListModule(*action_linear)
        self.next_state_linear = ListModule(*next_state_linear)
        self.action_out_linear = nn.Linear(in_features=self.emb_dim, out_features=num_actions, bias=True)
        self.state_out_linear = nn.Linear(in_features=self.emb_dim, out_features=num_states, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

    def forward_states(self, states, actions):
        st_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        s_a_emb = nn.ReLU()(self.s_linear(st_emb) + self.a_linear(act_emb))
        for i in range(self.num_layer):
            s_a_emb = nn.ReLU()(self.next_state_linear[i](s_a_emb))
        return F.log_softmax(self.state_out_linear(s_a_emb), dim=1), s_a_emb

    def forward_actions(self, states, next_states):
        st_emb = self.state_emb(states)
        next_emb = self.next_state_emb(next_states)
        s_s_emb = nn.ReLU()(self.s_linear(st_emb) + self.ns_linear(next_emb))
        for i in range(self.num_layer):
            s_s_emb = nn.ReLU()(self.action_linear[i](s_s_emb))
        return F.log_softmax(self.action_out_linear(s_s_emb), dim=1), s_s_emb

    def forward(self, states, actions, next_states):
        st_out, _ = self.forward_states(states, actions)
        act_out, _ = self.forward_actions(states, next_states)
        return st_out, act_out, Variable(torch.Tensor(1).fill_(0)).to(device)


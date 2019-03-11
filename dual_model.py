import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init_normc_
from model import DualModel

def zero_bias_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.bias.data.fill_(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data)

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init_normc_(m.weight.data)


class DualBaseline(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DualBaseline, self).__init__()
        self.emb_dim = 128
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
        self.action_out_linear = nn.Linear(in_features=self.emb_dim, out_features=num_actions, bias=True)
        self.state_out_linear = nn.Linear(in_features=self.emb_dim, out_features=num_states, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

    def forward_states(self, states, actions):
        st_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        s_a_emb = nn.Tanh()(self.s_linear(st_emb) + self.a_linear(act_emb))
        return F.log_softmax(self.state_out_linear(s_a_emb), dim=1)

    def forward_actions(self, states, next_states):
        st_emb = self.state_emb(states)
        next_emb = self.next_state_emb(next_states)
        s_s_emb = nn.Tanh()(self.s_linear(st_emb) + self.ns_linear(next_emb))
        return F.log_softmax(self.action_out_linear(s_s_emb), dim=1)

    def forward(self, states, actions, next_states):
        st_out = self.forward_states(states, actions)
        act_out = self.forward_actions(states, next_states)
        return st_out, act_out



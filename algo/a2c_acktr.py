import torch
import torch.nn as nn
import torch.optim as optim
from policies import optimal_policy
from policies import get_next_state
from policies import generate_fixed_data_set
from .kfac import KFACOptimizer
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_optimal_actions(states):
    idx = states.data.cpu().numpy()
    actions = []
    for id in idx:
        actions.append(optimal_policy(id))
    return torch.from_numpy(np.array(actions)).view(-1,1)

def get_optimal_actions_from_fixed_data_set(states, demo_cur_state):
    idx = states.data.cpu().numpy()
    actions = []
    for id in idx:
        if id in demo_cur_state:
            actions.append(optimal_policy(id))
    return torch.from_numpy(np.array(actions)).view(-1,1)


def get_demo_states(states):
    idx = states.data.cpu().numpy()
    next_state = []
    for id in idx:
        act = optimal_policy(id)
        next_state.append(get_next_state(id, act))

    return states, torch.from_numpy(np.array(next_state)).view(-1)


def get_demo_states_from_fixed_data_set(states, demo_cur_state, demo_next_state):
    idx = states.data.cpu().numpy()
    next_state = []
    cur_state = []
    for id in idx:
        if id in demo_cur_state:
            index = demo_cur_state.index(id)
            cur_state.append(demo_cur_state[index])
            next_state.append(demo_next_state[index])

    return torch.from_numpy(np.array(cur_state)).view(-1), torch.from_numpy(np.array(next_state)).view(-1)

def get_fixed_demo_data_set(eta = 0, eps = 0):
    return generate_fixed_data_set(eta, eps)


def select_states(cur_states, demo_cur_state, all_log_probs):
    valid_log_probs = []
    idx = cur_states.data.cpu().numpy()
    for id in range(len(idx)):
        if idx[id] in demo_cur_state:
            valid_log_probs.append(id)
    valid_log_probs = torch.from_numpy(np.array(valid_log_probs)).view(-1)
    return all_log_probs.index_select(0, valid_log_probs.long().to(device))


def check_consistency(observations, actions):
    print('observation:', observations.size())
    print('actions:', actions.size())
    states = observations[:-1].data.cpu().numpy()
    next_states = observations[1:].data.cpu().numpy()
    actions = actions.data.cpu().numpy()
    # print(states)
    # print(actions)
    # cur_states = states[1:]

    for i in range(observations.size(1)):
        cur = states[0][i]
        act = actions[0][i]
        next = get_next_state(cur, act)
        print(cur, ',', act, '->', next, '|', next_states[0][i])
    exit(172)
    return

class A2C_ACKTR(object):
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 dual_act_coef,
                 dual_state_coef,
                 dual_sup_coef,
                 policy_coef,
                 emb_coef,
                 demo_eta,
                 demo_eps,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.dual_act_coef = dual_act_coef
        self.dual_state_coef = dual_state_coef
        self.dual_sup_coef = dual_sup_coef
        self.policy_coef = policy_coef
        self.emb_coef = emb_coef
        self.demo_eta = demo_eta
        self.demo_eps = demo_eps

        self.max_grad_norm = max_grad_norm

        demo_cur_state, demo_opt_act, demo_next_state = generate_fixed_data_set(self.demo_eta, self.demo_eps)
        self.demo_cur_state = demo_cur_state
        self.demo_opt_act = demo_opt_act
        self.demo_next_state = demo_next_state
        print("a2c: p={:.1f}, e={:.1f}, v={:.1f}, da={:.1f}, ds={:.1f}, s={:.1f}, emb={:.3f}, eta={:.2f}, eps={:.2f} ".format(
            self.policy_coef, self.entropy_coef, self.value_loss_coef,
            self.dual_act_coef, self.dual_state_coef, self.dual_sup_coef, self.emb_coef, self.demo_eta, self.demo_eps))


        self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        # a2c policy and value
        values, action_log_probs, dist_entropy, states, all_log_probs = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.states[0].view(-1, self.actor_critic.state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        # dual loss:
        # get states, actions, next_stats in the agent's behavior
        cur_states = rollouts.observations[:-1].view(-1)
        next_states = rollouts.observations[1:].view(-1)
        actions = rollouts.actions.view(-1)
        log_states, log_actions, emb_loss = self.actor_critic.dual_predict(cur_states, actions, next_states)

        dual_state_loss = -log_states.gather(1, next_states.view(-1,1)).mean()
        dual_act_loss = -log_actions.gather(1, actions.view(-1,1)).mean()
        # acc.
        _, next_states_prediction = log_states.detach().max(dim=1)
        state_acc = next_states_prediction.eq(next_states.long()).float().mean()
        _, action_prediction = log_actions.detach().max(dim=1)
        action_acc = action_prediction.eq(actions.long()).float().mean()

        # policy sup loss:
        demo_state, demo_next_state = get_demo_states_from_fixed_data_set(cur_states,
                                        self.demo_cur_state, self.demo_next_state)
        sup_loss = 0
        if demo_state.size(0) > 0:
            pred_act, _ = self.actor_critic.dual_model.forward_actions(demo_state.long().to(device), demo_next_state.long().to(device))
            _, pred_act = pred_act.detach().max(dim=1)
            valid_log_probs = select_states(cur_states, self.demo_cur_state, all_log_probs)
            sup_loss = -valid_log_probs.gather(1, pred_act.view(-1, 1)).mean()
        # policy sup acc.
        opt_actions = get_optimal_actions_from_fixed_data_set(cur_states, self.demo_cur_state).view(-1).to(device)
        sup_acc = pred_act.eq(opt_actions.long()).float().mean()

        self.optimizer.zero_grad()
        (dual_state_loss * self.dual_state_coef +
         dual_act_loss * self.dual_act_coef +
         sup_loss * self.dual_sup_coef +
         value_loss * self.value_loss_coef +
         action_loss * self.policy_coef +
         emb_loss * self.emb_coef -
         dist_entropy * self.entropy_coef).backward()
        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), \
               dual_act_loss.item(), dual_state_loss.item(), sup_loss.item(), emb_loss.item(), \
               state_acc.item(), action_acc.item(), sup_acc.item(), demo_state.size(0) / cur_states.size(0)


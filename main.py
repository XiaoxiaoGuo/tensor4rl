import copy
import os
import time
import numpy as np
import torch
import random
from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from model import Policy
from storage import RolloutStorage
from visualize import visdom_plot
from pathlib import Path

import algo

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes
torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    model_dir = Path('./models') / args.env_name / args.log_dir
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    args.log_dir = str(log_dir)
    print('saving to', args.log_dir)

    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.add_timestep)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    obs_shape = envs.observation_space.n,

    actor_critic = Policy(obs_shape, envs.action_space, args.dual_type, args.dual_rank, args.dual_emb_dim)

    if args.cuda:
        actor_critic.cuda()

    agent = algo.A2C_ACKTR(actor_critic=actor_critic, value_loss_coef=args.value_loss_coef,
                               entropy_coef=args.entropy_coef, dual_act_coef=args.dual_act_coef,
                               dual_state_coef=args.dual_state_coef, dual_sup_coef=args.dual_sup_coef,
                               policy_coef=args.policy_coef, emb_coef=args.dual_emb_coef,
                               demo_eta=args.demo_eta, demo_eps=args.demo_eps,
                               lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.base.state_size)
    current_obs = torch.zeros(args.num_processes)
    def update_current_obs(obs):
        obs = torch.from_numpy(obs).float()
        current_obs[:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step])
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()


            current_obs *= masks.squeeze(1)

            update_current_obs(obs)
            rollouts.insert(current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy, \
        dual_act_loss, dual_state_loss, dual_sup, emb_loss, \
        state_acc, action_acc, sup_acc, miss_rate = agent.update(rollouts)
        
        rollouts.after_update()

        if j % args.save_interval == 0:
            save_path = run_dir / 'incremental'
            if not save_path.exists():
                os.makedirs(str(save_path))

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, str(save_path / ("model_ep_%i.pt" % j)))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f},"
                  "\t entropy {:.3f}, v {:.3f}, p {:.3f}, d-act {:.3f}/{:.3f}, d-state {:.3f}/{:.3f}, d-sup {:.3f}/{:.3f}/{:.3f}, emb {:.3f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy,
                       value_loss, action_loss, dual_act_loss, action_acc, dual_state_loss, state_acc, dual_sup, sup_acc, miss_rate, emb_loss))
        if args.vis and j % args.vis_interval == 0:
            try:
            #Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo, args.num_frames)
            except IOError:
                pass

if __name__ == "__main__":
    main()

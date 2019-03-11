import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='a2c')
    # fixed for all
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--clip-param', type=float, default=0.5,
                        help='ppo clip parameter (default: 0.2)')

    # a2c parameters
    parser.add_argument('--policy-coef', type=float, default=1,
                        help='policy term coefficient (default: 1)')
    parser.add_argument('--entropy-coef', type=float, default=0.5,
                        help='entropy term coefficient (default: 0.5)')
    parser.add_argument('--value-loss-coef', type=float, default=0.2,
                        help='value loss coefficient (default: 0.2)')
    # action inference model parameters
    parser.add_argument('--dual-act-coef', type=float, default=2,
                        help='dual action term coefficient 2')
    parser.add_argument('--dual-state-coef', type=float, default=0,
                        help='dual state coefficient (default: 0)')
    parser.add_argument('--dual-sup-coef', type=float, default=1,
                        help='dual supervision term coefficient 1')
    parser.add_argument('--dual-emb-coef', type=float, default=0.1,
                        help='dual supervision term coefficient 0.1')
    parser.add_argument('--dual-rank', type=int, default=2,
                        help='dual rank (default: 2)')
    parser.add_argument('--dual-emb-dim', type=int, default=128,
                        help='dual embedding dim. (default: 128)')
    parser.add_argument('--dual-type', default='dual',
                        help='dual model type: dual or mlp')
    parser.add_argument('--demo-eta', type=float, default=0.0,  #
                        help='state drop rate')
    parser.add_argument('--demo-eps', type=float, default=0.0,  #
                        help='expert action randomness')
    parser.add_argument('--log-dir', default='dual_a2c_eps_0.05',
                        help='directory to save agent logs ')


    # -----------------------------------------------
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed (default: 7)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=2,
                        help='number of forward steps in A2C (default: 2)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 100)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save interval, one save per n updates (default: 1000)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=1e5,
                        help='number of frames to train (default: 1e5)')
    parser.add_argument('--env-name', default='Taxi-v2')
    # parser.add_argument('--save-dir', default='./trained_models/',
    #                     help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-vis', action='store_true', default=True,
                        help='disables visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    return args

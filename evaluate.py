import numpy as np
import torch
import gym

RANDOM_SEED = 7
CUDA = True
GAME = 'Taxi-v2'
MODEL = "models/Taxi-v2/mlp_a2c/run1/incremental/model_ep_3100.pt"

torch.manual_seed(RANDOM_SEED)
if CUDA:
    torch.cuda.manual_seed(RANDOM_SEED)


def main():
    env = gym.make(GAME)

    actor_critic, _ = torch.load(MODEL)
    print(actor_critic)

    current_obs = torch.LongTensor(1)

    def update_current_obs(obs):
        obs = torch.from_numpy(np.array([obs])).float()
        current_obs[:] = obs

    for j in range(1):
        done = False
        obs = env.reset()

        time_step = 0
        total_reward = 0
        while not done:
            # Sample actions
            update_current_obs(obs)
            with torch.no_grad():
                _, action, _, _ = actor_critic.act(current_obs, None, None, deterministic=True)
            obs, reward, done, info = env.step(action.data.numpy()[0,0])
            # get optimal policy for cur_obs
            # get next state for the optimal policy
            #
            # display dual model part and model part

            time_step += 1
            total_reward += reward
            env.render()
            print('rewards:', total_reward, '\ttime:',time_step)

if __name__ == "__main__":
    main()

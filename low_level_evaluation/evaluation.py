import numpy as np

import pprint as pp

# from PPO_agent import PPO, PPOBuffer
##from LPPO_agent import LPPO, LPPOBuffer
from PPO_agent import PPO

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from gymnasium.wrappers import RecordVideo

def to_one_hot(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_one_hot = np.zeros((y.shape[0], num_columns))
    y_one_hot[range(y.shape[0]), y] = 1.0

    return y_one_hot

def uniform_sampling(latent_cont_dim, latent_disc_dim):
    z = None
    z_cont = None
    if not latent_cont_dim == 0:
        z_cont = np.random.uniform(-1, 1, size=(1, latent_cont_dim))
        if latent_disc_dim == 0:
            z = z_cont
    if not latent_disc_dim == 0:
        z_disc = np.random.randint(0, latent_disc_dim, 1)
        z_disc = to_one_hot(z_disc, latent_disc_dim)
        if latent_cont_dim == 0:
            z = z_disc
        else:
            z = np.hstack((z_cont, z_disc))

    return z

def evaluate_greedy_ppo(env_test, agent, args, test_iter, test_n, state_dim):
    print('start evaluation')
    state_test = env_test.reset(seed=0)
    ##z = uniform_sampling(latent_cont_dim, latent_disc_dim)
    env_test.start_video_recorder()
    eps_count = 0
    while eps_count < 10:
        print('episode: ',eps_count)
        return_epi_test = 0
        for t_test in range(int(args['max_episode_len'])):
            theta = 45.0
            phi = 0.0
            r = 4
            ##print('state_test',state_test)
            action_test = agent.select_action(np.reshape(state_test, (1, state_dim)), stochastic=False)
            ##print('action : ',action_test)
            state_test2, reward_test, terminal_test, info_test = env_test.step(action_test)
            ##print('state :',state_test2)
            ##print('reward : ',reward_test)
            state_test = state_test2
            return_epi_test = return_epi_test + reward_test
            if terminal_test:
                env_test.reset(seed=0)
                eps_count += 1
                break

        ##print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(test_n),
                                                                      ##int(return_epi_test)))
        print('| test_iter: ',test_iter,' | nn: ',test_n,' | return_epi_test: ',return_epi_test,' |')
        return_epi_test = 0
    env_test.close()
            
    
def main(args):
    print('load env')
    unity_env = UnityEnvironment(args["env"],no_graphics=True,worker_id=int(args['port_offset']))
    print('wrap env')
    test_env = UnityToGymWrapper(unity_env)
    test_env.seed(0)

    env_to_record = RecordVideo(test_env,'./videos')

    np.random.seed(int(args['random_seed']) )
    test_env.seed(int(args['random_seed']))

    action_bound = float(test_env.action_space.high[0])

    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    latent_cont_dim = int(args['latent_cont_dim'])
    latent_disc_dim = int(args['latent_disc_dim'])

    ##agent = LPPO(state_dim=state_dim, action_dim=action_dim,
                     ##latent_cont_dim=latent_cont_dim, latent_disc_dim=latent_disc_dim,iw = True if args['iw'] == 1 else False)
    agent = PPO(state_dim,action_dim,ac_kwargs=dict(hidden_sizes=(128,64)))
    
    agent.load_model(args["iter"],args["seed"],args["env"])
    return_test_epi = evaluate_greedy_ppo(env_to_record,agent,args,0,0,state_dim)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument('--env-id', type=int, default=6, help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument("--latent-cont-dim", default=2, type=int)  # dimension of the continuous latent variable
    parser.add_argument("--latent-disc-dim", default=0, type=int)  # dimension of the discrete latent variable

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--total-step-num', help='total number of time steps', default=1000000)
    parser.add_argument('--eval-step-freq', help='frequency of evaluating the policy', default=5000)
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--trial-idx', help='index of trials', default=13)
    parser.add_argument('--test-num', help='number of test episodes', default=10)
    parser.add_argument('--result-file', help='file name for result file', default='/trials_lppo_')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=3)
    parser.add_argument('--model-save-freq', help='frequency of evaluating the policy', default=200000)
    parser.add_argument('--iw',type=int,default=0)

    parser.add_argument("--iter",help = "iteration number",default=11)
    parser.add_argument("--seed",help = "training seed",default=13)
    parser.add_argument('--port_offset',default=0)
    args = parser.parse_args()

    args_tmp = parser.parse_args()

    if args_tmp.env is None:
        env_dict = {00 : "Pendulum-v0",
                    1 : "InvertedPendulum-v1",
                    2 : "InvertedDoublePendulum-v1",
                    3 : "Reacher-v3",
                    4 : "Swimmer-v3",
                    5 : "Ant-v3",
                    6 : "Hopper-v3",
                    7 : "Walker2d-v3",
                    8 : "HalfCheetah-v3",
                    9 : "Humanoid-v3",
                    }
        args_tmp.env = env_dict[args_tmp.env_id]
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)
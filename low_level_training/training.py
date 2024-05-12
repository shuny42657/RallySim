import numpy as np
from torch.optim import Adam

import pprint as pp

from PPO_agent import PPO,PPOBuffer

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

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

def transform_high_level(high_level_action):
    high_level_action = np.clip(high_level_action,-1.0,1.0)
    theta = high_level_action[0] * 5 + 45
    phi = high_level_action[1] * 30
    r = high_level_action[2] * 1.5 + 3

    y = r * np.cos(np.deg2rad(theta))
    x = r * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    z = r * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))

    return np.array([x,y,z])

def evaluate_greedy_ppo(env_test, agent, args, test_iter, test_n, state_dim, latent_cont_dim, latent_disc_dim):

    state_test = env_test.reset()
    ##z = uniform_sampling(latent_cont_dim, latent_disc_dim)

    return_epi_test = 0
    for t_test in range(int(args['max_episode_len'])):
        action_test = agent.select_action(np.reshape(state_test, (1, state_dim)), stochastic=False)
        state_test2, reward_test, terminal_test, info_test = env_test.step(action_test)
        state_test = state_test2
        return_epi_test = return_epi_test + reward_test
        if terminal_test:
            env_test.reset()
            break

    print('| test_iter: ',test_iter,' | nn: ',test_n,' | return_epi_test: ',return_epi_test,' |')

    return return_epi_test


def train_ppo(env, env_test, agent, args):

    # Initialize replay memory
    total_step_cnt = 0
    epi_cnt = 0
    test_iter = 0
    return_test = np.zeros((np.ceil(int(args['total_step_num']) / int(args['eval_step_freq'])).astype('int') + 1))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = PPOBuffer(state_dim,action_dim,int(args['steps_per_epoch']))

    while total_step_cnt in range( int(args['total_step_num']) ):

        state = env.reset()
        ep_reward = 0
        ep_len = 0

        ##z = uniform_sampling(latent_cont_dim, latent_disc_dim)

        for t in range(int(args['steps_per_epoch'])):
            # Select action randomly or according to policy

            action, value, logp = agent.select_action(np.array(state), stochastic=True)
            state2, reward, terminal, info = env.step(action)

            replay_buffer.store(state,action,reward,value,logp)
            state = state2

            ep_reward += reward

            total_step_cnt += 1
            ep_len += 1
            timeout = ep_len == int(args['max_episode_len'])
            terminal = terminal or timeout
            epoch_ended = t == int(args['steps_per_epoch']) - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                if timeout or epoch_ended:
                    _, v, _ = agent.select_action(np.array(state),stochastic=True)
                else:
                    v = 0

                replay_buffer.finish_path(v)

                epi_cnt += 1
                if epi_cnt % 10 == 0:
                    print('| Reward: ',ep_reward,' | Episode: ',epi_cnt,' | Total step num: ',total_step_cnt,' |')

                state, ep_reward, ep_len = env.reset(), 0, 0
                ##z = np.random.uniform(-1, 1, size=(1, latent_dim))

            # Evaluate the deterministic policy
            if total_step_cnt >= test_iter * int(args['eval_step_freq']) or total_step_cnt == 1:
                print('total_step_cnt', total_step_cnt)
                print('evaluating the deterministic policy...')
                for test_n in range(int(args['test_num'])):
                    return_epi_test = evaluate_greedy_ppo(env_test, agent, args, test_iter, test_n, state_dim)

                    # Store the average of returns over the test episodes
                    return_test[test_iter] = return_test[test_iter] + return_epi_test / float(args['test_num'])

                print('return_test[',test_iter,'] : ',return_test[test_iter])
                test_iter += 1

            if total_step_cnt % int(args['model_save_freq']) == 0:
                agent.save_model(iter=test_iter, seed=int(args['trial_idx']), env_name=args['env'],foldername=args["checkpoint_path"])


        print('total step cnt', total_step_cnt)
        agent.update(replay_buffer)

    return return_test


def main(args):
    for ite in range(int(args['trial_num'])):
        print('Trial Number:', ite)

        print("port_offset",args["port_offset"])
        channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(args["env"],no_graphics=True,worker_id=int(args['port_offset']),side_channels=[channel])
        channel.set_configuration_parameters(time_scale = 1.0)


        env = UnityToGymWrapper(unity_env,uint8_visual=False)

        np.random.seed(int(args['random_seed']) )
        env.seed(int(args['random_seed']))

        env_test = UnityToGymWrapper(unity_env)
        env_test.seed(int(args['random_seed']))

        print('action_space.shape', env.action_space.shape)
        print('observation_space.shape', env.observation_space.shape)

        assert (env.action_space.high[0] == -env.action_space.low[0])

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = PPO(state_dim,action_dim,ac_kwargs=dict(hidden_sizes=(128,64)))
        step_R_i = train_ppo(env, env_test, agent, args)

        result_path = "./results/trials/ppo"
        result_filename = result_path + args['result_file'] + '_' + args['env'] \
                            + '_trial_idx_' + str(int(args['trial_idx'])) + '.txt'
        try:
            import pathlib
            pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
            np.savetxt(result_filename, np.asarray(step_R_i))
        except:
            print("A result directory does not exist and cannot be created. The trial results are not saved")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env',default='High_level')

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--total-step-num', help='total number of time steps', default=1000000)
    parser.add_argument('--eval-step-freq', help='frequency of evaluating the policy', default=5000)
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--trial-idx', help='index of trials', default=1)
    parser.add_argument('--test-num', help='number of test episodes', default=10)
    parser.add_argument('--result_file', help='file name for result file', default='/trials_lppo_')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=3)
    parser.add_argument('--model-save-freq', help='frequency of evaluating the policy', default=200000)
    parser.add_argument('--checkpoint_path',help='path of directory in which checkpoints are saved',default = './models/lppo')
    parser.add_argument('--port_offset',help='by how many you offset port number',default = 0)
    parser.add_argument('--info_rate',type=float,default=0.2)
    parser.add_argument('--ppo',type=int,default=0)
    args = parser.parse_args()

    args_tmp = parser.parse_args()
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)
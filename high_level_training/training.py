import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from gym.spaces import Box

import pprint as pp

from PPO_agent import PPO, PPOBuffer
##from PPO_HRL_agent import PPO,PPOBuffer

from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

# import spinup.algos.pytorch.ppo.core as core
# from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

def evaluate_greedy_high_level(env_test, high_level_agent,low_level_agent, args, test_iter, test_n, state_dim):
    ##print('evaluation start')
    states_test = env_test.reset()
    agent_to_train = env_test.agents[0]
    obs_dict = {}
    low_level_states = {}
    high_level_flags = {agent : False for agent in env_test.agents}
    high_level_actions = {agent : 0.0 for agent in env_test.agents}
    low_level_actions = {}
    ##state_test = env_test.reset()
    return_epi_test = 0
    for t_test in range(int(args['max_episode_len'])):
        for agent in env_test.agents:
            obs_dict[agent] = states_test[agent]
            low_level_states[agent] = states_test[agent][:17]
        for agent in env_test.agents:
            if obs_dict[agent][19] > 0.5 and high_level_flags[agent] == False:
                target_value = high_level_agent.select_action(np.array(obs_dict[agent]),stochastic = False)
                high_level_actions[agent] = 2 * np.clip(target_value,-1.0,1.0) + 4.0
            low_level_states[agent][16] = high_level_actions[agent] ##最後の三つの部分を置き換える。
            high_level_flags[agent] = obs_dict[agent][19] > 0.5
        low_level_actions = {agent : low_level_agent.select_action(np.reshape(low_level_states[agent], (1, 17)),stochastic=False) for agent in env_test.agents}
        for agent in env_test.agents:
            lower_bound = -5.0
            upper_bound = 5.0
            low_level_actions[agent] = np.clip(low_level_actions[agent],lower_bound,upper_bound)

        new_states_test,rewards_test,terminals_test,infos_test = env_test.step(low_level_actions)
        states_test = new_states_test
        return_epi_test += rewards_test[agent_to_train]
        
        if any([terminals_test[a] for a in terminals_test]):
            print('evaluation came to terminal state')
            env_test.reset()
            break

    ##for agent in env_test.agents:
    ##print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(test_n),
                                                                      ##int(return_epi_test)))
    print('| test_iter: ',test_iter,' | nn: ',test_n,' | return_epi_test: ',return_epi_test,' |')

    return return_epi_test ##辞書型

def evaluate_greedy_low_level(env_test, low_level_agent, args, test_iter, test_n, state_dim, latent_cont_dim, latent_disc_dim):

    states_test = env_test.reset()
    ##state_test = env_test.reset()
    agent_to_evaluate = env_test.agents[0]
    low_level_actions = {}

    z = uniform_sampling(latent_cont_dim, latent_disc_dim)
    return_epi_test = 0
    for t_test in range(int(args['max_episode_len'])):
        selected_actions = {agent : low_level_agent.select_action(np.reshape(states_test[agent], (1, state_dim)), z, stochastic=False) for agent in env_test.agents}
        for agent in env_test.agents:
                lower_bound = -1.0
                upper_bound = 1.0
                selected_actions[agent] = np.clip(selected_actions[agent],lower_bound,upper_bound)
        ##action_test = agent.select_action(np.reshape(state_test, (1, state_dim)), z, stochastic=False)
        ##state_test2, reward_test, terminal_test, info_test = env_test.step(selected_actions)
        new_states, rewards_test,terminals_test,infos_test = env_test.step(selected_actions)
        states_test = new_states
        return_epi_test = return_epi_test + rewards_test[agent_to_evaluate]
        if any([terminals_test[a] for a in terminals_test]):
            env_test.reset()
            break

    ##print('LOW_LEVEL : test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(test_n),
                                                                      ##int(return_epi_test)))
    print('| LOW_LEVEL | test_iter: ',test_iter,'| nn: ',test_n,' |  return_epi_test: ',return_epi_test,' |')
    

    return return_epi_test

def evaluate_greedy_low_level_ppo(env_test, low_level_agent, args, test_iter, test_n, state_dim):

    states_test = env_test.reset()
    agent_to_evaluate = env_test.agents[0]
    low_level_actions = {}

    ##z = uniform_sampling(latent_cont_dim, latent_disc_dim)
    return_epi_test = 0
    for t_test in range(int(args['max_episode_len'])):
        selected_actions = {agent : low_level_agent.select_action(np.reshape(states_test[agent], (1, state_dim)), stochastic=False) for agent in env_test.agents}
        for agent in env_test.agents:
                lower_bound = -5.0
                upper_bound = 5.0
                selected_actions[agent] = np.clip(selected_actions[agent],lower_bound,upper_bound)
        new_states, rewards_test,terminals_test,infos_test = env_test.step(selected_actions)
        states_test = new_states
        return_epi_test = return_epi_test + rewards_test[agent_to_evaluate]
        if any([terminals_test[a] for a in terminals_test]):
            env_test.reset()
            break
    print('| LOW_LEVEL | test_iter: ',test_iter,'| nn: ',test_n,' |  return_epi_test: ',return_epi_test,' |')
    
    return return_epi_test

def to_1d_list(state):
    single_d = [item for sublist in state for item in sublist]
    return single_d

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


## use only one replay buffer
def train_high_level(env, env_test, high_level_agent,low_level_agent, args):
    states = env.reset()
    agent_to_train = env.agents[0]
    # Initialize replay memory
    total_step_cnt = 0
    epi_cnt = 0
    test_iter = 0
    returns_test = np.zeros((np.ceil(int(args['total_step_num']) / int(args['eval_step_freq'])).astype('int') + 1))
    test_length = returns_test.shape[0]
    ##returns_test = np.zeros((np.ceil(int(args['total_step_num']) / 100).astype('int') + 1))
    print('returns_test : ',returns_test.shape)

    agent_names = env.agents  
    state_dim = env.observation_space(agent_names[0]).shape[0]
    action_dim = env.action_space(agent_names[0]).shape[0]
    max_action = float(env.action_space(agent_names[0]).high[0])
    ##latent_dim = latent_cont_dim + latent_disc_dim

    replay_buffer = PPOBuffer(state_dim, 1, int(args['steps_per_epoch']),
                              args['gamma'], args['lam'] )
    while total_step_cnt in range( int(args['total_step_num']) ):
        states = env.reset()
        obs_dict = {} ##high-level states
        low_level_states = {} ##low-level states
        low_level_actions = {}
        high_level_flags = {agent : False for agent in env.agents} ##high-level actionをサンプルするタイミング決める用
        high_level_actions = {agent : np.zeros(3) for agent in env.agents}

        state_buffer = states[agent_to_train]
        action_buffer = 0.0
        value_buffer = 0.0
        logp_buffer = 0.0

        rally_count = 0
        ep_reward = 0
        ep_len = 0

        print('new epoch start')
        epoch_step = 0
        t = 0
        ##for t in range(int(args['steps_per_epoch'])):
        while epoch_step < int(args['steps_per_epoch']):
            for key in states.keys(): ##keys of 'states' are names of agents
                obs_dict[key] = states[key] 
                low_level_states[key] = states[key][:17]
            for agent in env.agents:
                if obs_dict[agent][19] > 0.5 and high_level_flags[agent] == False:
                    high_level_actions[agent] = high_level_agent.select_action(np.array(obs_dict[agent]),stochastic = True)
                    if agent == agent_to_train:
                        ##print('player1 samples target velocity')
                        if rally_count > 0: ##相手が打てた時
                            ##print('replay buffer populates')
                            replay_buffer.store(state_buffer,action_buffer,1.0,value_buffer,logp_buffer)
                            epoch_step += 1

                        """仮で貯めておいて、相手が打てたら報酬を与える。"""
                        state_buffer = obs_dict[agent_to_train]
                        action_buffer = high_level_actions[agent_to_train][0]
                        value_buffer = high_level_actions[agent_to_train][1]
                        logp_buffer = high_level_actions[agent_to_train][2]
                    else:
                        ##print('player2 samples target velocity')
                        rally_count += 1
                ##print('high_level_actions : ',high_level_actions[agent])
                low_level_states[agent][16] = 2 * np.clip(high_level_actions[agent][0],-1.0,1.0) + 4.0
                high_level_flags[agent] = obs_dict[agent][19] > 0.5
            
            """PPOを使ったバージョンに直す"""
            ##print('low_level_states : ',low_level_states)
            low_level_actions = {agent : low_level_agent.select_action(np.array(low_level_states[agent]),stochastic=True)[0] for agent in env.agents} ##LL_actions
            for agent in env.agents:
                lower_bound = -5.0
                upper_bound = 5.0
                low_level_actions[agent] = np.clip(low_level_actions[agent],lower_bound,upper_bound)
            new_states,rewards,terminals,infos = env.step(low_level_actions)

            states = new_states

            """一つだけで良い"""
            ep_reward += rewards[agent_to_train]

            total_step_cnt += 1
            ep_len += 1
            timeout = ep_len == int(args['max_episode_len'])
            terminal =  any([terminals[a] for a in terminals]) or timeout
            epoch_ended = epoch_step == int(args['steps_per_epoch']) - 1

            """TO-DO : Process the buffers when a terminal state is called"""
            if terminal or epoch_ended:
                replay_buffer.store(state_buffer,action_buffer,0.0,value_buffer,logp_buffer) ##報酬がもらえなかった最後の経験をバッファに追加
                epoch_step += 1
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                
                if timeout or epoch_ended:
                    for agent in env.agents:
                        obs_dict[agent] = states[key] ##余計なデータを省く（余計なデータなし！）
                    _, v, _ = high_level_agent.select_action(np.array(obs_dict[agent_to_train]),stochastic=True)
                    ##print('epoch_ended')
                else:
                    v = 0
                    ##print('terminal state')

                replay_buffer.finish_path(v)

                epi_cnt += 1
                if epi_cnt % 10 == 0: ##%10 -> %100
                        ##print('| Reward: {:d} | Episode: {:d} | Total step num: {:d} |'.format(int(ep_reward), epi_cnt, total_step_cnt ))
                        print('| Reward: ',ep_reward,' | Episode: ',epi_cnt,' | Total step num: ',total_step_cnt,' |')
                        print('High-Level Step count : ',epoch_step)

                ##resetting
                states = env.reset()
                ep_reward = 0
                ep_len = 0
                rally_count = 0

            # Evaluate the deterministic policy
            if total_step_cnt >= test_iter * int(args['eval_step_freq']) or total_step_cnt == 1:
                print('total_step_cnt', total_step_cnt)
                print('evaluating the deterministic policy...')
                for test_n in range(int(args['test_num'])):
                    return_epi_test = evaluate_greedy_high_level(env_test, high_level_agent, low_level_agent,args, test_iter, test_n, state_dim)

                    # Store the average of returns over the test episodes
                    if test_iter < test_length:
                        returns_test[test_iter] = returns_test[test_iter] + return_epi_test / float(args['test_num'])
                ##print('return_test[{:d}] {:d}'.format(int(test_iter), int(returns_test[test_iter])))
                print('return_test[',test_iter,'] : ',returns_test[test_iter])
                test_iter += 1

            if total_step_cnt % int(args['model_save_freq']) == 0:
                high_level_agent.save_model(iter=test_iter, seed=int(args['trial_idx']), env_name=args['high_level_env'],foldername=args['check_point_path_high_level'])
            ##t += 0

        print('total step cnt', total_step_cnt)
        print('updated')
        high_level_agent.update(replay_buffer)
    return returns_test


def train_low_level_ppo(env, env_test, low_level_agent, args):
     # Initialize replay memory
    agent_to_train = env.agents[0] ## key for player 1
    print('agent_to_train',agent_to_train)
    total_step_cnt = 0
    epi_cnt = 0
    test_iter = 0
    return_test = np.zeros((np.ceil(int(args['total_step_num']) / int(args['eval_step_freq'])).astype('int') + 1))

    state_dim = env.observation_space(agent_to_train).shape[0]
    action_dim = env.action_space(agent_to_train).shape[0]
    max_action = float(env.action_space(agent_to_train).high[0])
    ##latent_dim = latent_cont_dim + latent_disc_dim

    ##replay_buffer = LPPOBuffer(state_dim, action_dim, latent_dim, int(args['steps_per_epoch']),
                              ##args['gamma'], args['lam'] )
    replay_buffer = PPOBuffer(state_dim, action_dim, int(args['steps_per_epoch']),
                              args['gamma'], args['lam'] )

    while total_step_cnt in range( int(args['total_step_num']) ):
        states = env.reset()
        ##state = env.reset()
        ep_reward = 0
        ep_len = 0
        obs_dict = {}
        low_level_actions = {}

        ##z = uniform_sampling(latent_cont_dim, latent_disc_dim)
        for t in range(int(args['steps_per_epoch'])):
            # Select action randomly or according to policy
            for agent in env.agents:
                obs_dict[agent] = states[agent]
            ##selected_actions = {agent : low_level_agent.select_action(np.array(obs_dict[agent]),z,stochastic=True) for agent in env.agents}
            selected_actions = {agent : low_level_agent.select_action(np.array(obs_dict[agent]),stochastic=True) for agent in env.agents}
            for agent in env.agents:
                low_level_actions = {agent : selected_actions[agent][0] for agent in env.agents}
            for agent in env.agents:
                lower_bound = -5.0
                upper_bound = 5.0
                low_level_actions[agent] = np.clip(low_level_actions[agent],lower_bound,upper_bound)
            new_states, rewards, terminals, info = env.step(low_level_actions)

            # Store data in replay buffer
            replay_buffer.store(states[agent_to_train], low_level_actions[agent_to_train], rewards[agent_to_train], selected_actions[agent_to_train][1], selected_actions[agent_to_train][2])
            states = new_states

            ep_reward += rewards[agent_to_train]

            total_step_cnt += 1
            ep_len += 1
            timeout = ep_len == int(args['max_episode_len'])
            terminal = any([terminals[a] for a in terminals]) or timeout
            epoch_ended = t == int(args['steps_per_epoch']) - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                
                if timeout or epoch_ended: ##timeout or end of epoch
                    _, v, _ = low_level_agent.select_action(np.array(states[agent_to_train]), stochastic=True)
                else: ##terminal
                    ##print('episode end')
                    v = 0

                replay_buffer.finish_path(v)
                ##print('episode reward : ',ep_reward)


                epi_cnt += 1
                if epi_cnt % 10 == 0:
                    ##print('| LOW-LEVEL | Reward: {:d} | Episode: {:d} | Total step num: {:d} |'.format(int(ep_reward), epi_cnt, total_step_cnt ))
                    print('| LOW_LEVEL | Reward: ',ep_reward,'| Episode: ',epi_cnt,' | Total step num: ',total_step_cnt, '|')

                states, ep_reward, ep_len = env.reset(), 0, 0
                ##z = np.random.uniform(-1, 1, size=(1, latent_dim))

            # Evaluate the deterministic policy
            if total_step_cnt >= test_iter * int(args['eval_step_freq']) or total_step_cnt == 1:
                print('total_step_cnt', total_step_cnt)
                print('LOW-LEVEL : evaluating the deterministic policy...')
                for test_n in range(int(args['test_num'])):
                    return_epi_test = evaluate_greedy_low_level_ppo(env_test, low_level_agent, args, test_iter, test_n, state_dim)

                    # Store the average of returns over the test episodes
                    return_test[test_iter] = return_test[test_iter] + return_epi_test / float(args['test_num'])

                ##print('LOW-LEVEL : return_test[{:d}] {:d}'.format(int(test_iter), int(return_test[test_iter])))
                print('return_test[',test_iter,'] : ',return_test[test_iter])
                test_iter += 1

            if total_step_cnt % int(args['model_save_freq']) == 0:
                low_level_agent.save_model(iter=test_iter, seed=int(args['random_seed']), env_name=args['low_level_env'],foldername=args['check_point_path_low_level'])


        print('total step cnt', total_step_cnt)
        low_level_agent.update(replay_buffer)

    return return_test

def main(args):
    for ite in range(int(args['trial_num'])):
        print('Trial Number:', ite)

        print("Low-Level Training Finished")
        ## if train_high_level == True, run high-leve training
        unity_env = UnityEnvironment(args['high_level_env'],no_graphics=True,worker_id=int(args['port_offset_high']))
        high_level_env = UnityParallelEnv(unity_env)
        np.random.seed(int(args['random_seed']) )
        high_level_env.seed(int(args['random_seed']))

        high_level_env_test = UnityParallelEnv(unity_env)
        high_level_env_test.seed(int(args['random_seed']))

        agent_names = high_level_env.agents

        state_dim = high_level_env.observation_space(agent_names[0]).shape[0]
        action_dim = high_level_env.action_space(agent_names[0]).shape[0]
        print('action_space.shape', state_dim)
        print('observation_space.shape', action_dim)

        high_level_agent = PPO(state_dim=state_dim, action_dim=1,
                ac_kwargs=dict(hidden_sizes=(64,64)))
            
        low_level_agent = PPO(17,3,ac_kwargs=dict(hidden_sizes=(64,64)))
        if int(args['load_model']) == 1:
            low_level_agent.load_model(args["iter"],args["seed"],args["low_level_env"])

            ##low_level_agent.load_model(args['iter'],args['seed'],args['low-level-env']) ##<-t多分いらない
        
        if int(args['train_high_level']) == 1:
            step_R_i = train_high_level(high_level_env, high_level_env_test, high_level_agent,low_level_agent, args)
        else:
            low_level_agent = PPO(state_dim,action_dim,ac_kwargs=dict(hidden_sizes=(128,64)))
            step_R_i = train_low_level_ppo(high_level_env,high_level_env_test,low_level_agent,args)

        result_path = "./results"
        result_filename = result_path + args['result_file'] + '_' + args['high_level_env'] + '_'   \
                                + '_trial_idx_' + str(int(args['trial_idx'])) + '.txt'
        try:
            import pathlib
            pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
            print('result saved')
            np.savetxt(result_filename, np.asarray(step_R_i))
        except:
            print("A result directory does not exist and cannot be created. The trial results are not saved")
        print("High-Level Training Finished")
            

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--high_level_env', help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument('--low_level_env')
    parser.add_argument('--env-id', type=int, default=6, help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument("--latent-cont-dim", default=2, type=int)  # dimension of the continuous latent variable
    parser.add_argument("--latent-disc-dim", default=0, type=int)  # dimension of the discrete latent variable

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--total-step-num', help='total number of time steps', default=1000000)
    parser.add_argument('--eval-step-freq', help='frequency of evaluating the policy', default=100000)
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--trial_idx', help='index of trials', default=1)
    parser.add_argument('--test-num', help='number of test episodes', default=10)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=3)
    parser.add_argument('--model-save-freq', help='frequency of evaluating the policy', default=200000)
    parser.add_argument('--result_file',default='/hrl_')

    parser.add_argument('--load_model',default=0)
    parser.add_argument('--iter',default = 1001)
    parser.add_argument('--seed',default=13)
    parser.add_argument('--port_offset_high',default=0)
    parser.add_argument('--port_offset_low',default = 100)
    parser.add_argument('--check_point_path_low_level')
    parser.add_argument('--check_point_path_high_level')
    parser.add_argument('--train_high_level',type=int,default=0)
    args = parser.parse_args()

    args_tmp = parser.parse_args()

    """if args_tmp.env is None:
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
        args_tmp.env = env_dict[args_tmp.env_id]"""
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)
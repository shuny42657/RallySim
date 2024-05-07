import numpy as np
import scipy.signal
#from  .spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi, a = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        max_ind = torch.argmax(logits, keepdim=True)
        return Categorical(logits=logits), max_ind

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std), mu

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim,
                 hidden_sizes=(128, 64), activation=nn.Tanh):
        super().__init__()
        print('hidden_sizes',hidden_sizes)

        # policy builder depends on action space
        """if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)"""
        self.pi = MLPGaussianActor(state_dim,action_dim,hidden_sizes,activation) ##for the badminton task only Box comes in

        # build value function
        self.v = MLPCritic(state_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi, a_greedy = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), a_greedy.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        print('prt',self.ptr)
        print('max_size',self.max_size)
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

class PPO(object):
    def __init__(self,
            state_dim,
            action_dim,
            actor_critic=MLPActorCritic,
            ac_kwargs=dict(),
            steps_per_epoch=4000,
            gamma=0.99,
            clip_ratio=0.2,
            pi_lr=3e-4,
            vf_lr=3e-4,
            train_pi_iters=80,
            train_v_iters=80,
            lam=0.95,
            target_kl=0.01
    ):
        """
        Proximal Policy Optimization (by clipping),

        with early stopping based on approximate KL

        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with a
                ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
                module. The ``step`` method should accept a batch of observations
                and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``v``        (batch,)          | Numpy array of value estimates
                                               | for the provided observations.
                ``logp_a``   (batch,)          | Numpy array of log probs for the
                                               | actions in ``a``.
                ===========  ================  ======================================

                The ``act`` method behaves the same as ``step`` but only returns ``a``.

                The ``pi`` module's forward call should accept a batch of
                observations and optionally a batch of actions, and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``pi``       N/A               | Torch Distribution object, containing
                                               | a batch of distributions describing
                                               | the policy for the provided observations.
                ``logp_a``   (batch,)          | Optional (only returned if batch of
                                               | actions is given). Tensor containing
                                               | the log probability, according to
                                               | the policy, of the provided actions.
                                               | If actions not given, will contain
                                               | ``None``.
                ===========  ================  ======================================

                The ``v`` module's forward call should accept a batch of observations
                and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``v``        (batch,)          | Tensor containing the value estimates
                                               | for the provided observations. (Critical:
                                               | make sure to flatten this!)
                ===========  ================  ======================================


            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to PPO.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.

            gamma (float): Discount factor. (Always between 0 and 1.)

            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while
                still profiting (improving the objective function)? The new policy
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
                denoted by :math:`\epsilon`.

            pi_lr (float): Learning rate for policy optimizer.

            vf_lr (float): Learning rate for value function optimizer.

            train_pi_iters (int): Maximum number of gradient descent steps to take
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)

            train_v_iters (int): Number of gradient descent steps to take on
                value function per epoch.

            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used
                for early stopping. (Usually small, 0.01 or 0.05.)

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()

        # Set up logger and save configuration
        # logger = EpochLogger(**logger_kwargs)
        # logger.save_config(locals())

        # Random seed
        # seed += 10000 * proc_id()
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        # Instantiate environment
        # env = env_fn()
        # obs_dim = env.observation_space.shape
        # act_dim = env.action_space.shape

        # Create actor-critic module
        self.ac = actor_critic(state_dim, action_dim, **ac_kwargs).to(device)
        self.clip_ratio = clip_ratio

        # Set up optimizers for policy and value function
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = torch.optim.Adam(self.ac.v.parameters(), lr=vf_lr)

        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

    def select_action(self, state, stochastic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        a, v, logp, a_greedy = self.ac.step(state)
        if stochastic:
            return a.flatten(), v, logp
        return a_greedy.flatten()

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'].to(device), data['act'].to(device), data['adv'].to(device), data['logp'].to(device)

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'].to(device), data['ret'].to(device)
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self, buf):
        data = buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            # kl = mpi_avg(pi_info['kl'])
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                # logger.log('Early stopping at step %d due to reaching max kl.' % i)
                print('Early stopping at step %d due to reaching max kl.')
                break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)  # average grads across MPI processes
            self.pi_optimizer.step()

        # logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

    def save_model(self, iter, seed, env_name, foldername='./models/ppo/high_level'):
        try:
            import pathlib
            pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)

            torch.save(self.ac.state_dict(),
                       foldername + '/ppo_actor_' + env_name + '_seed' + str(seed) + '_iter' + str(iter) + '.pth')

            print('models is saved for iteration', iter)

        except:
            print("A result directory does not exist and cannot be created. The trial results are not saved")

    def load_model(self, iter, seed, env_name, foldername='models/ppo'):

        self.ac.load_state_dict(torch.load(
            foldername + '/ppo_actor_' + env_name + '_seed' + str(seed) + '_iter' + str(iter) + '.pth'))
import time

from collections import OrderedDict
import pickle
import numpy as np
import tensorflow as tf
import gym
import os
import sys

from cs285.infrastructure.env import *
from cs285.infrastructure.utils import *
from cs285.infrastructure.tf_utils import create_tf_session
from cs285.infrastructure.logger import Logger

from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import get_wrapper_by_name

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.losses = []
        self.logger = Logger(self.params['logdir'])
        self.sess = create_tf_session(self.params['use_gpu'], which_gpu=self.params['which_gpu'])

        # Set random seeds
        seed = self.params['seed']
        tf.set_random_seed(seed)
        np.random.seed(seed)

        #############
        ## ENV
        #############
        # TODO: mak env.
        # Make the gym environment
        print('Agent Params: ', self.params['agent_params'], end='\n\n')
        self.env = Env(self.params['agent_params']['env_params'])
        # if 'env_wrappers' in self.params:
        #     # These operations are currently only for Atari envs
        #     self.env = wrappers.Monitor(self.env, os.path.join(self.params['logdir'], "mri"), force=True)
        #     self.env = params['env_wrappers'](self.env)
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = True
        # Observation and action sizes
        ob_dim = self.env.observation_shape
        ac_dim = (1,)

        self.params['agent_params']['discrete'] = discrete
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        # if 'model' in dir(self.env):
        #     self.fps = 1/self.env.model.opt.timestep
        # elif 'env_wrappers' in self.params:
        #     self.fps = 30 # This is not actually used when using the Monitor wrapper
        # else:
            # self.fps = self.env.env.metadata['video.frames_per_second']


        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.sess, self.env, self.params['agent_params'])

        #############
        ## INIT VARS
        #############

        tf.global_variables_initializer().run(session=self.sess)

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            #print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent):
                # only perform an env step and add to replay buffer for DQN
                self.agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                paths, envsteps_this_batch, train_video_paths = self.collect_training_trajectories(itr, initial_expertdata, collect_policy, self.params['batch_size'])

            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            loss = self.train_agent()
            self.losses.append(loss)
            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                if isinstance(self.agent, DQNAgent):
                    self.perform_dqn_logging()
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, loss)


                # save policy
                if self.params['save_params']:
                    print('\nSaving agent\'s actor...')
                    self.agent.actor.save(self.params['logdir'] + '/policy_itr_'+str(itr))
                    self.agent.critic.save(self.params['logdir'] + '/critic_itr_'+str(itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, load_initial_expertdata, collect_policy, batch_size):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # DONE
        # T decide whether to load training data or use
        # HINT: depending on if it's the first iteration or not,
            # decide whether to either
                # load the data. In this case you can directly return as follows
                # ``` return loaded_paths, 0, None ```

                # collect data, batch_size is the number of transitions you want to collect.
        if itr == 0:
            if load_initial_expertdata is not None:
                paths = pickle.load(open(self.params['expert_data'], 'rb'))
                return paths, 0, None

        # DONE
        # T collect data to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## DONE: T look in utils and implement sample_n_trajectories
            train_video_paths = sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            # DONE
            # T sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])

            # DONE
            # T use the sampled data for training
            # HINT: use the agent's train function
            # HINT: print or plot the loss for debugging!
            # for _ in range(10):
            loss = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

        return loss

    def do_relabel_with_expert(self, expert_policy, paths):
        print("\nRelabelling collected observations with labels from an expert policy...")

        # DONE
        # T relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        for i in range(len(paths)):
            acs = expert_policy.get_action(paths[i]["observation"])
            paths[i]["action"] = acs

        return paths

    ####################################
    ####################################
    def perform_dqn_logging(self):
        episode_rewards = self.losses[-100:]
        self.mean_episode_reward = np.mean(episode_rewards)

        if len(self.losses) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, loss):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            if isinstance(loss, dict):
                logs.update(loss)
            else:
                logs["Training loss"] = loss

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

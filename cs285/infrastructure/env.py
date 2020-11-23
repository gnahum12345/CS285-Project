from collections import deque
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import os
import uuid

class Env:
    def __init__(self, args):
        print(args)
        self.action_shape = args['mask_shape']

        # self.ksp = args['ksp'][:,:,32:288]
        self.data_path = args['ksp_data_path']
        self.files = os.listdir(self.data_path)

        self.index = 0
        self.get_ksp()
        self.coord = args['coord']

        self.step_idx = 0

        ### Set up for which environment we should use
        self.total_var = args['total_var'] # use total variation or not.
        self.loss_type = args['loss_type'] # this is 1 for l1, or 2 for l2.
        self.cartesian = args['cartesian'] # true or false.

        #### Fully sampled image for calculating error metric ####
        self.losses = []
        self.rewards = []

        self.action_mask = np.ones(self.action_shape)
        self.fully_sampled = self.get_image()
        np.save('pre_fs.npy', self.fully_sampled)
        ##### Setting up normal runs #####

        self.action_mask = np.zeros(self.action_shape) # 256 lines of k-space. For radial its 360 degrees of freedom. This is the mask.
        self.observation_shape = (256,256,2)
        self.action_space = list(range(108, 148))


        self.state_buffer = deque([], maxlen=args['history_length'])
        self.training = True

        self.window = args['history_length']

        self.next_step = np.zeros((256,256))
        self.prev_step = np.zeros((256,256)) # useless???
        self.prev_loss = 0
        self.last_action = None
        self.high = 1
        self.counter = 20

    def sample(self):
        if len(self.action_space) != 0:
            act = self.action_space.pop()
            print(len(self.action_space))
            return [act]

        if self.high == 1:
            self.counter += 1
            self.high = -1
        else:
            self.high = 1

        act = 128 + self.high*self.counter
        if act > 255 or act < 0:
            return self.last_action

        return [act]


    def get_ksp(self):
        n = len(self.files)
        self.f = self.files[self.index % n]
        # self.index += 1

        while '.npy' not in self.f and self.index < n:
           self.f = self.files[self.index]
           self.index += 1

        ksp_data_path = os.path.join(self.data_path, self.f)
        self.ksp =  np.load(ksp_data_path)[:,:,32:288]  #cropped to 256,256

    def get_image(self):
        # import pdb; pdb.set_trace()
        visibile_data = self.ksp * self.action_mask
        assert self.ksp.shape == visibile_data.shape, 'Data should be the same shape'
        # total variation
        lamda = 0.005
        if self.total_var: # best one to use now.
            mps = mr.app.EspiritCalib(visibile_data).run() # This only works for cartesian.
            return mr.app.TotalVariationRecon(visibile_data, mps, lamda, coord=self.coord).run() # requires coordinates

        if self.cartesian:
            return sp.rss(sp.ifft(visibile_data),axes=0) # (256,256)

        dcf = (self.coord[...,0]**2 + self.coord[...,1]**2)**0.5
        img_grid = sp.nufft_adjoint(visibile_data * dcf, coord)
        return img_grid # not quite right. need 2 dimensions (real, imaginary)

    def _reset_buffer(self):
        # TODO: make this more efficient.
        for _ in range(self.window):
            self.state_buffer.append(np.zeros(self.prev_step.shape))

    def save(self, file_dir='./'):
        np.save(file_dir + 'losses_{}.npy'.format(self.step_idx), self.losses)
        np.save(file_dir + 'rewards_{}.npy'.format(self.step_idx), self.rewards)
        np.save(file_dir + 'post_fs.npy',  self.fully_sampled)

    def reset(self,file_dir='./'):
        np.save(file_dir + 'losses_{}.npy'.format(self.step_idx), self.losses)
        np.save(file_dir + 'rewards_{}.npy'.format(self.step_idx), self.rewards)
        np.save(file_dir + 'post_fs.npy',  self.fully_sampled)

        print('done')
        self.action_space = list(range(108, 148))
        self.high = 1
        self.counter = 20
        self.losses = []
        self.rewards = []

        obs = self.get_image()
        self.action_mask = np.zeros(self.action_shape)
        self.get_ksp()
        ret = np.stack((obs, self.action_mask)).transpose([1,2,0])
        # self.index = 0å
        self._reset_buffer()
        self.state_buffer.append(obs)
        self.prev_step = np.zeros((256,256))
        self.prev_loss = 0
        self.last_action = None
        return ret

    def rreset(self, acts, rewards=[]):
        self.reset()
        for action in acts:
            obs, re, done = self.step(action)
            rewards.append(re)
        print('done reseting with acts')
        return rewards

    def act(self, action):
        '''
        Pick the line of k-space and sample it and return the reward.
        '''
        from skimage.measure import compare_ssim as ssim
        self.action_mask[action, :] = np.ones(self.action_shape[-1])
        self.next_step = self.get_image()
        # print(self.action_shape, next_step.shape, self.prev_step)

        if self.loss_type == 1:
            # FIX: this might be broken...
            loss = np.sum(np.abs(self.next_step - self.fully_sampled))
        elif self.loss_type == 2:
            # FIX: this might also be broken.
            loss = np.sum(np.square(self.next_step - self.fully_sampled))
        elif self.loss_type == 3:
            # raise Exception("This hasn't been implemented yet. \nPlease use l1/l2 losses")
            loss = ssim(self.fully_sampled, self.next_step, data_range=1)
            print('SSIM LOSS: {}'.format(loss))
        # if self.last_action is not None:
        #     loss = loss + (self.last_action[0] - action[0]) # loss decreases if they are cose together.
        #     print("NEW LOSS: {}".format(loss))
            # print(type(loss))
        # print('Act: index: {}, nonZero: {}\n\n\n\n\n loss: {}, prev_loss: {}'.format(self.index, np.count_nonzero(self.action_mask), loss, self.prev_loss))

        return loss

    def step(self, action):
        self.prev_step = self.next_step # update prev_step.
        # import pdb; pdb.set_trace()
        loss = self.act(action)


        reward = (loss - self.prev_loss) # prev_step - next_step / fully_sampled.
        if self.last_action is not None:
            if self.last_action[0] - action[0] == 0: # same action this should never happen.
                reward = 0
            print('idx: {} \naction: {}, last_action: {}, diff: {}\nReward: {}'.format(self.step_idx, action, self.last_action, abs(self.last_action[0] - action[0]), reward))
        # print(type(loss))
        self.step_idx += 1
        np.save('./img_{}.npy'.format(self.step_idx), self.next_step)
        np.save('./mask_{}.npy'.format(self.step_idx), self.action_mask)
        self.losses.append(loss)
        self.rewards.append(reward)

        self.last_action = action

        done = np.abs(reward) < 5e-3 #0.5%
        # if done:
        #     if reward == 0:
        #         import pdb; pdb.set_trace()

        self.prev_loss = loss

        observation = self.prev_step
        obs = np.stack((observation, self.action_mask))
        obs = obs.transpose([1, 2, 0])

        self.state_buffer.append(observation)
        return obs, reward, done

    def eval(self):
        self.training = False

    def render(self):
        pl.ImagePlot(self.next_step, title='Current State')

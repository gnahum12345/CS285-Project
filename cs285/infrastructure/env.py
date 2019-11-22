from collections import deque
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl

class Env:
    def __init__(self, args):
        print(args)
        self.action_shape = args['mask_shape']
        print('ActionShap: \n\n ', self.action_shape)
        self.action_mask = np.zeros(self.action_shape) # 256 lines of k-space. For radial its 360 degrees of freedom. This is the mask.
        self.observation_shape = (256,256,2)
        self.action_space = list(range(256))
        self.ksp = args['ksp'][:,:,32:288]
        self.coord = args['coord']
        self.state_buffer = deque([], maxlen=args['history_length'])
        self.training = True

        self.window = args['history_length']

        self.baseline = args['baseline']
        self.total_var = args['total_var']
        self.loss_type = args['loss_type'] # this is l1, or l2.
        self.cartesian = args['cartesian']
        self.prev_step = np.zeros((256,256)) # useless???
        self.prev_loss = float('-inf')

    def sample(self):
        rand = np.random.choice(self.action_space, 1)
        self.action_space.remove(rand)
        return rand

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

    def reset(self):
        obs = self.get_image()
        self.action_mask = np.zeros(self.action_shape)
        ret = np.stack((obs, self.action_mask)).transpose([1,2,0])

        self._reset_buffer()
        print(type(obs))
        self.state_buffer.append(obs)
        self.prev_step = np.zeros((256,256))
        self.prev_loss = float('-inf')
        return ret

    def act(self, action):
        '''
        Pick the line of k-space and sample it and return the reward.
        '''

        self.action_mask[action] = np.ones(self.action_shape[-1])
        next_step = self.get_image()
        print('ACT:\nn\n\n\n\n')
        print(self.action_shape, next_step.shape, self.prev_step)

        if self.loss_type == 1:
            # FIX: this might be broken...
            loss = np.sum(np.abs(next_step - self.prev_step))
        elif self.loss_type == 2:
            # FIX: this might also be broken.
            loss = np.sum(np.square(next_step - self.prev_step))
        elif self.loss_type == 3:
            raise Exception("This hasn't been implemented yet. \nPlease use l1/l2 losses")

        self.prev_step = next_step # update prev_step.
        return loss

    def step(self, action):
        # import pdb; pdb.set_trace()
        loss = self.act(action)
        reward = (loss - self.prev_loss) / self.prev_loss
        self.prev_loss = loss
        done = reward < 0.005

        observation = self.prev_step
        obs = np.stack((observation, self.action_mask))
        obs = obs.transpose([1, 2, 0])

        self.state_buffer.append(observation)
        return obs, reward, done

    def eval(self):
        self.training = False

    def render(self):
        pl.ImagePlot(self.prev_step, title='Current State')

import gym
from gym import spaces
import pygame
import numpy as np
import torch
import os
import imageio
from torchtnt.utils.timer import Timer


class WorldModelEnv(gym.Env):
    

    def __init__(self, tokenizer, transformer, init_frame, max_steps,context_length, size = 256):
        self.res = size
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_steps = max_steps
        self.context_length = context_length
        self.timer = Timer()
        self.init_f = torch.Tensor(init_frame)
        # print(self.init_f)
        # print(self.init_f.shape)
        self.frames = [self.init_f]

        self.context_q = [self.init_f for i in range(2)]


    def _get_obs(self):
        return self.frames[-1]

    def reset(self):
        super(reset, None)
        # there's nothing much else to do.
        self.frames = [self.init_f]
    
    def step(self, action, pred_step_func):

        # timer start
        with self.timer.time("step"):
        
            # this assumes context length 2. can make it dynamic but i don't care rn. # what the fuck?
            self.context_q = [self.context_q[1], self.context_q[2], self.frames[-1].reshape(1,3,256,256)]
            f = torch.stack(self.context_q)

            next_frame = pred_step_func(self.tokenizer, self.transformer, torch.Tensor(f), len(self.frames), actions=torch.FloatTensor(action))

        #timer stop
        dur = self.timer.recorded_durations['step'][-1]
        print(f"step {len(self.frames)} took {dur:.4f} seconds...")

        self.frames.append(torch.Tensor(next_frame))
        # self.render()

        return self.frames[-1]


    def summary(self):
        print(f"Full trajectory image generation took {sum(self.timer.recorded_durations['step']):.4f} seconds.")
    def render(self):
        print("render wip")








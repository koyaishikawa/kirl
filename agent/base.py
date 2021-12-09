import torch
from torch import nn
import numpy as np


class Base:       
    def observe(self):
        raise NotImplementedError()
        
    def update(self):
        raise NotImplementedError()

    def act(self):
        raise NotImplementedError()

    def evaluation(self):
        raise NotImplementedError()
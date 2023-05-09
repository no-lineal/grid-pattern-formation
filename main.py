import numpy as np
import torch

from utils import generate_run_ID
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN
from trainer import Trainer

import argparse

import os
from datetime import datetime

# viz
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

# external

save_dir = './models/' + str(datetime.now()) + '/'
os.mkdir( save_dir )
os.mkdir( save_dir + 'trajectories/' )

parser.add_argument(
    '--save_dir',
    default=save_dir,
    help='directory to save trained models'
    )

parser.add_argument(
    '--device',
    default='cuda' if torch.cuda.is_available() else 'cpu',
    help='device to use for training'
    )

parser.add_argument(
    '--example_place_cells', 
    default=True,
    help='use the pre-computed place cells center'
)

# learning

parser.add_argument(
    '--RNN_type', 
    default='RNN', 
    help='RNN or LSTM'
    )

parser.add_argument(
    '--activation', 
    default='relu', 
    help='recurrent nonlinearity'
    )

parser.add_argument(
    '--weight_decay', 
    default=1e-4, 
    help='strength of weight decay on recurrent weights'
    )

parser.add_argument(
    '--n_epochs',
    default=100,
    help='number of training epochs'
    )

parser.add_argument(
    '--n_steps',
    default=1000,
    help='batches per epoch'
    )

parser.add_argument(
    '--learning_rate',
    default=1e-4,
    help='gradient descent learning rate'
    )

parser.add_argument(
    '--batch_size',
    default=200,
    help='number of trajectories per batch'
    )

# trajectory simulation

parser.add_argument(
    '--Np',
    default=512,
    help='number of place cells'
    )

parser.add_argument(
    '--Ng',
    default=4096,
    help='number of grid cells'
    )

parser.add_argument(
    '--sequence_length',
    default=20,
    help='number of steps in trajectory'
    )

parser.add_argument(
    '--periodic',
    default=False,
    help='trajectories with periodic boundary conditions'
    )

parser.add_argument(
    '--place_cell_rf',
    default=0.12,
    help='width of place cell center tuning curve (m)'
    )

# DoG

parser.add_argument(
    '--DoG', 
    default=True, 
    help='use difference of gaussians tuning curves'
    )

parser.add_argument(
    '--surround_scale',
    default=2,
    help='if DoG, ratio of sigma2^2 to sigma1^2'
    )

# shape

parser.add_argument(
    '--box_width',
    default=2.2,
    help='width of training environment'
    )

parser.add_argument(
    '--box_height', 
    default=2.2, 
    help='height of training environment'
    )

options = parser.parse_args()

options.run_ID = generate_run_ID(options)

print('\n')
print(f'using device: {options.device}')
print(f'rnn type: {options.RNN_type}')
print('\n')

place_cells = PlaceCells(options)

# viz
us = place_cells.us
plt.figure(figsize=(5,5))
plt.scatter( us.cpu()[:,0], us.cpu()[:,1], c='lightgrey', label='Place cell centers' )
plt.savefig( options.save_dir + 'place_cells.png' )

# trajectory simulation

trajectory_generator = TrajectoryGenerator( options, place_cells )

# model

if options.RNN_type == 'RNN':

    model = RNN( options, place_cells )
    model = model.to( options.device )

    print('\n')
    print(f'model parameters: ')
    print(model)
    print('\n')

else:

    pass

# train
trainer = Trainer( options, model, trajectory_generator )
trainer.train( n_epochs=options.n_epochs, n_steps=options.n_steps )
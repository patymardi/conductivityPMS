#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:58:10 2023

@author: shaheim ogbomo-harmitt + conductivityPMS

"""

import os
EXAMPLE_DESCRIPTIVE_NAME = 'SIMULATION'
EXAMPLE_AUTHOR = 'conductivityPMS' 
EXAMPLE_DIR = os.path.dirname(__file__)
GUIinclude = False

import numpy as np
import shutil
import random
from scipy import spatial
from datetime import date
import pandas as pd
from carputils import settings
from carputils import tools
from carputils import mesh
from carputils import model
from PSD import *
from carp_to_pv import *

def parser():
    
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')
    group.add_argument('--M_lump',
                        type=int,
                        default='0',
                        help='set 1 for mass lumping, 0 otherwise. Mass lumping will speed up the simulation. Use with regular meshes.')
    group.add_argument('--stim_size',
                        type=str,
                        default='2000.0',
                        help='stimulation edge square size in micrometers')
    # Parameters of PSD
    group.add_argument('--PSD_bcl',
                        type=float,
                        default=315,
                        help='BCL in ms')   
    group.add_argument('--radius',
                        type=float, 
                        default = 7500, # 75000
                        help='radius of circles in which to set the phase from -pi to +pi')
    group.add_argument('--seeds',
                        type=int, 
                        default= 50,
                        help='# of initial seeds in which to set the phase from -pi to +pi')
    group.add_argument('--chirality',
                        type=int, 
                        default=1,
                        help='Chirality of the rotation: -1 for counterclockwise, 1 for clockwise')
    group.add_argument('--cv',
                        type=float, 
                        default=0.6,
                        help='conduction velocity in m/s')
    group.add_argument('--mesh_pth',
                        type=float, 
                        default=None,
                        help='file path to carp files')
    group.add_argument('--parameters_pth',
                        type=float, 
                        default=None,
                        help='file path to carp .par file')
    group.add_argument('--node_ID',
                        type=float, 
                        default=None,
                        help='Node ID of phase singularity node [x,y,z]')

    return parser


def jobID(args):
    
    """
    Generate name of top level output directory.
    """
    out_DIR = os.getcwd() + '/'  + 'Sim'
    
    return out_DIR


@tools.carpexample(parser, jobID)


def run(args, job):
    
    # Setting up mesh - path to test mesh 
    
    meshname = args.mesh_pth

    # Simulation parameters and conductivites - tagging all fibrosis regions with same parameters
    
    cmd = tools.carp_cmd(os.path.join(EXAMPLE_DIR, args.parameters_pth))

    print ('-----------------------------------')
    print('MESH HAS BEEN SETUP WITH REGION TAGS')
    print ('-----------------------------------')

    xyz_pth = args.mesh_pth + '.pts'
    triangles_pth = args.mesh_pth + '.elem'
    xyz = np.loadtxt(xyz_pth, skiprows=1)
    triangles = np.loadtxt(triangles_pth, skiprows=1, usecols = (1,2,3), dtype = int)
    
    pv_obj = carp_to_pv(args.mesh_pth)
    point = pv_obj.point[args.node_ID]
    centre = np.asarray(point)
    PSD(args, job, cmd, meshname, xyz, triangles,centre)

if __name__ == '__main__':
    run()

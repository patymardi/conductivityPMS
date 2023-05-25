#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:58:10 2023

@author: Patricia Martinez + Shaheim Ogbomo-Harmitt 

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
import pyvista as pv
import matplotlib

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
                        type=str,
                        default='../data/meshes',
                        help='file path to carp files')
    group.add_argument('--parameters_pth',
                        type=float, 
                        default=None,
                        help='file path to carp .par file')
    group.add_argument('--node_ID',
                        type=float, 
                        default=None,
                        help='Node ID of phase singularity node [x,y,z]')
    group.add_argument('--gil_file', type=str, default="gil.dat", help='File name where the intracellular conductivities in longitudinal direction are stored')
    group.add_argument('--gel_file', type=str, default="gel.dat", help='File name where the extracellular conductivities in longitudinal direction are stored')
    group.add_argument('--mesh',
                        type=str, default='case_1',
                        help='meshname directory. Example: case_1')
    group.add_argument('--results_dir',
                        type=str,
                        default='../results',
                        help='path to results folder')
    group.add_argument('--selection_mode', type=str, choices=['manual', 'file'], default='manual',
                        help='Mode for selecting the point and get ID: manual or file')
    group.add_argument('--point_file', type=str, default='point.txt',
                        help='Point ID file name')

    return parser


def jobID(args):
    
    """
    Generate name of top level output directory.
    """

    today = date.today()
    out_DIR = '{}/{}_{}'.format(args.results_dir,today.isoformat(),args.mesh)
    
    return out_DIR


@tools.carpexample(parser, jobID)


def run(args, job):
    
    # Setting up mesh - path to test mesh 
    
    meshname = '{}/{}'.format(args.mesh_pth,args.mesh)

    # Simulation parameters and conductivites
    
    cmd = tools.carp_cmd(args.parameters_pth)
    
    #Read the ge_vec file with intra and extracellular conductivities
    g_scale = ['-ge_scale_vec', '..data/{}'.format(args.gil_file),
               '-gi_scale_vec', '..data/{}'.format(args.gel_file)]
    
    cmd += g_scale

    print ('-----------------------------------')
    print('MESH HAS BEEN SETUP WITH REGION TAGS')
    print ('-----------------------------------')

    xyz_pth = meshname + '.pts'
    triangles_pth = meshname + '.elem'
    xyz = np.loadtxt(xyz_pth, skiprows=1)
    triangles = np.loadtxt(triangles_pth, skiprows=1, usecols = (1,2,3), dtype = int)

    #Convert carp files to vtk to visualize in Pyvista
    pv_mesh = pv.read('temp.vtk')
    #pv_mesh = carp_to_pv(meshname)

    # Define the landmarks
    if args.selection_mode == 'manual':
        point_xyz = select_point_manually(pv_mesh)
        np.savetxt('{}/point.txt'.format(args.mesh_pth),point_xyz, fmt='%f')
    else:
        point_xyz = load_point_from_file('{}/{}'.format(args.mesh_pth,args.point_file))

    #point = pv_obj.point[args.node_ID]
    centre = np.asarray(point_xyz)

    #For multiphase
    PSD(args, job, cmd, meshname, xyz, triangles,centre)

    #model.induceReentry.PSD(args, job, cmd, meshname, xyz, triangles,centre)

def select_point_manually(pv_mesh):

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the MRI geometry to the plotter
    plotter.add_mesh(pv_mesh, color='blue')

    # Prompt the user to select point using the PyVista plotter
    print("Please select the point by clicking on the geometry.")
    plotter.add_text('Select the point and then close the window', position='lower_left')

    plotter.enable_point_picking(show_message="Press P to pick")
    plotter.show()
    point_xyz = plotter.picked_point

    print("Coordinates: ", point_xyz)
    plotter.close()

    return point_xyz


def load_point_from_file(file_path):
    return np.loadtxt(file_path,dtype=float)



if __name__ == '__main__':
    run()

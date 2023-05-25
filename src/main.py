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
    group.add_argument('--point_file', type=str, default='path_to_ID_file.txt',
                        help='Path to the ID file')

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
    pv_obj = carp_to_pv(meshname)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the MRI geometry to the plotter
    plotter.add_mesh(pv_obj, color='gray')

    # Define the landmarks
    if args.selection_mode == 'manual':
        pointxyz = select_point_manually(plotter)
    else:
        pointxyz = load_point_from_file(args.point_file)

    #point = pv_obj.point[args.node_ID]
    centre = np.asarray(pointxyz)

    #For multiphase
    PSD(args, job, cmd, meshname, xyz, triangles,centre)

    #model.induceReentry.PSD(args, job, cmd, meshname, xyz, triangles,centre)
def select_point_manually(plotter):

    # Prompt the user to select point using the PyVista plotter
    print("Please select the point by clicking on the geometry.")

    # def pick_point(mesh, event):
    #     if event == "PointPickEvent":
    #         landmarks.append(mesh.points[mesh.point_pick_index])
    #         # Display the selected landmark
    #         plotter.add_mesh(pv.PolyData(landmarks[-1]), color='red', point_size=10)

    plotter.enable_point_picking(show_message="Press P to pick")
    plotter.show()

    pointID = plotter.picked_point
    print(pointID)
    return pointID


def load_point_from_file(file_path):
    return np.loadtxt(file_path)



if __name__ == '__main__':
    run()

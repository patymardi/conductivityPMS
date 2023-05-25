import os
import sys
import vtk

from datetime import date
from carputils import settings
from carputils import tools

import numpy as np
from carputils.carpio import igb
from scipy.spatial import cKDTree
import csv
import random
from vtk.numpy_interface import dataset_adapter as dsa
import Methods_fit_to_clinical_LAT

from sklearn.metrics import mean_squared_error

vtk_version = vtk.vtkVersion.GetVTKSourceVersion().split()[-1].split('.')[0]

def parser():
    # Generate the standard command line parser
    parser = tools.standard_parser()
    # Add arguments
    parser.add_argument('--gil_file', type=str, default="gil.dat", help='File name where the intracellular conductivities in longitudinal direction are stored')
    parser.add_argument('--gel_file', type=str, default="gel.dat", help='File name where the extracellular conductivities in longitudinal direction are stored')
    parser.add_argument('--model',
                        type=str,
                        default='Courtemanche',
                        help='input ionic model')
    parser.add_argument('--mesh',
                        type=str, default='',
                        help='meshname directory. Example: ')

def jobID(args):
    today = date.today()
    mesh = args.mesh.split('/')[-1]
    ID = '../results/{}_{}'.format(today.isoformat(), mesh)
    return ID

@tools.carpexample(parser, jobID)
def run(args, job):

    meshdir = '../data/meshes/'
    meshname = '../data/meshes/{}/bilayer/{}'.format(args.mesh, args.geometry)
    simid = job.ID


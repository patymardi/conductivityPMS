#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:21:51 2022

@author: Shaheim Ogbomo-Harmitt

Phase Singularity Distribution AF Initiation

Source Code: https://git.opencarp.org/openCARP/carputils/-/blob/master/carputils/model/induceReentry.py

"""

import os
import math

from carputils import settings
from carputils.carpio import igb

import numpy as np
import numpy.matlib
from numpy import linalg as LA
from scipy import spatial
from scipy.sparse import coo_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import bicgstab
from scipy.sparse.linalg import LinearOperator

# PSD

def assignRandomCoord(points, sing_points):

    coordinates = np.zeros((sing_points,3))
    coordinates[0] = points[np.random.randint(len(points))]
    directions = np.ones(sing_points, dtype=int)*(-1)
    balls = np.array([], dtype=int)
    for i in range(1,sing_points):
        balls = np.append(balls, spatial.cKDTree(points).query_ball_point(coordinates[i-1], 2*radius))
        p = np.ones(len(points), dtype=float)
        p[balls] = 0.0
        p /= sum(p)
        coordinates[i] = points[np.random.choice(len(points), replace=False, p=p)]
        distance, index = spatial.cKDTree(coordinates[0:i]).query(coordinates[i])
        if distance < 3.5*radius:
            directions[i] = 1

    return coordinates, directions

def assignDirection(coordinates):

    directions = np.ones(len(coordinates), dtype=int)*(-1)
    for i in range(1,len(coordinates)):
        distance, index = spatial.cKDTree(coordinates[0:i]).query(coordinates[i])
        if distance < 3.5*radius:
            directions[i] = 1

    return directions

def makeCircle3D(startCoord, radius, seedNum, normV, point1, point2, nodes):
    
    increment = (2 * np.pi) / seedNum
    seedCount = 0
    t = 0.
    timePoints = np.zeros((1, seedNum), dtype=float)
    vector = point1 - point2
    u = vector / LA.norm(vector)
    
    seeds_list = np.zeros((seedNum, 3))

    while seedCount < seedNum:
        coordinate = ((radius * np.cos(t)) * u) + np.cross((radius * np.sin(t)) * normV, u) + startCoord
        seeds_list[seedCount, :] = coordinate
        distance,index_node = spatial.cKDTree(nodes).query(coordinate)
        timePoints[0, seedCount] = index_node
        seedCount += 1
        t += increment
    
    return timePoints

def assignPhaseValues(seeds, originalCoord, directions, normV, point1, point2, nodes):
    
    # run plane info functions multiple times
    
    phaseCount = 0
    
    gammaPhi = np.zeros((2, seeds * len(originalCoord)), dtype=complex)
    while phaseCount < len(originalCoord):
        # assigns the known nodes
        gammaPhi[0, (phaseCount * seeds) : (phaseCount + 1) * seeds ] = makeCircle3D(originalCoord[phaseCount], radius, seeds, normV[phaseCount], point1[phaseCount], point2[phaseCount], nodes)
        if directions[phaseCount] == 1: # clockwise
            seeds_range = np.arange(1., seeds+1.)
        else: # counterclockwise
            seeds_range = np.arange(seeds, 0., -1.)
        seeds_range =  seeds_range/seeds # fractions, period itself doesn't matter
        phase_values = seeds_range*2*np.pi
        gammaPhi[1, (phaseCount * seeds) : (phaseCount + 1) * seeds ] = np.exp(1j*phase_values) #turns to phase values

        phaseCount += 1

    return gammaPhi


def planeInfo(coordinate, nodes, triangles):

    distance,index_node = spatial.cKDTree(nodes).query(coordinate, n_jobs=-1)

    points = [(row) for counter, row in enumerate(triangles) if index_node in row]
    saved = nodes[points[0]]

    plane_info = []
    for i in range(0,len(saved)):
        if points[0][i] != index_node:
            plane_info.append(saved[i])

    vec1 = saved[0] - saved[1]
    vec2 = saved[0] - saved[2]
    preNorm = np.cross(vec1,vec2)
    Norm = preNorm / LA.norm(preNorm)
    plane_info.append(Norm)
    plane_info = np.array(plane_info)

    return plane_info

def deflate(x):
    A_sol = A.dot(x) + np.mean(x)*np.ones((nv,1), dtype=float).reshape((nv,))
    return A_sol

def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

def eikonal_solver(nodes, triangles, knownNodes, phi0):

    global nv
    nv = len(nodes)
    nt = len(triangles)

    phi = np.ones((nv,1), dtype = complex)
    phi[knownNodes,0] = phi0
    tol = 10**(-2)
    iterationNum = 0
    updateChange = 1
    updateChange2 = 1
    maxIterations = 30
    triangles = np.array(triangles)
    

    u = (nodes[triangles[:,1],:]-nodes[triangles[:,0],:])/1000
    v = (nodes[triangles[:,2],:]-nodes[triangles[:,0],:])/1000
    # scalar products
    u2 = np.transpose(np.sum(u**2,axis=1))
    v2 = np.transpose(np.sum(v**2,axis=1))
    uv = np.transpose(np.sum(np.multiply(u,v),axis=1))
    u2.shape = (len(u2),1)
    v2.shape = (len(v2),1)
    uv.shape = (len(uv),1)
    # determinant
    delta = np.multiply(u2,v2) - np.multiply(uv,uv)

    e = np.ones((1,3), dtype=int)
    
    # (us,vs) is the dual basis of (u,v)
    us = np.multiply(np.multiply(np.divide(v2,delta),e), u) - np.multiply(np.multiply(np.divide(uv,delta),e), v)   
    vs = np.multiply(np.multiply(np.divide(u2,delta),e), v) - np.multiply(np.multiply(np.divide(uv,delta),e), u)

    # components of the gradient matrix
    Gx = np.column_stack((-us[:,0]-vs[:,0], us[:,0], vs[:,0]))     
    Gy = np.column_stack((-us[:,1]-vs[:,1], us[:,1], vs[:,1]))
    Gz = np.column_stack((-us[:,2]-vs[:,2], us[:,2], vs[:,2]))

    # indices in the sparse matrices
    col = np.concatenate(triangles, axis=0)
    row = np.zeros(3*nt)
    row[0:len(row)-2:3] = range(0,nt)
    row[1:len(row)-1:3] = range(0,nt)
    row[2:len(row):3] = range(0,nt)

    data = np.concatenate(Gx, axis=0)
    Gradx = coo_matrix((data, (row, col)),shape=(nt,nv)).tocsr()
    data = np.concatenate(Gy, axis=0)
    Grady = coo_matrix((data, (row, col)),shape=(nt,nv)).tocsr()
    data = np.concatenate(Gz, axis=0)
    Gradz = coo_matrix((data, (row, col)),shape=(nt,nv)).tocsr()

    # area of the triangles
    St = np.matlib.repmat(np.sqrt(delta)/2,1,3)

    data = np.concatenate(St, axis=0)
    Tt2v = coo_matrix((data, (col, row)),shape=(nv,nt))

    # area associated with each vertex and its inverse
    Sv = np.sum(Tt2v,axis=1)/3
    data = np.concatenate(np.reciprocal(Sv), axis=1)
    invS = dia_matrix((data,[0]),shape=(nv,nv))

    # interpolation triangle to vertex and vertex to triangle
    Tt2v = (invS/3).dot(Tt2v)
    data = np.ones(3*nt)*1/3
    Tv2t = coo_matrix((data, (row, col)),shape=(nt,nv)).tocsr()

    # create the sparse matrices
    data = np.concatenate(np.multiply(-St,Gx),axis=0)
    Divx = invS.dot(coo_matrix((data, (col, row)),shape=(nv,nt)).tocsr())
    data = np.concatenate(np.multiply(-St,Gy),axis=0)
    Divy = invS.dot(coo_matrix((data, (col, row)),shape=(nv,nt)).tocsr())
    data = np.concatenate(np.multiply(-St,Gz),axis=0)
    Divz = invS.dot(coo_matrix((data, (col, row)),shape=(nv,nt)).tocsr())

    # Calculating speed per triangle

    maxT = 1000.0/(2*np.pi)
    dDivisor = 2083.3

    c = np.ones(nt)*CV*maxT
    D = np.ones(nt)*(CV**2)*maxT/dDivisor

    # creates diagonal version for all of the D and c values
    DDiag = dia_matrix((D, [0]),shape=(nt,nt))
    cDiag = dia_matrix((c, [0]),shape=(nt,nt))

    Diag = identity(nt)
    DGradx = DDiag.dot(Diag.dot(Gradx))
    DGrady = DDiag.dot(Diag.dot(Grady))
    DGradz = DDiag.dot(Diag.dot(Gradz))

    CGradx = cDiag.dot(Diag.dot(Gradx))
    CGrady = cDiag.dot(Diag.dot(Grady))
    CGradz = cDiag.dot(Diag.dot(Gradz))

    print("Solving laplacian interpolation")

    while (updateChange2 > 0.1) and (iterationNum < 30):
        iterationNum = iterationNum+1

        phiRefLastIteration = phi
    
        # set value of phi on the triangles
        conjphi = np.conj(phi)
        data = np.concatenate(Tv2t.dot(conjphi),axis=0)
        phit = dia_matrix((data,[0]),shape=(nt,nt))
        # matrix A inf
        Ainf = Divx.dot(phit).dot(DGradx) + Divy.dot(phit).dot(DGrady) + Divz.dot(phit).dot(DGradz)
        # replace row i with row i of the identity matrix
        for row in knownNodes:
            csr_row_set_nz_to_val(Ainf, row, 0)

        # And to remove zeros from the sparsity pattern:
        Ainf.eliminate_zeros()

        Ainf[np.unravel_index(np.dot(knownNodes,nv) + knownNodes, (nv,nv))] = 1 #makes sure these values are fixed
        # right hand side finf, which has length nv with knownNodes
        finf = np.zeros((nv,1), dtype = complex)
        finf[knownNodes] = phi[knownNodes]

        # solve equation
        phiinf = spsolve(Ainf,finf)[np.newaxis].T
        
        # normalize
        phi = np.divide(phiinf,np.abs(phiinf))

        # set measured values
        phi[knownNodes,0] = phi0

        # set difference
        updateChange2 = LA.norm(phiRefLastIteration-phi)
        print("iterationNum=",iterationNum,"with updateChange=",updateChange2)

    phiLaplace = phi

    # Eikonal interpolation

    print("Solving eikonal diffusion equation")

    Diff = Divx.dot(DGradx) + Divy.dot(DGrady) + Divz.dot(DGradz)

    iterationNum = 0
    updateChange = 1.
    
    global A
    while (updateChange > tol) and (iterationNum < maxIterations):
        iterationNum = iterationNum+1
        print("iterationNum=",iterationNum,"with updateChange=",updateChange)
    
        # set value of phi on the triangles
        conjphi = np.conj(phi)
        phit = Tv2t.dot(conjphi)

        # norm of the gradient
        Norm = np.sqrt(np.abs(CGradx.dot(phi))**2 \
             + np.abs(CGrady.dot(phi))**2 \
             + np.abs(CGradz.dot(phi))**2)
        
        # Propagation velocity for every triangle
        cv = np.reciprocal(Norm)
        # matrix B
        B = dia_matrix((np.concatenate(CGradx.dot(conjphi),axis=0),[0]),shape=(nt,nt)).dot(CGradx) \
          + dia_matrix((np.concatenate(CGrady.dot(conjphi),axis=0),[0]),shape=(nt,nt)).dot(CGrady) \
          + dia_matrix((np.concatenate(CGradz.dot(conjphi),axis=0),[0]),shape=(nt,nt)).dot(CGradz) 

        # matrix A
        A = Diff \
          + 3./4*Tt2v.dot(dia_matrix((np.concatenate(cv,axis=0),[0]),shape=(nt,nt)).dot(np.imag(dia_matrix((np.concatenate(Tv2t.dot(phi),axis=0),[0]),shape=(nt,nt)).dot(B)))) \
          + 1./4*np.imag(dia_matrix((np.concatenate(phi,axis=0),[0]),shape=(nv,nv)).dot(Tt2v).dot(dia_matrix((np.concatenate(cv,axis=0),[0]),shape=(nt,nt)).dot(B)))
        
        # right hand side f
        f = Tt2v.dot(Norm-1.) \
          - Divx.dot(np.imag(np.multiply(phit, DGradx.dot(phi)))) \
          - Divy.dot(np.imag(np.multiply(phit, DGrady.dot(phi)))) \
          - Divz.dot(np.imag(np.multiply(phit, DGradz.dot(phi))))

        # solve system using deflation method to obtain correction term theta
        try:
            LU = spilu(A.tocsc(), drop_tol=10**(-3))
        except RuntimeError:
            phiEikonal = phiLaplace
            return phiLaplace, phiEikonal, -1000
        M_x = lambda x: LU.solve(x)
        M = LinearOperator((nv,nv), M_x)
        try:
            A_sol = LinearOperator((nv,nv), matvec = deflate)
            theta, num_iter = bicgstab(A_sol, f, tol=10**(-10), maxiter=100, M=M)
        except RuntimeError:
            phiEikonal = phiLaplace
            return phiLaplace, phiEikonal, -1000
        alpha = np.mean(theta)

        theta -= alpha

        T = 1000.0/(1.0 + alpha)
        print(T)

        # under relaxation step
        thetarelaxed = theta*min(1, 0.1/np.max(np.abs(theta)))
    
        # correct estimate for phi
        phi = np.multiply(phi,np.array(np.exp(1j*thetarelaxed)).reshape(nv,1))
        
        # set difference
        updateChange = LA.norm(thetarelaxed)

    phiEikonal = phi

    return phiLaplace, phiEikonal, T

    
def assignWallValues(surfacePoints, allPoints, surfaceValues):
    # assignWallValues: transfers phase data from the surface mesh to the
    # volumetric mesh. 
    #
    # Requires: 
    #   -surfacePoints: the coordinates of all of the nodes on the surface
    #   mesh.
    #   -allPoints: the coordinates of all of the nodes on the volumetric mesh.
    #   -surfaceValues: the phase values corresponding with the nodes on the
    #   surface mesh. 
    #
    # Results: returns a phi matrix which contains phase values for every point
    # on the volumetric mesh. 

    phi_vol = np.zeros((len(allPoints), 1), dtype = complex)
    distance, index_node = spatial.cKDTree(surfacePoints).query(allPoints, n_jobs=-1)
    phi_vol = surfaceValues[index_node]

    return phi_vol

def multi_phase_info(coordinates,xyz,triangles):
    
    normV_array = []
    point1_array = []
    point2_array = []
    
    for coordinate in coordinates:
        # normV, point1, point2
        #plane_info[2::3], plane_info[0:len(plane_info)-2:3], plane_info[1:len(plane_info)-1:3]
        plane_info = planeInfo(coordinate, xyz, triangles)
        normV_array.append(plane_info[2::3])
        point1_array.append(plane_info[0:len(plane_info)-2:3])
        point2_array.append(plane_info[1:len(plane_info)-1:3])
        
    return normV_array,point1_array,point2_array

def PSD(args, job, cmd, meshname, xyz, triangles, centre,tend = 500):
    """
    Phase Singularity Distribution:

    It consists of manually placing phase singularities on the geometrical model
    and then solving the Eikonal equation to estimate the activation time map. 
    Based on this initial state, you can simulate electrical wave propagation 
    by solving the monodomain equation.

    Input: parser arguments (args), output directory (job.ID), struct containing imp_regions and gregions (cmd_ref),
    meshname, stimulation point location (x,y,z) and prepacing directory (steady_state_dir)
    
    Args:
    '--M_lump',
        type=int,
        default='1',
        help='set 1 for mass lumping, 0 otherwise. Mass lumping will speed up the simulation. Use with regular meshes.'
    '--cv',
        type=float, 
        default=0.3,
        help='conduction velocity in m/s'
    '--PSD_bcl',
        type=float,
        default=160,
        help='BCL in ms' 
    '--radius',
        type=float, 
        default=10000.0,
        help='radius of circles in which to set the phase from -pi to +pi'
    '--seeds',
        type=int, 
        default=50,
        help='# of initial seeds in which to set the phase from -pi to +pi'
    '--chirality',
        type=int, 
        default=-1,
        help='Chirality of the rotation: -1 for counterclockwise, 1 for clockwise'
       
    """

    global coordinates, CV, radius
    CV = args.cv*1000 # here in mm/s
    radius = args.radius

    directions = []
    plane_info = []
    coordinates = np.atleast_2d(centre) # we locate the phase singularity in the centre
    print(len(coordinates))
    
    directions.append(args.chirality)
    print("Coordinates: ",coordinates)
    print("Directions: ",directions)
    
    """
    
    This where the phase values are evaluated for each node on the mesh.
    
    """
    
    # BEGIN REPITIION HERE
    
    # Just need to make second 'assignPhaseValues' function to allow addition of another phase singularity
    
    #plane_info = planeInfo(coordinates, xyz, triangles)
    
    #print(coordinates)
    
    normV, point1, point2 = multi_phase_info(coordinates,xyz,triangles)
    initialConditions = assignPhaseValues(args.seeds, coordinates, directions,normV, point1, point2, xyz)
    
    
    seedIndex = [int(initialConditions[0, i]) for i in range(len(initialConditions[0]))]
    
    phi0 = initialConditions[1, :]
    
    """
    
    Here is when the Eikonal equation is solved – should not require addition of multiple phase singularites. Double check
    
    """
    laplacian_sol, eikonal_sol, T = eikonal_solver(xyz, triangles, seedIndex, phi0)

    print("Assigning volume values")
    if T > -1000:
        print("Using Eikonal phases")
    else:
        print("T<-1000 - Using laplace phases")
        
    points_vol = np.loadtxt(meshname + '.pts', skiprows = 1)
    phi_Volume = assignWallValues(xyz, points_vol, eikonal_sol)
    phi_Volume_laplace = assignWallValues(xyz, points_vol, laplacian_sol)

    phase = np.angle(phi_Volume)
    phase_laplace = np.angle(phi_Volume_laplace)
    print("Computing LATs on the volume mesh with T=", T)
    factor = args.PSD_bcl*1.05

    LATS = [factor*(i/(2*np.pi)) for i in (phase + np.pi)] #Max LAT = BCL*1.05
    
    PHASE = [ i[0] for i in phase]
    PHASE_laplace = [ i[0] for i in phase_laplace]

    if not os.path.exists(job.ID):

        os.makedirs(job.ID)

    print("Writing LATs")
    writefile = job.ID + '/LATS.dat'
    file = open(writefile, 'w')

    phase_file = open(job.ID + '/phase.dat','w')
    phase_laplace_file = open(job.ID + '/phase_laplace.dat','w')

    for line in range(len(phase)):
        file.write("%f\n" % LATS[line])
        phase_file.write( '{}\n'.format(PHASE[line]))
        phase_laplace_file.write( '{}\n'.format(PHASE_laplace[line]))

    file.close()
    phase_file.close()
    phase_laplace_file.close()

    writestatef = 'state_FINAL'
    tsav_state = tend

    # Setting the stimulus
    stim = ['-num_stim', 0]
    cmd += stim
    cmd +=['-meshname', meshname,
        '-write_statef', writestatef,
        '-num_tsav', 1,
        '-tsav[0]', tsav_state,
        '-tend',     tend,
        '-simID',    job.ID,
        '-mass_lumping', args.M_lump,
        '-prepacing_lats', job.ID + '/LATS.dat',
        '-prepacing_beats', 5,
        '-prepacing_bcl' , args.PSD_bcl]


    # Run simulation
    
    job.carp(cmd)
    
    


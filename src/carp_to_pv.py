"""
Created on Thu May 25 11:04:04 2023

@author: shaheimogbomo-harmitt
"""

import os 
import pyvista as pv 


def carp_to_pv(pth,save_flag = False):
    
    # create temp dir for .vtk
    
    vtk_pth = os.getcwd() + '/' + 'temp'
    
    command = "meshtool convert" + " " + "-imsh=" + pth + " " + "-ifmt=carp_txt" + " " + "-omsh=" + vtk_pth + " -ofmt=vtk"
    print(command)
    os.system(command)
    
    pv_mesh = pv.read(vtk_pth + '.vtk')
    
    if save_flag == True:
        
        os.remove(vtk_pth + '.vtk')
    return pv_mesh


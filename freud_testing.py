#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:32:40 2023

@author: mmccrackan
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord

import freud

import requests
import json

# this function is from EDD, it lets you get the velocities from a given position
# and distance
def DVcalculator(alpha, delta, system='supergalactic', 
                 parameter='distance', value=20, calculator='NAM'):
    
    """
    Inputs: 
        alpha: (float) [deg]
            first coordinate parameter  (RA,  Glon, SGL)
        delta: (float) [deg]
            second coordinate parameter (Dec, Glat, SGB)  
        system: (string)
            coordinate system: 
            Options are:
                "equatorial"
                "galactic"
                "supergalactic"
        parameter: (string)
            the quantity whose value is provided
            Options are:
                "distance"
                "velocity"
        value: (float)
            the value of the input quantity
            distance in [Mpc] and velocity in [km/s]
            
        calculator: desired Cosmicflows caluclator
            Options are:
                "NAM" to query the calculator at http://edd.ifa.hawaii.edu/NAMcalculator
                "CF3" to query the calculator at http://edd.ifa.hawaii.edu/CF3calculator
        
    Output:
        A python dictionary which contains the distance and velocity of the 
        given object and the coordinate of the object in different systems

    """
    
    coordinate = [float(alpha), float(delta)]
    query  = {
              'coordinate': coordinate,
              'system': system,
              'parameter': parameter,
              'value': float(value)
             }
    headers = {'Content-type': 'application/json'}
    
    API_url = 'http://edd.ifa.hawaii.edu/'+calculator+'calculator/api.php'
    
    try:
        r = requests.get(API_url, data=json.dumps(query), headers=headers)
        output = json.loads(r.text) # a python dictionary
    except:
        print("Something went wrong!")  
        print("Please check your intput parameters ...")
        output = None

    return output

if __name__ == "__main__":
    # tables containing CF4 and CF3 data
    cf4 = Table.read('/Users/mmccrackan/fcrao/data/cf4_2.txt',format='csv')
    cf3 = Table.read('/Users/mmccrackan/fcrao/data/cf3_2.txt',format='csv')
            
    
    # the freud voronoi tesellsation crashes for the full CF4 sample
    factor = 10
    x = cf4['SGX'][::factor]
    y = cf4['SGY'][::factor]
    # set z to 0 to do 2d voronoi tesellation
    z = np.abs(cf4['SGZ']*0)[::factor]

    # data points
    points = np.array([x,y,z]).transpose()

    ''' voronoi'''
    # 2D
    # box size must contain all points
    L = 700
    # create 2d square
    box = freud.box.Box.square(L)
    voro = freud.locality.Voronoi()
    cells = voro.compute((box, points)).polytopes
    plt.figure()
    plt.hist(np.log(voro.volumes))
    plt.title('2D voronoi Volumes')
    plt.xlabel('log volumes (Mpc)')

    plt.figure()
    ax = plt.gca()
    voro.plot(ax=ax, cmap="RdBu")
    ax.scatter(points[:, 0], points[:, 1], s=2, c="k")
    ax.set_title('CF4 Downsampled 2D voronoi diagram')
    plt.show()

    #3d
    factor = 10
    x = cf4['SGX'][::factor]
    y = cf4['SGY'][::factor]
    z = cf4['SGY'][::factor]

    points = np.array([x,y,z]).transpose()
    box = freud.box.Box.cube(L)
    voro = freud.locality.Voronoi()
    cells = voro.compute((box, points)).polytopes
    plt.figure()
    plt.hist(np.log(voro.volumes))
    plt.title('3D voronoi Volumes')
    plt.xlabel('log volumes (Mpc)')

    ''' Nearest neighbor '''
    factor = 1 # don't need to downsample this one
    x = cf4['SGX'][::factor]
    y = cf4['SGY'][::factor]
    z = cf4['SGY'][::factor]
    
    points = np.array([x,y,z]).transpose()

    box = freud.box.Box.cube(L)
    aq = freud.locality.AABBQuery(box, points)

    # query points are the positions you are intersted in finding the nearest 
    # neightbors of
    query_points = np.array([x[0],y[0],z[0]]).transpose()
    distances = []
    volumes = []

    num_neighbors = 4

    k = 0
    # Here, we ask for the 4 nearest neighbors of each point in query_points.
    query_result = aq.query(query_points, dict(num_neighbors=num_neighbors))
    nlist = query_result.toNeighborList()
    for (i, j) in nlist:
        # Note that we have to wrap the bond vector before taking the norm;
        # this is the simplest way to compute distances in a periodic system.
        distances.append(np.linalg.norm(box.wrap(query_points[i] - points[j])))
        if k==num_neighbors - 1:
            volumes.append(distances[-1]**-3)
            k = 0
        else:
            k = k + 1

    avg_distance = np.mean(distances)

    plt.figure()
    plt.title('Nearest Neighbor Volumes')
    plt.hist(np.log(volumes))
    plt.xlabel('log density (Mpc)')
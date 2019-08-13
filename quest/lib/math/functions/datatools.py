# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:41:12 2011

@author: thomas
"""
import copy

import numpy


def hist2func(x, H, binEdges):
    """
    Example:
    import numpy
    import matplotlib.pylab as p

    H = numpy.array([0,2,1])
    binEdges = numpy.array([0,5,10,15])

    x = numpy.linspace(-5, 20, 50)
    y = hist2func(x, H, binEdges)

    p.plot(x,y)
    p.show()
    """
    if type(x) == float or numpy.float:
        x = numpy.array([x])
    if type(x) == list:
        x = numpy.array(x).flatten()
    re = []
    if type(x) == numpy.ndarray:
        for xi in x.flatten():
            if xi>max(binEdges) or xi < min(binEdges):
                re.append(0.0)
            else:
                sel = numpy.where(xi<binEdges)
                re.append(H[sel[0][0]-1])
        if len(re) == 1:
            return re[0]
        else:
            return re
    else:
        raise TypeError('unknon input type')


def overlapping_region(dataset1, dataset2):
    x1, y1 = dataset1
    x2, y2 = dataset2

    # Sort the arrays in ascending order in x
    x1 = numpy.array(x1)
    y1 = numpy.array(y1)
    inds = x1.argsort()
    x1 = x1[inds]
    y1 = y1[inds]
    x1 = numpy.ma.array(x1)
    y1 = numpy.ma.array(y1)
    
    x2 = numpy.array(x2)
    y2 = numpy.array(y2)
    inds = x2.argsort()
    x2 = x2[inds]
    y2 = y2[inds]
    x2 = numpy.ma.array(x2)
    y2 = numpy.ma.array(y2)

    # Calculate range for adjustment of spacing
    rng = (max(min(x1), min(x2)), min(max(x1), max(x2)))
    
    # mask the irrelevant parts
    m1l = x1.data < rng[0]
    m1u = x1.data > rng[1]
    m1 = m1l + m1u
    x1.mask = m1
    y1.mask = m1
    
    m2l = x2.data < rng[0]
    m2u = x2.data > rng[1]
    m2 = m2l + m2u
    x2.mask = m2
    y2.mask = m2
    
    # create new lists: overlap - o and rest -r
    # overlap
    ox1 = x1.compressed()
    oy1 = y1.compressed()
    ox2 = x2.compressed()
    oy2 = y2.compressed()
    
    return (ox1, oy1), (ox2, oy2)


def align_x_spacing(dataset1, dataset2, method = 'linear-close'):
    (ox1, oy1), (ox2, oy2) = dataset1, dataset2
    #Assume that data is more or less equaliy spaced
    # t- template array, r - rescale array
    if len(ox1) <len(ox2):
        tx = ox1
        ty = oy1
        rx = ox2
        ry = oy2
        cm = "r2"
    else:
        tx = ox2
        ty = oy2
        rx = ox1
        ry = oy1
        cm = "r1"

    # Create of template-arrays as new arrays - n
    nx = copy.deepcopy(tx)
    ny = copy.deepcopy(ty)
 
    if method == 'linear-close':
        j = 0 # counter for tx
        ry1 = ry[0]
        rx1 = rx[0]
        for i, rxi in enumerate(rx):
            if j < len(tx)-1:               
                if rxi > tx[j]:
                    rx2 = rxi
                    ry2 = ry[i]
                    if ry2-ry1 != 0 and rx1-rx2 != 0:
                        m = (ry2-ry1)/(rx1-rx2)
                        ny[j] = m*tx[j] + ry1 - m * rx1
                    else:
                        ny[j] = ry1
                    j += 1
                elif rxi == tx[j]:
                    ny[j] = ry[i]
                    j += 1
                else:
                    ry1 = ry[i]
                    rx1 = rx[i]
            else:
                ny[j] = ry[i:].means()
                                        
    if cm == "r1":
        rx1 = nx
        ry1 = ny
        rx2 = tx
        ry2 = ty
    elif cm == "r2":
        rx2 = nx
        ry2 = ny
        rx1 = tx
        ry1 = ty            
    return ((rx1, ry1), (rx2, ry2))


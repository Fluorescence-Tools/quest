import copy

import numpy

import _special


def hist2func(x, H, binEdges):
    """
    Extrapolation of a histogram to a new x-axis. Here the parameter x can be any
    numpy array as return value another array is obtained containing the values
    of the histogram at the given x-values. This function may be useful if the spacing
    of a histogram has to be changed.

    :param x: array
    :param H: array
        Histogram values
    :param binEdges: array
        The bin edges of the histogram
    :return: A list of the same size as the parameter x

    Examples
    --------

    >>> H = np.array([0,2,1])
    >>> binEdges = np.array([0,5,10,15])

    >>> x = np.linspace(-5, 20, 17)
    >>> x
    array([ -5.    ,  -3.4375,  -1.875 ,  -0.3125,   1.25  ,   2.8125,
             4.375 ,   5.9375,   7.5   ,   9.0625,  10.625 ,  12.1875,
            13.75  ,  15.3125,  16.875 ,  18.4375,  20.    ])
    >>> y = hist2func(x, H, binEdges)
    [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0.0, 0.0, 0.0, 0.0]
    """

    if type(x) == float or numpy.float:
        x = numpy.array([x])
    if type(x) == list:
        x = numpy.array(x).flatten()
    re = []
    if type(x) == numpy.ndarray:
        for xi in x.flatten():
            if xi > max(binEdges) or xi < min(binEdges):
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
    """

    :param dataset1: tuple
        The tuple should consist of the x and y values. Whereas the x and y values have to be a numpy array.
    :param dataset2: tuple
        The tuple should consist of the x and y values. Whereas the x and y values have to be a numpy array.
    :return: Two tuples
    """
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
                ny[j] = ry[i:].mean_xyz()
                                        
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


if __name__ == "__main__":
    print("Test align_x_spacing")
    import matplotlib.pylab as p
    x1 = numpy.linspace(-1, 12, 10)
    y1 = numpy.cos(x1)
    a1 = (x1, y1)
    
    x2 = numpy.linspace(-1, 12, 11)
    y2 = numpy.sin(x2)
    a2 = (x2, y2)
    
    (rx1, ry1), (rx2, ry2) = align_x_spacing(a1, a2)
    p.plot(x1, y1, 'r')
    p.plot(rx1, ry1, 'k')
    p.plot(x2, y2, 'g')
    p.plot(rx2, ry2, 'b')
    p.show()
    
    print("Test overlay")
    import matplotlib.pylab as p
    x1 = numpy.linspace(-5, 5, 10)
    y1 = numpy.sin(x1)
    a1 = (x1, y1)
    
    x2 = numpy.linspace(0, 10, 10)
    y2 = numpy.sin(x2)
    a2 = (x2, y2)
    
    (rx1, ry1), (rx2, ry2) = overlapping_region(a1, a2)
    print(ry1)
    print(ry2)

    p.plot(x1, y1, 'r')
    p.plot(rx1, ry1, 'k')
    p.plot(x2, y2, 'g')
    p.plot(rx2, ry2, 'b')
    p.show()    


def binCount(data, binWidth=16, binMin=0, binMax=4095):
    """
    Count number of occurrences of each value in array of non-negative ints.

    :param data: array_like
        1-dimensional input array
    :param binWidth:
    :param binMin:
    :param binMax:
    :return: As return values a numpy array with the bins and a array containing the counts is obtained.
    """
    return _special.binCount(data, binWidth, binMin, binMax)


def histogram1D(pos, data=None, nbPt=100):
    """
    Calculates histogram of pos weighted by weights.

    :param pos: 2Theta array
    :param weights: array with intensities
    :param bins: number of output bins
    :param pixelSize_in_Pos: size of a pixels in 2theta
    :param nthread: maximum number of thread to use. By default: maximum available.
        One can also limit this with OMP_NUM_THREADS environment variable

    :return 2theta, I, weighted histogram, raw histogram

    https://github.com/kif/pyFAI/blob/master/src/histogram.pyx
    """
    return _special.histogram1D(pos, data, nbPt)

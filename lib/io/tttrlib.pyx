import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, int64_t, uint_fast32_t, uint_fast16_t
cimport cython

#http://stackoverflow.com/questions/3046305/simple-wrapping-of-c-code-with-cython
@cython.boundscheck(False)
def beckerMerged(char[:] b):
    cdef unsigned long length = (b.shape[0] - 4) / 4
    cdef np.ndarray[np.uint64_t, ndim=1] event = np.zeros(length, dtype=np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] mt = np.zeros(length, dtype=np.uint64)
    cdef np.ndarray[np.uint32_t, ndim=1] tac = np.zeros(length, dtype=np.uint32)
    cdef np.ndarray[np.uint8_t, ndim=1] can = np.zeros(length, dtype=np.uint8)

    cdef unsigned int g, i
    cdef unsigned char b0, b1, b2, b3
    cdef unsigned char inv, mtov
    cdef unsigned long ov, ovfl
    ov, g, mtov, inv = 0, 0, 0, 0
    for i in range(1, length):
        b3 = b[4*i+3]
        inv = (b3 & 128) >> 7
        mtov = (b3 & 64) >> 6
        b0, b1, b2 = b[4*i], b[4*i+1], b[4*i+2]
        if (inv==0) and (mtov==0):
            event[g]=g
            tac[g]=((b3 & 0x0F)<<8 | b2)
            ovfl= ov*4096;
            mt[g]= ((b1 & 15)<<8 | b0) + ovfl
            can[g] = ((b1 & 0xF0) >> 4)
            g += 1
        else:
            if (inv==0) and (mtov==1):
                ov += 1
                event[g]=g
                tac[g]=((b3 & 0x0F)<<8 | b2)
                ovfl= ov*4096
                mt[g]= ((b1 & 15)<<8 | b0) + ovfl
                can[g] = ((b1 & 0xF0) >> 4)
                g += 1
            else:
                if (inv==1) and (mtov==1):
                    ov += ((b3 & 15)<<24) | ((b2<<16) | ((b1<<8) | b0))
    return g, mt, 4095 - tac, can

@cython.boundscheck(False)
def ht3(char[:] b):
    cdef unsigned long length = (b.shape[0]) / 4
    cdef np.ndarray[np.uint64_t, ndim=1] event = np.zeros(length, dtype=np.uint64)
    cdef np.ndarray[np.uint64_t, ndim=1] mt = np.zeros(length, dtype=np.uint64)
    cdef np.ndarray[np.uint32_t, ndim=1] tac = np.zeros(length, dtype=np.uint32)
    cdef np.ndarray[np.uint8_t, ndim=1] can = np.zeros(length, dtype=np.uint8)

    cdef unsigned long long i
    cdef unsigned long long g

    cdef unsigned char b0=0, b1=0, b2=0, b3=0
    cdef unsigned char inv=0
    cdef long long ovfl = 0

    ov, g, inv = 0, 0, 0
    for i in range(length):
        b3=b[4*i+3]
        inv= (b3 & 254) >> 1
        b0=b[4*i]
        b1=b[4*i+1]
        b2=b[4*i+2]
        if inv == 127:
            ov += 1
        else:
            event[g]=g
            tac[g] = ((b3 & 1)<<14 | b2<<6 |(b1 & 252)>>2)
            ovfl = ov * 1024
            mt[g]= (((b1 & 3)<<8 |b0 ) + ovfl)
            can[g] = ((b3 & 254) >> 1)
            g += 1
            if can[g] > 64:
                can[g] -= 64
    return g, mt, tac, can


def iss(np.ndarray[np.uint8_t, ndim=1] data):
    cdef unsigned int i, j, k
    cdef unsigned int ch1, ch2
    cdef unsigned int step, offset, sizeFlag, phMode
    cdef unsigned int frequency

    # CHANNEL PHOTON MODE (first 2 bytes)
    # in brackets int values
    # H (72)one channel time mode, h (104) one channel photon mode
    # X (88) two channel time mode, x (120) two channel photon mode
    step = 1 if (data[1] == 72) or (data[1] == 104) else 2
    phMode = 0 if (data[1] == 72) or (data[1] == 88) else 1
    print "Ph-Mode (0/1):\t%s" % phMode
    print "Nbr. Ch.:\t%s" % step

    #Data is saved as 0: 16-bit or 1: 32-bit
    # TODO cython typed memory view
    if data[10]:
        print "Datasize: 16bit"
        b = data.view(dtype=np.uint16)
        offset = 256 / 2
    else:
        print "Datasize: 32bit"
        b = data.view(dtype=np.uint32)
        offset = 256 / 4
    print b
    cdef unsigned long length = (b.shape[0])
    cdef np.ndarray[np.uint64_t, ndim=1] mt = np.zeros(length, dtype=np.uint64)
    cdef np.ndarray[np.uint32_t, ndim=1] tac = np.zeros(length, dtype=np.uint32)
    cdef np.ndarray[np.uint8_t, ndim=1] can = np.zeros(length, dtype=np.uint8)

    if step == 1:
        k = 1 if phMode else 0
        for i in range(offset, b.shape[0]):
            ch1 = b[i]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
    elif step == 2:
        k = 0
        for i in range(offset, b.shape[0], 2):
            ch1 = b[i]
            ch2 = b[i + 1]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
                mt[k] = mt[k-1] + ch2
                can[k] = 1
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
                for j in range(ch2):
                    mt[k] = i
                    can[k] = 1
                    k += 1
    return k, mt[:k], tac[:k], can[:k]


import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, log


cdef extern from "mtrandom.h":
    cdef cppclass MTrandoms:
        void seedMT()
        double random0i1i() nogil
        double random0i1e() nogil

cdef MTrandoms rmt
rmt.seedMT()

cdef double eps = 244.14062E-6

@cython.cdivision(True)
cdef long random_c(long max) nogil:
    return rand() % max

@cython.cdivision(True)
cdef double ranf() nogil:
    return <double> rmt.random0i1e()
    #return <double> rand() / RAND_MAX


@cython.cdivision(True)
@cython.boundscheck(False)
def simulate_photon_trace(unsigned long n_ph, char[:] collided,
                            double quenching_prob=0.5, double t_step=0.01, double tau0=0.25):
    cdef np.ndarray[dtype=np.float64_t, ndim=1] dts = np.zeros(n_ph, dtype=np.float64)
    cdef np.ndarray[dtype=np.uint8_t, ndim=1] phs = np.zeros(n_ph, dtype=np.uint8)
    cdef double em_p = t_step / tau0
    cdef long n_frames = collided.shape[0]
    cdef long[:] shift_nbrs = np.random.randint(0, n_frames / 2, n_ph)
    cdef unsigned long n_step
    cdef unsigned long n_collision

    cdef unsigned long i, j, shift_nbr

    if quenching_prob == 0:
        for i in range(n_ph):
            dts[i] = log(1./(ranf() + eps)) * tau0
            phs[i] = 1

    elif quenching_prob >= 1.0:

        for i in range(n_ph):
            # Look-up when photon was emitted
            dts[i] = log(1./(ranf() + eps)) * tau0
            phs[i] = 1
            n_step = <unsigned long> (dts[i] / t_step)
            # if molecule is quenched within this time-difference
            # no photon is emitted.
            shift_nbr = shift_nbrs[i]
            for j in range(shift_nbr, shift_nbr + n_step):
                if collided[j]:
                    dts[i] = 0
                    phs[i] = 0
                    break

    elif 0 < quenching_prob < 1:
        # Radiation boundary condition
        for i in range(n_ph):
            # Look-up when photon was emitted
            dts[i] = log(1./(ranf() + eps)) * tau0
            phs[i] = 1

            # count the number of collisions within dt
            n_step = <unsigned long> (dts[i] / t_step)
            shift_nbr = shift_nbrs[i]
            for j in range(shift_nbr, shift_nbr + n_step):
                if collided[j] and ranf() < quenching_prob:
                    dts[i] = 0
                    phs[i] = 0
                    break
    return dts, phs


@cython.cdivision(True)
@cython.boundscheck(False)
def simulate_photon_trace_kQ(unsigned long n_ph, char[:] collided,
                            double kQ=0.5, double t_step=0.01, double tau0=0.25):
    cdef np.ndarray[dtype=np.float64_t, ndim=1] dts = np.zeros(n_ph, dtype=np.float64)
    cdef np.ndarray[dtype=np.uint8_t, ndim=1] phs = np.zeros(n_ph, dtype=np.uint8)
    cdef double em_p = t_step / tau0
    cdef long n_frames = collided.shape[0]
    cdef long[:] shift_nbrs = np.random.randint(0, n_frames / 2, n_ph)
    cdef unsigned long n_step
    cdef unsigned long n_collision
    cdef double p_quench =  kQ * t_step
    cdef unsigned long i, j, shift_nbr

    if kQ == 0:
        for i in range(n_ph):
            dts[i] = log(1./(ranf() + eps)) * tau0
            phs[i] = 1
    else:
        # Radiation boundary condition
        for i in range(n_ph):
            # Look-up when photon was emitted
            dts[i] = log(1./(ranf() + eps)) * tau0
            phs[i] = 1

            # count the number of collisions within dt
            n_step = <unsigned long> (dts[i] / t_step)
            shift_nbr = shift_nbrs[i]
            n_collision = 0
            for j in range(shift_nbr, shift_nbr + n_step):
                if collided[j] and ranf() < p_quench:
                    dts[i] = 0
                    phs[i] = 0
                    break

    return dts, phs


@cython.cdivision(True)
@cython.boundscheck(False)
def simulate_photon_trace_rate(unsigned long n_ph, float[:] kQ, double t_step=0.01, double tau0=0.25):
    cdef np.ndarray[dtype=np.float64_t, ndim=1] dts = np.zeros(n_ph, dtype=np.float64)
    cdef np.ndarray[dtype=np.uint8_t, ndim=1] phs = np.zeros(n_ph, dtype=np.uint8)
    cdef double em_p = t_step / tau0
    cdef long n_frames = kQ.shape[0]
    cdef long[:] shift_nbrs = np.random.randint(0, n_frames / 2, n_ph)
    cdef unsigned long n_step
    cdef unsigned long n_collision

    cdef unsigned long i, j, shift_nbr

    # Radiation boundary condition
    for i in range(n_ph):
        # Look-up when photon was emitted
        dts[i] = log(1./(ranf() + eps)) * tau0
        phs[i] = 1

        # count the number of collisions within dt
        n_step = <unsigned long> (dts[i] / t_step)
        shift_nbr = shift_nbrs[i]
        for j in range(shift_nbr, shift_nbr + n_step):
            if ranf() < kQ[j] * t_step:
                dts[i] = 0
                phs[i] = 0
                break
    return dts, phs



'''
    A simple rejection sampling script for custom statistical distributions
    Author: Damien Robertson - robertsondamien@gmail.com

    Usage:
      $ python rejection.py -h for help and basic instruction

'''
from __future__ import print_function

print(__doc__)

try:
    import numpy as np
except ImportError:
    print("Numpy not installed, try - pip install numpy")
    import sys
    sys.exit()

#
#   CUSTOM PDF CLASS ##########################################################
#

class custom_df:

    '''
        Custom_df class defines the custom distribution function from 
        the user.

            __init__ : Define the piecewise components of the distribution
                       along with any other parameters you need.

            df       : Distribution function - accepts a n-length vector only.

            sample   : Sample the distribution function - accepts a single 
                       float only.

    '''

    def __init__(self):

        self.lgn = lambda m, a, mc, s: a * np.exp(-(np.log10(m) - \
                                       np.log10(mc))**2.0 / (2*s**2.0))

        self.pwr = lambda m, a, s: a*m**(-s)

        self.p = [0.093, 0.2, 0.55, 0.0415, 1.35]

    def df(self,x):

        assert((isinstance(x, float) != True)), \
            "Distribution not found - passed values are a float - use .sample"
        
        #
        # Define your custom distribution here that accepts a vector.
        #

        idx = np.min(np.where(x >= 1.0)[0])
        n = np.empty(len(x))
        n[:idx] = self.lgn(x[:idx], self.p[0], self.p[1], self.p[2])
        n[idx:] = self.pwr(x[idx:], self.p[3], self.p[4])

        return n

    def sample(self,x):

        assert(isinstance(x, float)), \
            "Sample not taken - passed values are a tuple - use .df"
        
        #
        # Define how to sample the distribution given a single value
        #

        if x > 1.0:

            dn = self.pwr(x, self.p[3], self.p[4])

        else:

            dn = self.lgn(x, self.p[0], self.p[1], self.p[2])

        return dn

#
#   END CUSTOM PDF CLASS ######################################################
#
#
#
#    MAIN PROGRAM
#

import argparse

INPUT = argparse.ArgumentParser(description='rejection.py user parameters')

#
#   USER PARAMETER INPUT
#     REQUIRED PARAMETERS:
#       a   = lower limit of distribution
#       b   = upper limit of distribution
#

INPUT.add_argument('a', metavar='a', type=float, 
                    nargs=1, help='lower limit of distribution')
INPUT.add_argument('b', metavar='b', type=float, 
                    nargs=1, help='upper limit of distribution')

#
#   USER PARAMETER INPUT
#     OPTIONAL PARAMETERS:
#       plot         = True or False
#       logspace     = True or False
#       output       = True or False
#       verbose      = True or False
#       exploratory  = True or False
#

INPUT.add_argument('-np', '--no-plot', dest='noplot', action='store_false',
                   help='Do not produce plot')

INPUT.add_argument('-lg', '--logspace', dest='logspace', action='store_true',
                   help='Independent axis in evenly sampled in logspace')

INPUT.add_argument('-o', '--output', dest='output', action='store_true',
                   help='Write output file')

INPUT.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                   help='Tells you more information during run')

INPUT.add_argument('-ex', '--exploratory', dest='explore', action='store_true',
                   help='No sampling - plots distribution for exploration')

OPTS = INPUT.parse_args()

#
#   USER PARAMETER CHECK AND READOUT
#     This section will fail if:
#       Parameters 'a' is not less than 'b'
#

assert OPTS.a[0] < OPTS.b[0], "INPUT ERROR - CHECK DISTRIBUTION RANGE"

if OPTS.verbose:

    print("\nPYTHON SCRIPT: rejection")
    print()
    print("INPUT DISTRIBUTION LIMIT a) :", OPTS.a[0])
    print("INPUT DISTRIBUTION LIMIT b) :", OPTS.b[0])
    print("LOGSPACE                    :", OPTS.logspace)

#
#   Main rejection sampling loop
#
#
#
#   Checks if user specifies logspace at execution
#

if OPTS.logspace:

    X = np.logspace(OPTS.a[0], OPTS.b[0], 1000) 

else:

    X = np.linspace(OPTS.a[0], OPTS.b[0], 1000)

#
#   Assigns my_pdf to custom_pdf object
# 

my_df = custom_df()

#
#   
#

if OPTS.explore:

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed, try - pip install matplotlib")
        import sys
        sys.exit()

    plt.figure()
    plt.plot(X, my_df.df(X), 'k')

    if OPTS.logspace:

        plt.xscale('log')

    plt.show()

else:

#
#   box_roof sets the limit of the box function to sample from.
#
    box_roof = np.max(my_df.df(X)) * 1.1

#
#   Rejection sampling loop:
#       N : decided how many samples to pull from 'my_pdf'
#       P : are the random pulls that pass the rejection criteria
#       F : are the ranfom pulls that fail the rejection criteria
#       df_sample : is the array of random samples that have passed, P, the 
#                   selection criteria.
#       tl_sample : is the array of all random samples that have passed, P,
#                   and failed, F.
#

    N = 10**4
    P = 0
    F = 0
    df_sample = np.empty(N)
    tl_sample = np.array([])

    while P < N:

        T = 10**np.random.uniform(OPTS.a[0], OPTS.b[0])
        tl_sample = np.append(tl_sample, T)
        U = np.random.uniform(0, 1)

        if (U*box_roof) <= my_df.sample(T):

            df_sample[P] = T
            P += 1
        
        else:

            F += 1

    if OPTS.verbose:

        print("PASS/FAIL RATE              : %.3f" %(P/F))

#
#   CHECK IF OUTPUT WRITE IS TRUE
#

    if OPTS.output:

        print("\nWriting DCF output file to: rejection_output.csv")
        np.savetxt('rejection_output.csv', df_sample, fmt="%.6f", \
                    delimiter=',')

#
#   PLOT RESULTS
#     No brainer - plots the results. If the user wishes to suppress the plot
#     one should use the -np or --no-plot flag on the command line.
#
#   Requires python module matplotlib.
#

    if OPTS.noplot:

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed, try - pip install matplotlib")
            import sys
            sys.exit()

        plt.figure(0)

        if OPTS.logspace:
            plt.hist(np.log10(df_sample), bins=150, \
                     range=(OPTS.a[0], OPTS.b[0]), histtype='step', color='r')
            plt.hist(np.log10(tl_sample), bins=150, \
                     range=(OPTS.a[0], OPTS.b[0]), histtype='step', color='k')
            plt.xlabel("Log(x)")
        else:
            plt.hist(df_sample, bins=150, range=(OPTS.a[0], OPTS.b[0]), \
                     histtype='step', color='r')
            plt.hist(tl_sample, bins=150, range=(OPTS.a[0], OPTS.b[0]), \
                     histtype='step', color='k')
            plt.xlabel("x")

        plt.ylabel("N")
        plt.show()

#
#   END
#

'''
Calculate electron bridge rates for 229-thorium.
Wave functions are expanded in spherical harmonics on a radial grid.
Angular parts can be analytically solved, radial parts are integrated.
The photon multipolarity is dominantly E1,
while the electronic and nuclear transition is a mixture of E2 and M1
see 10.1103/PhysRevA.103.053120
Workflow:
    1. integrals.py
    2. tmis.py
    3. rates.py
TODO: Conceptually mixed real and complex spherical harmonics
'''
import os
from fileio import read_pickle
from integrals import inters_initial_final
from config import CONFIG


def construct_eb_loop():
    '''
    Create the rate iterator
    '''
    loop = [("T_E2", component) for component in range(-2, 2+1)] +\
           [("T_M1", component) for component in range(-1, 1+1)]
    return loop


if __name__ == "__main__":
    TMIS = read_pickle(os.path.join(CONFIG["indir"], "tmis.pckl"))
    INTERS, INITIAL, FINAL, DATA = inters_initial_final()
    ISO = 8.338

    Q_E1 = [complex(0., 0.) for _ in range(3)]
    Q_LOOP = construct_eb_loop()
    for E1_Q in range(-1, 1+1):
        Q_SUM = complex(0., 0.)
        for K_TM, Q_TM in Q_LOOP:
            INTER_SUM = complex(0., 0.)
            for INTER in INTERS:
                NUMER = TMIS[(*FINAL, "Q_E1", E1_Q, *INTER)] *\
                    TMIS[(*INTER, K_TM, Q_TM, *INITIAL)]
                DENUM = DATA[INITIAL]["Energy"] - DATA[INTER]["Energy"] - ISO
                INTER_SUM += NUMER/DENUM
                NUMER = TMIS[(*FINAL, K_TM, Q_TM, *INTER)] *\
                    TMIS[(*INTER, "Q_E1", E1_Q, *INITIAL)]
                DENUM = DATA[FINAL]["Energy"] - DATA[INTER]["Energy"] + ISO
                INTER_SUM += NUMER/DENUM
            Q_SUM += INTER_SUM * (-1)**Q_TM  # TODO: * nuclear
        Q_E1[E1_Q] = Q_SUM
    RATE = sum(comp.conjugate() * comp for comp in Q_E1).real


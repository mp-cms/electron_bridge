'''
Construct transition moment integrals. Supported Operators
    - Q_E1
    - T_E2
    - T_M1
'''
import os
import math
from tqdm import tqdm
from fileio import read_pickle, save_pickle
from config import CONFIG
from integrals import inters_initial_final, construct_lm_loop


def f_qe1_i(radials, gaunts, final, initial, t_m):
    '''
    Calculates <f|Q_E1|i>, where Q_EI = -r
    '''
    assert -1 <= t_m <= 1
    if final[2] != initial[2]:
        return complex(0., 0.)
    lm_loop = construct_lm_loop(CONFIG["lmax"])
    sum_lm = complex(0., 0.)
    for (lf_q, mf_q, li_q, mi_q) in lm_loop:
        angular_part = gaunts[(lf_q, 1, li_q, mf_q, t_m, mi_q)]
        if angular_part != 0.:
            radial_part = radials[(*final, lf_q, mf_q, 1,
                                   *initial, li_q, mi_q)]
            sum_lm += angular_part * radial_part  # TODO: check (-1)**mf_q
    return -math.sqrt(4.*math.pi/3.) * sum_lm


def f_te2_i(radials, gaunts, final, initial, t_m):
    '''
    Calculates <f|T_E2|i>, where T_E2 = -1/r³ sqrt(4π/5)Y_{2,q}
    '''
    assert -2 <= t_m <= 2
    if final[2] != initial[2]:
        return complex(0., 0.)
    lm_loop = construct_lm_loop(CONFIG["lmax"])
    sum_lm = complex(0., 0.)
    for (lf_q, mf_q, li_q, mi_q) in lm_loop:
        angular_part = gaunts[(lf_q, 2, li_q, mf_q, t_m, mi_q)]
        if angular_part != 0.:
            radial_part = radials[(*final, lf_q, mf_q, -3,
                                   *initial, li_q, mi_q)]
            sum_lm += angular_part * radial_part * (-1)**mf_q
    return -math.sqrt(4.*math.pi/5.) * sum_lm


def f_tm1_i(radials, gaunts, final, initial, t_m):
    '''
    Calculates <f|T_M1|i>,
    where T_M1 = 1/c[l/r³ - σ/(2r³) + 3r(r·σ)/2r⁵ + 4πσδ(r)/3]
    '''
    assert -1 <= t_m <= 1
    alpha = 1.  # TODO: Correct factor
    c_0 = 137.  # Speed of light in atomic units
    origin = 1.  # TODO: get origin
    lm_loop = construct_lm_loop(CONFIG["lmax"])
    sum_lm = complex(0., 0.)
    for (lf_q, mf_q, li_q, mi_q) in lm_loop:
        if final[2] == initial[2]:
            angular_part = gaunts[(lf_q, 2, li_q, mf_q, t_m, mi_q)]
            radial_part = complex(0., 0.)  # TODO: This could be dangerous
            factor = 0.
            if angular_part != 0:
                radial_part = radials[(*final, lf_q, mf_q, -3,
                                       *initial, li_q, mi_q)]
                factor = 4*math.pi*final[2]*alpha*t_m/c_0
                if t_m == 0:
                    factor *= 1/math.sqrt(5*math.pi)
                sum_lm += factor * angular_part * radial_part

            if lf_q == li_q and mf_q == mi_q:
                if t_m in [-1, 1] and abs(mf_q) != lf_q:
                    radial_part = radials[(*final, lf_q, mf_q + t_m, -3,
                                           *initial, li_q, mi_q)]
                    factor = 1./(math.sqrt(2)*c_0) *\
                        math.sqrt((lf_q - mf_q)*(lf_q + mf_q + 1))
                if t_m == 0:
                    factor = 1./c_0 * (mf_q + 1./(4*math.pi))
                sum_lm += radial_part * factor + origin
        else:
            q_m = t_m + math.copysign(1, initial[2])
            angular_part = gaunts[(lf_q, 2, li_q,
                                   mf_q, q_m, mi_q)]
            if angular_part != 0.:
                factor = 1.  # TODO: Correct factor
                radial_part = radials[(*final, lf_q, mf_q, -3,
                                       *initial, li_q, mi_q)]
                sum_lm += radial_part * angular_part * factor
            if q_m == 0:
                sum_lm += math.copysign(1, initial[2]) * origin
    return sum_lm


if __name__ == "__main__":
    print()
    print("    ELECTRON BRIDGE TRANSITIONS")
    print("    ===========================")
    RADIALS = read_pickle(os.path.join(CONFIG["indir"], "radial.pckl"))
    GAUNTS = read_pickle(f"gaunt_{CONFIG['lmax']}.pckl")
    INTERS, INITIAL, FINAL, _ = inters_initial_final()

    TMIS = {}
    print("::: Calculating Q_E1")
    for INTER in tqdm(INTERS):
        for T_M in range(-1, 1+1):
            TMIS[(*FINAL, "Q_E1", T_M, *INTER)] =\
                f_qe1_i(RADIALS, GAUNTS, FINAL, INTER, T_M)
            TMIS[(*INTER, "Q_E1", T_M, *INITIAL)] =\
                f_qe1_i(RADIALS, GAUNTS, INTER, INITIAL, T_M)
    print("::: Calculating T_E2")
    for INTER in tqdm(INTERS):
        for T_M in range(-2, 2+1):
            TMIS[(*FINAL, "T_E2", T_M, *INTER)] =\
                f_te2_i(RADIALS, GAUNTS, FINAL, INTER, T_M)
            TMIS[(*INTER, "T_E2", T_M, *INITIAL)] =\
                f_te2_i(RADIALS, GAUNTS, INTER, INITIAL, T_M)
    print("::: Calculating T_M1")
    for INTER in tqdm(INTERS):
        for T_M in range(-1, 1+1):
            TMIS[(*FINAL, "T_M1", T_M, *INTER)] =\
                f_tm1_i(RADIALS, GAUNTS, FINAL, INTER, T_M)
            TMIS[(*INTER, "T_M1", T_M, *INITIAL)] =\
                f_tm1_i(RADIALS, GAUNTS, INTER, INITIAL, T_M)
    save_pickle(os.path.join(CONFIG["indir"], "tmis.pckl"), TMIS)
    print("::: EXIT\n")

'''
Calculate angular and radial integrals for
transition moment integral components
'''
from multiprocessing import Pool, cpu_count
import re
import os
# from time import time

import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
# from IPython import embed
from sympy.physics.wigner import real_gaunt

import fileio
from config import CONFIG


def merge_info(info, alm):
    '''
    Make a dictionay of band information and expansion coefficients
    '''
    data = {"band": info[0],
            "energy": info[1],
            "occupation": info[2],
            "localization": info[3],
            "coefficients": alm}
    return data


def trapezoidal_integration(integrand, grid):
    '''
    Integrate the integrand on the grid using the trapezoidal rule
    '''
    integral = 0.
    for grid_i, _ in enumerate(grid):
        if grid_i == 0:
            continue
        integral += (integrand[grid_i-1] + integrand[grid_i])/2. *\
                    (grid[grid_i] - grid[grid_i-1])
    return integral


def calculate_gaunt(lmax):
    '''
    Calculate all necessary Gaunt coefficients
    '''
    return {(l1_q, t_l, l2_q, m1_q, t_m, m2_q):
            complex(real_gaunt(l1_q, t_l, l2_q, m1_q, t_m, m2_q, prec=16))
            for l1_q in range(0, lmax+1)
            for t_l in range(1, 2+1)
            for l2_q in range(0, lmax+1)
            for m1_q in range(-l1_q, l1_q+1)
            for t_m in range(-t_l, t_l+1)
            for m2_q in range(-l2_q, l2_q+1)}


def get_gaunt_coeffs(lmax):
    '''
    Load or calculate all Gaunt coefficients up to lmax
    Performance could be drastically increased by exploiting symmetry
    '''
    fname = f"gaunt_{lmax}.pckl"
    if os.path.exists(fname):
        return fileio.read_pickle(fname)
    print("::: Calculating Gaunt coefficients")
    gaunts = calculate_gaunt(lmax)
    fileio.save_pickle(fname, gaunts)
    return gaunts


def mark_fi(data, config):
    '''
    label final and initial states
    '''
    initial = (config["Initial"]["Index"],
               config["Initial"]["K-Point"],
               config["Initial"]["Spin"])
    final = (config["Final"]["Index"],
             config["Final"]["K-Point"],
             config["Final"]["Spin"])
    data[initial]["Initial"] = True
    data[initial]["Occupation"] = False  # not true but practical
    data[final]["Final"] = True
    data[final]["Occupation"] = False
    return data, initial, final


def radial_integral(alm_f, alm_i, grid, r_exp):
    '''
    Calculate a single radial integral
    '''
    integrand = [np.conj(alm_rf) * alm_ri * grid_r**r_exp
                 for grid_r, alm_rf, alm_ri
                 in zip(grid, alm_f, alm_i)]
    return trapezoidal_integration(integrand, grid)


def construct_lm_loop(lmax):
    '''
    Construct the {l, m, l', m'} iterable
    '''
    lm_loop = [(lf_q, mf_q, li_q, mi_q)
               for lf_q in range(-lmax, lmax+1)
               for mf_q in range(-lf_q, lf_q+1)
               for li_q in range(-lmax, lmax+1)
               for mi_q in range(-li_q, li_q+1)]
    return lm_loop


def construct_lm_loop_short(lmax):
    '''
    Construct the {l, m} iterable
    '''
    lm_loop = [(l_q, m_q)
               for l_q in range(-lmax, lmax+1)
               for m_q in range(-l_q, l_q+1)]
    return lm_loop


def calc_radial(wfs, gaunts, final, initial, grid, q_m, r_exp):
    '''
    Calculate radial integrals times r^{+1}
    '''
    radials = {}
    lm_loop = construct_lm_loop(CONFIG["lmax"])
    for lf_q, mf_q, li_q, mi_q in lm_loop:
        all_gaunt = [gaunts[(lf_q, q_m, li_q, mf_q, t_m, mi_q)] == 0
                     for t_m in range(-q_m, q_m+1)]
        if all(all_gaunt):
            continue
        alm_i = [alm_ri[(li_q, mi_q)] for alm_ri in wfs[initial]]
        alm_f = [alm_rf[(lf_q, mf_q)] for alm_rf in wfs[final]]
        radials[(*final, lf_q, mf_q, r_exp, *initial, li_q, mi_q)] =\
            radial_integral(alm_f, alm_i, grid, 2+r_exp)
    return radials


def calc_radial_incr(wfs, gaunts, final, initial, grid, r_exp, incr):
    '''
    Calculate radial integrals times r^{+1}
    '''
    radials = {}
    lm_loop = construct_lm_loop_short(CONFIG["lmax"])
    for l_q, m_q in lm_loop:
        if abs(m_q) == l_q:
            continue
        alm_i = [alm_ri[(l_q, m_q)] for alm_ri in wfs[initial]]
        alm_f = [alm_rf[(l_q, m_q+incr)] for alm_rf in wfs[final]]
        radials[(*final, l_q, m_q+incr, r_exp, *initial, l_q, m_q)] =\
            radial_integral(alm_f, alm_i, grid, 2+r_exp)
    return radials


def get_radial_integrals(wfs, gaunts, grid, fname="radial.pckl"):
    '''
    Calculate all necessary radial integrals
    '''
    fname = os.path.join(CONFIG["indir"], "radial.pckl")
    # if os.path.exists(fname):
    #     print("::: Loading Radial Integrals")
    #     with open(fname, "rb") as stream:
    #         return pickle.load(stream)

    inters, initial, final, _ = inters_initial_final()
    print("::: Calculating Radial Integrals")
    radials = {}
    for inter in tqdm(inters):
        if inter[2] == initial[2] == final[2]:  # Spin component
            radials.update(calc_radial(wfs, gaunts, inter, initial, grid, 1, 1))
            radials.update(calc_radial(wfs, gaunts, final, inter, grid, 1, 1))
            radials.update(calc_radial_incr(wfs, gaunts, inter, initial, grid, -3, 1))
            radials.update(calc_radial_incr(wfs, gaunts, final, inter, grid, -3, 1))
            radials.update(calc_radial_incr(wfs, gaunts, inter, initial, grid, -3, -1))
            radials.update(calc_radial_incr(wfs, gaunts, final, inter, grid, -3, -1))
        radials.update(calc_radial(wfs, gaunts, inter, initial, grid, 2, -3))
        radials.update(calc_radial(wfs, gaunts, final, inter, grid, 2, -3))
    fileio.save_pickle(fname, radials)
    return radials


def inters_initial_final():
    '''
    Obtain the intermediates (inters), initial and final keys
    '''
    data = fileio.read_info(os.path.join(CONFIG["indir"], "info.yaml"),
                            CONFIG["K-Point"])
    data, initial, final = mark_fi(data, CONFIG)
    inters = [key for key in data if not data[key]["Occupation"]]
    return inters, initial, final, data


def print_setup():
    '''
    Prints information regarding the current setup
    '''


if __name__ == "__main__":
    print()
    print("    ELECTRON BRIDGE INTEGRALS")
    print("    =========================")
    assert CONFIG["Initial"]["K-Point"] == CONFIG["Final"]["K-Point"]
    GRID = fileio.read_grid(os.path.join(CONFIG["indir"], "r_grid.yaml"))
    REX = re.compile("^B[0-9]+_K[0-9]+_[UP|DN]{2}_L[0-9]+\.yaml$")
    WF_FILES = (FILE for FILE in os.listdir(CONFIG["indir"])
                if os.path.isfile(os.path.join(CONFIG["indir"], FILE)))
    WF_FILES = (os.path.join(CONFIG["indir"], FILE)
                for FILE in WF_FILES if REX.match(FILE))
    WFS = fileio.read_wfs(WF_FILES, l_spin=CONFIG["l_spin"], kpoint=1)
    GAUNTS = get_gaunt_coeffs(CONFIG["lmax"])
    RADIALS = get_radial_integrals(WFS, GAUNTS, GRID)
    print("::: EXIT\n")

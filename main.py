'''
Calculate electron bridge rates for 229-thorium.
Wave functions are expanded in spherical harmonics on a radial grid.
Angular parts can be analytically solved,  radial parts are integrated.
The photon multipolarity is dominantly E1,
while the electronic and nuclear transition is a mixture of E2 and M1
see 10.1103/PhysRevA.103.053120
TODO: Conceptually mixed real and complex spherical harmonics
TODO: More k-points
'''
import pickle
from multiprocessing import Pool, cpu_count

import yaml
import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
# from IPython import embed
from sympy import KroneckerDelta
from sympy.physics.wigner import real_gaunt
from scipy.constants import speed_of_light as c0


def read_yaml(fname):
    '''
    Read yaml file
    '''
    with open(fname, "r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def read_alm(band):
    '''
    Reads a yaml file containing expansion coefficients
    at each point in the radial grid
    '''
    lmax = 12
    fname = f"{band}_{lmax}.yaml"
    with open(fname, "r", encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=yaml.CSafeLoader)
    # data loads as list of dicts but the keys and values are just strings
    alm_band = [{(l_q, m_q): complex(0., 0.)
                 for l_q in range(0, lmax+1) for m_q in range(-l_q, l_q+1)}
                for _ in range(len(data))]
    for r_i, alm_data in enumerate(data):
        for key_str in alm_data.keys():

            l_q = int(key_str.split(',')[0][1:])
            m_q = int(key_str.split(',')[1][:-1])
            val = alm_data[key_str].replace(" ", "").replace("im", "j")
            alm_band[r_i][(l_q, m_q)] = complex(val)
    return alm_band


def extrapolate_nucleus(alm, grid):
    '''
    Extrapolate each alm coefficient to the nucleus
    Its radius is assumed to be grid[0]
    '''
    alm_0 = {(l_q, m_q): complex(0., 0.) for (l_q, m_q) in alm[0].keys()}
    coeffs = np.array([[np.log10(grid[1]), 1], [np.log10(grid[2]), 1]])
    ordinate = np.array([np.real(alm[0][(0, 0)]),
                         np.real(alm[1][(0, 0)])])
    k_real, d_real = np.linalg.solve(coeffs, ordinate)
    nucleus_real = k_real*np.log10(grid[0]) + d_real
    ordinate = np.array([np.imag(alm[0][(0, 0)]),
                         np.imag(alm[1][(0, 0)])])
    k_imag, d_imag = np.linalg.solve(coeffs, ordinate)
    nucleus_imag = k_imag*np.log10(grid[0]) + d_imag
    alm_0[(0, 0)] = complex(nucleus_real, nucleus_imag)
    alm.insert(0, alm_0)
    return alm


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


def l_plus(l_q, m_q):
    '''
    Application of l_+ Y_lm
    The spherical harmonic Y_lm+1 is omitted
    '''
    return -1/np.sqrt(2) * np.sqrt((l_q - m_q) * (l_q + m_q + 1))


def l_minus(l_q, m_q):
    '''
    Application of l_- Y_lm
    The spherical harmonic Y_lm-1 is omitted
    '''
    return 1/np.sqrt(2) * np.sqrt(l_q*(l_q + 1) - m_q*(m_q - 1))


def l_0(m_q):
    '''
    Application of l_0 Y_lm
    The spherical harmonic Y_lm is omitted
    '''
    return m_q


def f_qe1_i(alm_final, alm_initial, grid, transition_m):
    '''
    Calculates <f|Q_E1|i>, where Q_EI = -r
    '''
    assert -1 <= transition_m <= 1
    sum_lm = complex(0., 0.)
    for (li_q, mi_q) in alm_initial[0].keys():
        for (lf_q, mf_q) in alm_final[0].keys():
            angular_part = real_gaunt(li_q, lf_q, 1, mi_q, mf_q, transition_m,
                                      prec=16)
            if angular_part < 1e-10:
                continue
            integrand = [np.conj(alm_rf[(lf_q, mf_q)]) * alm_ri[(li_q, mi_q)] *
                         grid_r**3
                         for grid_r, alm_ri, alm_rf
                         in zip(grid, alm_initial, alm_final)]
            radial_part = trapezoidal_integration(integrand, grid)
            sum_lm += radial_part * angular_part
    return complex(np.sqrt(4.*np.pi/3.) * sum_lm)


def f_te2_i(alm_final, alm_initial, grid, transition_m):
    '''
    Calculates <f|T_E2|i>, where T_E2 = -1/r³ sqrt(4π/5)Y_{2,q}
    '''
    assert -2 <= transition_m <= 2
    sum_lm = complex(0., 0.)
    for (li_q, mi_q) in alm_initial[0].keys():
        for (lf_q, mf_q) in alm_final[0].keys():
            angular_part = real_gaunt(li_q, lf_q, 2, mi_q, mf_q, transition_m,
                                      prec=16)
            if angular_part < 1e-10:
                continue
            integrand = [np.conj(alm_rf[(lf_q, mf_q)]) * alm_ri[(li_q, mi_q)] *
                         1./grid_r
                         for grid_r, alm_ri, alm_rf
                         in zip(grid, alm_initial, alm_final)]
            radial_part = trapezoidal_integration(integrand, grid)
            sum_lm += radial_part * angular_part
    return complex(np.sqrt(4.*np.pi/5.) * sum_lm)


def f_tm1_i(alm_final, alm_initial, grid, transition_m):
    '''
    Calculates <f|T_M1|i>,
    where T_M1 = 1/c[l/r³ - σ/(2r³) + 3r(r·σ)/2r⁵ + 4πσδ(r)/3]
    '''
    assert -1 <= transition_m <= 1
    factor = np.sqrt(3./(20.*np.pi))
    m_s = 1./2.
    sum_lm = complex(0., 0.)
    for (li_q, mi_q) in alm_initial[0].keys():
        for (lf_q, mf_q) in alm_final[0].keys():
            if transition_m == 1:
                angular_part = -1/np.sqrt(2) * l_plus(li_q, mi_q) *\
                               KroneckerDelta(lf_q, li_q) *\
                               KroneckerDelta(mf_q, mi_q+1) +\
                               real_gaunt(lf_q, 2, li_q, mf_q, 1, mi_q,
                                          prec=16) *\
                               4*np.pi*factor*m_s
            elif transition_m == -1:
                angular_part = -1/np.sqrt(2) * l_minus(li_q, mi_q) *\
                               KroneckerDelta(lf_q, li_q) *\
                               KroneckerDelta(mf_q, mi_q-1) +\
                               real_gaunt(lf_q, 2, li_q, mf_q, -1, mi_q,
                                          prec=16) *\
                               4*np.pi*factor*m_s
            elif transition_m == 0:
                angular_part = (1./(4.*np.pi) * l_0(mi_q)) *\
                               KroneckerDelta(lf_q, li_q) *\
                               KroneckerDelta(mf_q, mi_q) +\
                               real_gaunt(lf_q, 2, li_q, mf_q, 0, mi_q,
                                          prec=16) *\
                               4*np.pi*factor*m_s/np.sqrt(5.*np.pi)
            if angular_part < 1e-10:
                continue
            integrand = [np.conj(alm_rf[(lf_q, mf_q)]) * alm_ri[(li_q, mi_q)] *
                         1./grid_r
                         for grid_r, alm_ri, alm_rf
                         in zip(grid, alm_initial, alm_final)]
            radial_part = trapezoidal_integration(integrand, grid)
            sum_lm += radial_part * angular_part
    return complex(1./c0 * sum_lm)


def single_tmi(final, operator, initial, grid, data):
    '''
    Calculate transition moment integrals <f|O|i>
    data is modified during the run and should be a dictionary
    '''
    if operator == "Q_E1":
        for t_m in range(-1, 1+1):
            tmi = f_qe1_i(final['coefficients'], initial['coefficients'],
                          grid, t_m)
            data[(final['band'], operator, t_m, initial['band'])] = tmi
    elif operator == "T_E2":
        for t_m in range(-2, 2+1):
            tmi = f_te2_i(final['coefficients'], initial['coefficients'],
                          grid, t_m)
            data[(final['band'], operator, t_m, initial['band'])] = tmi
    elif operator == "T_M1":
        for t_m in range(-1, 1+1):
            tmi = f_tm1_i(final['coefficients'], initial['coefficients'],
                          grid, t_m)
            data[final['band']][operator][t_m][initial['band']] = tmi
    return data


if __name__ == "__main__":
    print("    ELECTRON BRIDGE RATES")
    print("    =====================")
    R_NUCLEUS = 5.7557e-15 * 1./1e-10  # https://www-nds.iaea.org/radii/
    BAND_FINAL = 235
    BAND_INITIAL = 236
    BAND_MIN = 236
    BAND_MAX = 240

    print("::: Reading input .yaml files")
    INFO = read_yaml("./info.yaml")
    GRID = read_yaml("./r_grid.yaml")
    GRID.insert(0, R_NUCLEUS)
    INITIAL = merge_info(INFO[BAND_INITIAL-1],
                         extrapolate_nucleus(read_alm(BAND_INITIAL), GRID))
    FINAL = merge_info(INFO[BAND_FINAL-1],
                       extrapolate_nucleus(read_alm(BAND_FINAL), GRID))
    INTERMEDIATES = [merge_info(INFO[BAND-1],
                                extrapolate_nucleus(read_alm(BAND), GRID))
                     # if BAND != BAND_INITIAL] ?
                     for BAND in range(BAND_MIN, BAND_MAX+1)]
    INTERMEDIATES.append(FINAL)

    TMIS_ALL = {}
    print("::: Calculating Q_E1 Integrals")
    with Pool(cpu_count()) as p:
        print("=>  <f|Q_E1|u>")
        for T_M in range(-1, 1+1):
            print(f"q = {T_M}")
            ITERATOR = ((FINAL['coefficients'], INT['coefficients'],
                         GRID, T_M)
                        for INT in INTERMEDIATES)
            TMIS = p.starmap(f_qe1_i,
                             tqdm(ITERATOR, total=len(INTERMEDIATES)))
            for INT, TMI in zip(INTERMEDIATES, TMIS):
                TMIS_ALL[(FINAL["band"], "Q_E1", T_M, INT["band"])] = TMI
        print(":: <u|Q_E1|f>")
        for T_M in range(-1, 1+1):
            print(f"q = {T_M}")
            ITERATOR = ((INT['coefficients'], INITIAL['coefficients'],
                         GRID, T_M)
                        for INT in INTERMEDIATES)
            TMIS = p.starmap(f_qe1_i,
                             tqdm(ITERATOR, total=len(INTERMEDIATES)))
            for INT, TMI in zip(INTERMEDIATES, TMIS):
                TMIS_ALL[(INT["band"], "Q_E1", T_M, INITIAL["band"])] = TMI
    print("=>  Saving to TMIs.pckl")
    with open('TMIs.pckl', 'wb') as handle:
        pickle.dump(TMIS_ALL, handle, protocol=-1)
    print("::: Calculating T_E2 Integrals")
    with Pool(cpu_count()) as p:
        print("=>  <f|Q_E2|u>")
        for T_M in range(-2, 2+1):
            print(f"q = {T_M}")
            ITERATOR = ((FINAL['coefficients'], INT['coefficients'],
                         GRID, T_M)
                        for INT in INTERMEDIATES)
            TMIS = p.starmap(f_te2_i,
                             tqdm(ITERATOR, total=len(INTERMEDIATES)))
            for INT, TMI in zip(INTERMEDIATES, TMIS):
                TMIS_ALL[(FINAL["band"], "T_E2", T_M, INT["band"])] = TMI
        print(":: <u|Q_E1|f>")
        for T_M in range(-1, 1+1):
            print(f"q = {T_M}")
            ITERATOR = ((INT['coefficients'], INITIAL['coefficients'],
                         GRID, T_M)
                        for INT in INTERMEDIATES)
            TMIS = p.starmap(f_te2_i,
                             tqdm(ITERATOR, total=len(INTERMEDIATES)))
            for INT, TMI in zip(INTERMEDIATES, TMIS):
                TMIS_ALL[(INT["band"], "T_E2", T_M, INITIAL["band"])] = TMI
    print("=>  Saving to TMIs.pckl")
    with open('TMIs.pckl', 'wb') as handle:
        pickle.dump(TMIS_ALL, handle, protocol=-1)
    with Pool(cpu_count()) as p:
        print("=>  <f|T_M1|u>")
        for T_M in range(-1, 1+1):
            print(f"q = {T_M}")
            ITERATOR = ((FINAL['coefficients'], INT['coefficients'],
                         GRID, T_M)
                        for INT in INTERMEDIATES)
            TMIS = p.starmap(f_tm1_i,
                             tqdm(ITERATOR, total=len(INTERMEDIATES)))
            for INT, TMI in zip(INTERMEDIATES, TMIS):
                TMIS_ALL[(FINAL["band"], "Q_M1", T_M, INT["band"])] = TMI
        print(":: <u|T_M1|f>")
        for T_M in range(-1, 1+1):
            print(f"q = {T_M}")
            ITERATOR = ((INT['coefficients'], INITIAL['coefficients'],
                         GRID, T_M)
                        for INT in INTERMEDIATES)
            TMIS = p.starmap(f_tm1_i,
                             tqdm(ITERATOR, total=len(INTERMEDIATES)))
            for INT, TMI in zip(INTERMEDIATES, TMIS):
                TMIS_ALL[(INT["band"], "T_M1", T_M, INITIAL["band"])] = TMI
    print("::: Saving to TMIs.pckl")
    with open('TMIs.pckl', 'wb') as handle:
        pickle.dump(TMIS_ALL, handle, protocol=-1)
    # single_tmi(FINAL, "Q_E1", INITIAL, GRID, TMIS)

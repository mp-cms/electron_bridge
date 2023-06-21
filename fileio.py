'''
Share some common file io operations
'''
import pickle
import yaml


def read_yaml(fname):
    '''
    Read yaml file
    '''
    print(f"Reading file {fname}")
    with open(fname, "r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def read_pickle(fname):
    '''
    read pickle file
    '''
    print(f"Reading file {fname}")
    with open(fname, "rb") as stream:
        return pickle.load(stream)


def save_pickle(fname, data):
    '''
    Save to .pckl file
    '''
    print(f"Saving file {fname}")
    with open(fname, "wb") as stream:
        pickle.dump(data, stream, protocol=-1)


def read_info(fname, kpoint=1):
    '''
    Read info.yaml file
    Restructure such that a dict is returned with keys:
    (Band Index, K-Point, Spin)
    '''
    info = read_yaml(fname)
    data = {}
    print("Note: Only direct transitions allowed (for now?)")
    print(f"     Considering K-Point: {kpoint}")
    for band in info:
        #  Force equal k-transitions (for now?)
        if band["K-Point"] == kpoint:
            data[(band["Index"],
                  band["K-Point"],
                  band["Spin"])] = {"Energy": band["Energy"],
                                    "Occupation": band["Occupation"]}
    return data


def read_grid(fname):
    '''
    Read r_grid.yaml file
    Adds the nuclear radius to the grid
    This is a fix for extrapolation
    TODO: Confirm that this is sensible
    '''
    print("::: Reading grid .yaml file")
    a_0 = 0.529177210903  # bohr radius
    grid = read_yaml(fname)
    # r_nucleus = 5.7557e-15 * 1./1e-10  # https://www-nds.iaea.org/radii/
    # grid.insert(0, r_nucleus)
    grid = [r_val / a_0 for r_val in grid]
    return grid


def read_alm(fname):
    '''
    Reads a yaml file containing expansion coefficients
    at each point in the radial grid
    '''
    a_0 = 0.529177210903  # Bohr Radius
    data = read_yaml(fname)
    with open(fname, "r", encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=yaml.CSafeLoader)
    lmax = int(fname.split("L")[1].split(".")[0])
    # data loads as list of dicts but the keys and values are just strings
    alm_band = [{(l_q, m_q): complex(0., 0.)
                 for l_q in range(0, lmax+1) for m_q in range(-l_q, l_q+1)}
                for _ in range(len(data))]
    for r_i, alm_data in enumerate(data):
        for key_str in alm_data.keys():
            l_q = int(key_str.split(',')[0][1:])
            m_q = int(key_str.split(',')[1][:-1])
            val = alm_data[key_str].replace(" ", "").replace("im", "j")
            alm_band[r_i][(l_q, m_q)] = complex(val) * a_0**1.5
    return alm_band


def read_wfs(files, l_spin=True, kpoint=1):
    '''
    Reads the expansion coefficients from indir
    merges info and expansion coefficient entries
    '''
    print("::: Reading wavefunction .yaml files")
    print("Note: Only direct transitions allowed (for now?)")
    print(f"     Considering K-Point: {kpoint}")
    wfs = {}
    for file in files:
        descr = file.split("/")[-1].split("_")
        index = int(descr[0].split("B")[1])
        kpt = int(descr[1].split("K")[1])
        spin = descr[2]
        if spin == "UP":
            spin = 0.5
        else:
            spin = -0.5
        if kpoint == kpt:
            a_lm = read_alm(file)
            wfs[(index, kpoint, spin)] = a_lm
    if not l_spin:
        for (band, kpoint, spin) in wfs.copy():
            wfs[(band, kpoint, -0.5)] = wfs[(band, kpoint, 0.5)]
    return wfs

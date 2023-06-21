'''
This file contains all user defined options for the calculation
of electron bridge matrix elements
'''
CONFIG = {"lmax": 6,  # Maximum l-qunatum number from the basis set to use
          "Initial": {"Spin": 0.5,
                      "Index": 3,
                      "K-Point": 1},  # Initial state description
          "Final":   {"Spin": 0.5,
                      "Index": 2,
                      "K-Point": 1},  # Final state description
          # Directory contianing r_grid, info and wave function files
          "indir": "../wavefun_ylm",
          "l_spin": True,  # True: Spin-polarized, False: Not spin-polarized
          "K-Point": 1,
          }

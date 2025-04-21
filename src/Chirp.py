# File: Chirp.py
# Single-component Chirp
# By Eulogio G.H.

import numpy as np

class Chirp():

    def __init__(self, tt, center_frequency, chirp_rate):

        """
        Parameters:
        
        CF: center frequency
        CR: chirp rate
        """                
        self._CF = center_frequency
        self._CR = chirp_rate
        
        # single-component LFM signal        
        st = np.exp(1j * np.pi * (2*self._CF*tt + self._CR*(tt**2)))
        self._tt = np.asanyarray(tt)
        self._st = np.asanyarray(st)

    def get_CF(self):
        return self._CF
        
    def get_CR(self):
        return self._CR   
             
    def Axis(self):
        return self._tt
                     
    def Signal(self):
        return self._st

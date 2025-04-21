# File: frft_utils.py
# Computing methods for (simplified) FrFT
# By Eulogio G.H.

import numpy as np

from numpy.fft import fft,ifft,fftfreq,fftshift,ifftshift
from numpy import pi,exp,cos,sin,tan,sqrt

# Basic Functions   
    
def vec_operator(A): 
    """
    The vec operator transforms a mxn matrix into a nm-length vector
    by stacking the columns
    >>>>> vec_operator(A) <<<<<
    where  A:    mxn matrix
           v:    nm-length vector
    """  
    [m,n] = A.shape
    # From bidimensional to unidimensional  
    v = np.zeros(m*n)+1j*np.zeros(m*n)
    for c in range(n):
        for r in range(m):        
            v[int(r+m*c)] = A[r,c]          
    return v

def invvecT_operator(v,m): 
    """
    The inverse of vec operator transforms a nm-length vector
    into a mxn matrix and it is transposed   
    >>>>> invvecT_operator(v,m) <<<<<
    where  A:    transpose of mxn matrix invvec of v
           m:    length of column vector   
           v:    nm-length vector
    """  
    assert len(v) % m == 0
    n = len(v) // m            
    # From unimensional to bidimensional
    A = np.array([[0+1j*0 for c in range(n)] for r in range(m)])
    for c in range(n):
        for r in range(m):
            A[r,c] = v[int(r+m*c)]            
    return A.T
       
# DFrFT
    
def dfrft(tt,xt,a):
    """
    DFrFT as Chirp Convolutions
    >>>>> dfrft(tt,xt,a,b) <<<<<
    where  tt:   time axis for x(t)      
           xt:   DT signal
           a:    rotation angle factor, alpha = a*pi/2     
    """        
    assert len(tt) == len(xt)    
    if a % 2==0:
        if (a // 2) % 2 == 0:
            Xa = xt
        else:
            Xa = xt[::-1]
    else:     
        # Time axis normalization
        N = len(xt)
        T = (tt[-1]-tt[0])/(len(tt)-1) 
        S = T*sqrt(N)
        tt = tt/S        
        # Angle of rotation        
        alpha = a*pi/2     
        # Inner chirp modulation 
        rr = np.arange(-N,N)        
        gt = exp(1j*pi/N*(rr**2)/sin(alpha))        
        h1t = exp(-1j*pi*(tt**2)*tan(alpha/2))
        ht = h1t*xt
        # Amplitude, outer chirp modulation       
        A1a = sqrt(1-1j/tan(alpha))
        Aa = A1a*h1t/sqrt(N)
        # Zero padding method
        N_DFT = 2*N
        gt = np.append(gt,np.zeros(N_DFT-len(gt),dtype=complex))
        ht = np.append(ht,np.zeros(N_DFT-len(ht),dtype=complex))  
        # FFT-based fast Conv
        X1a = ifft(fft(gt)*fft(ht))
        No = N
        Xa = Aa*X1a[No:No+N]   
    return Xa
    
def dfrft1(tt,xt,a):
    """
    Chip convolution-based DFrFT using additivity property
    >>>>> dfrft1(tt,xt,a,b) <<<<<
    where  tt:   time axis for x(t)      
           xt:   DT signal
           a:    rotation angle factor, alpha = a*pi/2     
    """        
    assert len(tt) == len(xt)        
    N = len(xt)
    # Properties of FrFT     
    if abs(a) > 2:
        raise Warning("p must be in [-2,2]")     
    elif (a > 0 and a < 0.5) or (a > 1.5 and a < 2):    
        a = a-1                    
        xt = dfrft(tt,xt,1)                             
    elif (a > -0.5 and a < 0) or (a > -2 and a < -1.5):    
        a = a+1                   
        xt = dfrft(tt,xt,-1)                   
    # Computation of FrFT     
    if a == 0:
        Xa = xt
    elif abs(a) == 2:
        Xa = xt[::-1]    
    else:                     
        Xa = dfrft(tt,xt,a)                       
    return Xa  

# DLCT

def dlct(tt,xt,a):
    """
    A Discrete LCT as chirp convolutions
    >>>>> dlct(tt,xt,a,b) <<<<<
    where  tt:   time axis for x(t)      
           xt:   DT signal
           a:    rotation angle factor, alpha = a*pi/2     
    """        
    assert len(tt) == len(xt)    
    if a % 2==0:
        if (a // 2) % 2 == 0:
            Xa = xt
        else:
            Xa = xt[::-1]
    else:     
        # Time axis normalization
        N = len(xt)
        T = (tt[-1]-tt[0])/(len(tt)-1) 
        S = T*sqrt(N)
        tt = tt/S        
        # Angle of rotation        
        alpha = a*pi/2     
        # Inner chirp modulation 
        rr = np.arange(-N,N)        
        ht = exp(-1j*pi/N*(rr**2)*tan(alpha))        
        # Amplitude, outer chirp modulation       
        Aa = sqrt(1j*tan(alpha)/N)
        # Zero padding method
        N_DFT = 2*N
        xt = np.append(xt,np.zeros(N_DFT-len(xt),dtype=complex))
        ht = np.append(ht,np.zeros(N_DFT-len(ht),dtype=complex))  
        # FFT-based fast Conv
        X1a = ifft(fft(xt)*fft(ht))
        No = N
        Xa = Aa*X1a[No:No+N]   
    return Xa 

# DSmFrFT

def dsmfrft(tt,xt,a):
    """
    Discrete Simplified FrFT (DSmFrFT) decomposed into FFT algorithm
    >>>>> dsmfrft(tt,xt,a,b) <<<<<
    where  tt:   time axis for x(t)      
           xt:   DT signal
           a:    rotation angle factor, alpha = a*pi/2     
 
           args: b, level of approximation
    """        
    assert len(tt) == len(xt)        
    if a % 2==0:
        if (a // 2) % 2 == 0:
            Xa = xt
        else:
            Xa = xt[::-1]
    else:     
        # Time axis normalization
        N = len(xt)
        T = (tt[-1]-tt[0])/(len(tt)-1) 
        S = T*sqrt(N)
        tt = tt/S        
        # Angle of rotation        
        alpha = a*pi/2            
        # Single chirp modulation
        A1a = sqrt(1/1j)
        h1t = A1a*exp(1j*pi*(tt**2)/tan(alpha))
        xt = h1t*xt      
        # FFT
        X1a = fft(ifftshift(xt))
        Xa = fftshift(X1a)/sqrt(N)
    return Xa           

def dsmfrft1(tt,xt,a):
    """
    DSmFrFT with convertibility property 
    >>>>> asmfrft2(tt,xt,a,b) <<<<<
    where  tt:   time axis for x(t)      
           xt:   DT signal
           a:    rotation angle factor, alpha = a*pi/2     
 
           args: b, level of approximation
    """        
    assert len(tt) == len(xt)        
    N = len(xt)
    # Properties of FrFT     
    if abs(a) > 2:
        raise Warning("p must be in [-2,2]")    
    else:
        xs = np.copy(ifftshift(xt)) 
    Flag = False    
    if (a > 0 and a < 0.5) or (a > 1.5 and a < 2):   
        Flag = True                
        Xs = fft(xs)/sqrt(1j*N)                   
        xt = np.copy(fftshift(Xs))  
    elif (a > -0.5 and a < 0) or (a > -2 and a < -1.5):    
        Flag = True                    
        Xs = fft(xs)/sqrt(N)                   
        Xs = Xs[::-1]
        xt = np.copy(fftshift(Xs))   
    # Computation of FrFT     
    if a == 0:
        Xa = xt
    elif abs(a) == 2:
        Xa = xt[::-1]    
    else:    
        if Flag:                       
            Xa = dlct(tt,xt,a)                   
        else:
            Xa = dsmfrft(tt,xt,a)         
    return Xa    
          
# Local DSmFrFT        

def ldsmfrft(tt,xt,a,s,Q,*args):
    """
    Local DSmFrFT Based on FFT algorithm
    >>>>> ldsmfrft(tt,xt,a,s,q,b) <<<<<
    where  tt:   time axis for x(t)      
           xt:   DT signal
           a:    rotation angle factor, alpha = a*pi/2     
           s:    starting point
           Q:    Q continuos point
           args: b, level of approximation
    """        
    assert len(tt) == len(xt)     
    assert np.mod(len(tt),Q)==0
    if a % 2==0:
        if (a // 2) % 2 == 0:
            Xa = xt
        else:
            Xa = xt[::-1]
    else:     
        # Time axis normalization
        N = len(xt)
        T = (tt[-1]-tt[0])/(len(tt)-1) 
        S = T*sqrt(N)
        tt = tt/S        
        # Angle of rotation        
        alpha = a*pi/2        
        
        # CT input mapping
        P = N//Q        
        A = invvecT_operator(xt,P)    
        
        # The inner chirp term multiplying  
        B = np.array([[0.0+1j*0.0 for i in range(P)] for k in range(Q)])        
        for idp in range(P):
            p = idp - N//2
            for q in range(Q):
                B[q,idp] = exp(1j*pi*(2*p*q+P*q**2)/(Q*tan(alpha)))
                B[q,idp] = B[q,idp]*exp(-1j*2*pi*(q*s)/Q)
        B = np.multiply(A,B)         
                
        # The inner DFT by columns
        for c in range(P):
            B[:,c] = fft(B[:,c])
            
        # The twiddle-factor application  
        C = np.array([[0.0+1j*0.0 for i in range(P)] for k in range(Q)])        
        for idp in range(P):
            p = idp - N//2
            for r in range(Q):
                C[r,idp] = exp(1j*pi*(p**2)/(N*tan(alpha)))
                C[r,idp] = C[r,idp]*exp(-1j*2*pi*p*(r+s)/N)
        C = np.multiply(B,C)  
        
        # The outer DFT by row summation       
        X1a = np.sum(C,axis=1)        

        # Amplitude of SmFrFT      
        Aa =1/sqrt(1j*N)      
        Xa = Aa*X1a
    return Xa                        

# File: doa_utils.py
# DoA Estimation for linear Chirps in the FrFT domain
# By Eulogio G.H.

import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft,ifft,fftfreq,fftshift,ifftshift
from numpy import pi,exp,cos,sin,tan,sqrt

import math
import cv2
import scipy.signal as ss
from sklearn.linear_model import LinearRegression

import Chirp
import frft_utils as FrFT

# Chirp signal processing

def addnoise(st, SNR, axis=0, ddof=0):
    """
    Add white gaussian noise n(t) to the signal s(t)
    x(t)=s(t)+n(t)
    >>>>> addnoise(st,SNR,axis,ddof) <<<<<
    where  st:   sampled CT signal s(t)
           nt:   sampled CT noise n(t)
           SNR:  SNR in dB. Generate values from the uniform distribution  
           std = sqrt(mean(x)), where x = abs(a - a.mean())**2
    """          
    # The signal s(t) and its power, dBV^2
    st = np.asanyarray(st)     
    Ps = 10*math.log10(st.std(axis=axis, ddof=ddof)**2+st.mean(axis=axis)**2)
    # White noise generation
    y = np.random.randn(len(st)) # Gaussian noise n(t)
    y = (y-y.mean(axis=axis))/y.std(axis=axis, ddof=ddof)
                                 #(y - mean(y))/std(y)
    Pn = Ps - SNR                # noise power, dB
    Pn = 10 ** (Pn/10.0)         # noise power, V^2
    sigma = sqrt(Pn)             # noise RMS, V    
    nt = sigma*y;                # noise for given SNR
    xt = st + nt;                # signal + noise mixture   
    return xt

def steering_vector_ULA(tt,fc,mu,aoa,ula,c=3e8):
    """
    time‐variant steering vector  
    >>>>> steering_vector_ULA(tt,fc,mu,aoa,ula,c) <<<<<
    where  tt    : time axis for s(t)
           fc    : center frequency       
           mu    : chirp rate 
           aoa   : incident angles [rad] 
           ula   : N-length uniform linear array            
           c     : wave velocity
    """
    N = len(tt) # number of snapshots, time in [s]
    L = len(aoa) # number of sources, angle in [rad]
    M = len(ula) # Number of sensors   
    # Delay matrix
    tau = ula[:, None]*np.sin(aoa)/c    #M*L
    # Time-variant steering vector
    at = np.array([[[0.0+1j*0.0 for i in range(N)] for j in range(L)] for k in range(M)])
    for l in range(L):        
        for n in range(N):
            ft = fc[l] + mu[l]*tt[n]
            at[:,l,n] = np.exp(-1j*2*pi*ft*tau[:,l]+1j*pi*mu[l]*(tau[:,l])**2)   
    return at
    
def output_signal_ULA(tt,fc,mu,ula,aoa,snr,c=3e8):
    """
    Output signal of ULA, one target, one LFM signal is received
    >>>>> output_signal_ULA(tt,fc,mu,aoa,ula,snr,c) <<<<<
    where  tt    : time axis for x(t)
           fc    : center frequency    
           mu    : chirp rate
           ula   : N-length uniform linear array 
           snr   : SNR in dB
           c     : wave velocity
    """  
    assert len(fc)==len(aoa)
    assert len(mu)==len(aoa)    
    # Time-variant steering vector
    at = steering_vector_ULA(tt,fc,mu,aoa,ula,c)
    # Output signal
    N = len(tt)
    M = len(ula)   
    xt = np.array([[0.0+1j*0.0 for i in range(N)] for j in range(M)])     
    for m in range(M):
        if len(aoa)==1:
            # single-target case
            chirp = Chirp.Chirp(tt,fc,mu)
            smt = np.sum(at[m,:,:],axis=0)*chirp._st
        else: 
            # multi-target case
            #stt= np.tile(st, (len(aoa), 1))
            L = len(aoa)
            stt = np.array([[0.0+1j*0.0 for i in range(N)] for j in range(L)])
            for l in range(L):
                chirp = Chirp.Chirp(tt,fc[l],mu[l])                
                stt[l,:] = chirp._st
            smt = np.sum(at[m,:,:]*stt,axis=0)
        xt_Re = addnoise(smt.real, snr)
        xt_Im = addnoise(smt.imag, snr)
        xt[m,:] = xt_Re + 1j*xt_Im 
    return xt

# FrFT-based array signal processing    

def anti_diagonal_matrix(M):
    """
    An M x M anti-diagonal matrix
    >>>>> anti_diagonal_matrix(M) <<<<<
    where  M:  number of rows in the output
    """
    return np.fliplr(np.eye(M))
    
def FB_averaging(X):
    """
    Forward and Backward (FB) averaging 
    >>>>> FB_averaging(X) <<<<<
    where  X:   received signals in FrFT domain
           R:   averaged covariance matrix     
    """         
    Rxx = (X @ X.conj().T) / X.shape[1]
    
    J = anti_diagonal_matrix(X.shape[0])
    Y = J @ X.conj()
    Ryy = (Y @ Y.conj().T) / X.shape[1]
   
    R = (Rxx+Ryy)/2 
    return R
    
def FB_spatial_smooting(received_signals, L):
    """
    Spatial smooting and FB averaging   
    >>>>> FB_spatial_smooting(received_signals, L) <<<<<
    where  X:    received signals in FrFT domain
           L:    length of subarray     
           R:    averaged covariance matrix
    """     
    M = received_signals.shape[0] # number of sensors in ULA
    P = M-L+1 # Number of subarrays   
    # Smoothed covariance forward subarrays
    Rf = np.array([[0.0+1j*0.0 for i in range(L)] for j in range(L)])
    for p in range(P):
        Xf = received_signals[p:p+L,:]
        Rf = Rf+(Xf @ Xf.conj().T) # L-by-L matrix
    Rf = Rf / P        
    # Smoothed covariance of backward subarrays
    Rb = np.array([[0.0+1j*0.0 for i in range(L)] for j in range(L)])
    received_signals_reversed = received_signals[::-1,:]
    for p in range(P):
        Xb = received_signals_reversed[p:p+L,:]
        Rb = Rb+(Xb.conj() @ Xb.T) # L-by-L matrix
    Rb = Rb / P     
    R = (Rf + Rb) / 2      
    return R        
    
def peak_alignment(tt, Xa, mo, qOP, d, aOP, op, c=3e8):   
    """
    Peak alignment in the FrFT domain
    >>>>> peak_alignment(tt, Xa, mo, qOP, d, aOP, op, c) <<<<<
    where  tt:    time axis for x(t)  
           Xa:    FrFT of received signals
           mo:    Index of reference sensor, mo=np.where(ula == 0)[0][0] 
           qOP:   Index of peaks at reference sensor, -N/2<=q<N/2
           d:     array distance 
           aOP:   optimal fractional order         
    """ 
    M = Xa.shape[0] # number of sensors
    N = Xa.shape[1] # number of snapshots            
    alpha = aOP*pi/2
                      
    # Single chirp modulation
    fs = (len(tt)-1)/(tt[-1]-tt[0])    
    S = sqrt(N)/fs
    uu = tt/S      
    A1a = sqrt(1j/sin(alpha))
    h1t = A1a*exp(1j*pi*(uu**2)/tan(alpha))       
    
    if op=='dfrft':
        qD = fs*d*cos(alpha)/c
    elif op=='dsmfrft' or op=='ldsmfrft':
        qD = fs*d*cos(alpha)/c
        qD = qD/sin(alpha)
    else:
        raise Warning("Invalid operator!")        
    
    # Peak alignment loop
    Xao = 0 * np.copy(Xa) # output signals
    Xanp = np.copy(Xa) # no peak signals     
    aux_signals = np.copy(Xanp) # aux signals    
    for i, qOPi in enumerate(qOP): 
        iOP = int(qOPi) + N//2                        
        for r in range(1,M):
            m = np.mod(mo + r,M)  
            qMin = int(np.floor(iOP-abs((m-mo)*qD)))
            qMax = int(np.ceil(iOP+abs((m-mo)*qD)))
            if qMin==qMax:
                qMax=qMax+1        
            if qMin < 0:
                qMin = 0
            if qMax > N-1:
                qMax = N-1 
            aux_signals[m,:qMin]=0+1j*0
            aux_signals[m,qMax:]=0+1j*0      
        frft_peaks = np.array([np.argmax(np.abs(signal)) for signal in aux_signals])
        # Frequency variable scaling
        if op=='dsmfrft' or op=='ldsmfrft':
            qOPi_dfrft = round(qOPi * sin(alpha))
            iOP = int(qOPi_dfrft) + N//2   
        # Peak-position fittng    
        for m in range(M):
            if op=='dfrft':  
                Xao[m,iOP] += aux_signals[m,frft_peaks[m]]
            elif op=='dsmfrft' or op=='ldsmfrft':
                Xao[m,iOP] += aux_signals[m,frft_peaks[m]] * h1t[iOP]
            else:    
                raise Warning("Invalid operator!")                 
            Xanp[m,frft_peaks[m]] = 0+1j*0     
        aux_signals = np.copy(Xanp) # aux signals           
    return Xao    
    
def steering_vector_FrFT_domain(N, fs, qOP, aOP, angle, ula, op, c=3e8):
    """
    Steering vector in fractional Fourier domain
    >>>>> steering_vector_FrFT_domain(fs, qOP, aOP, angle, ula, c) <<<<<
    where  ula   : M-length uniform linear array 
           angle : an incident angle [rad] 
           aOP   : optimal fractional order
           qOP   : index of peaks
           fs    : sampling rate
           N     : number of snapshots
    """
    nd = fs * (ula/c) * sin(angle) # M-length vector
    nd2 = nd**2 # M-length vector    
    alphaOP = np.tile(aOP*pi/2,len(qOP)) # K-length vector   
    
    if op=='dfrft':
        Aux = 2 * nd[:, None] * qOP # M-by-K matrix
        Aux = (pi/N) * Aux * np.tile(np.sin(alphaOP),(len(ula),1))
    elif op=='dsmfrft' or op=='ldsmfrft':
        Aux = 2 * nd[:, None] * qOP # M-by-K matrix
        Aux = (pi/N) * Aux

    steering_vector = np.exp(-1j * Aux)
    return steering_vector     

def eigenvalue_pairing(eigenvalues, tt, qOP, aOP, d, op, c=3e8):
    """
    Eigenvalue pairing for direction finding  
    >>>>> eigenvalue_pairing(eigenvalues, tt, qOP, aOP, ula, c) <<<<<
    where  tt     : time axis for x(t)
           qOP    : index of peaks
           aOP    : optimal fractional order           
           d      : inner spacing of ULA
    """ 
    N = len(tt)
    fs = (N-1)/(tt[-1]-tt[0])
    aux0 = 2 * pi * (fs/N) * d/c  
    # Rolling peak positions
    r_list = [i // 2 * (-1) ** i for i in range(2*len(qOP))]    
    flag = False
    for r in r_list[1:]:
        qOP_roll = np.roll(qOP,r)
        if op=='dfrft':      
            aux = aux0 * qOP_roll * sin(aOP*pi/2)  
            aux = -np.angle(eigenvalues) / aux
        elif op=='dsmfrft' or op=='ldsmfrft':
            aux = aux0 * qOP_roll
            aux = -np.angle(eigenvalues) / aux
        elif op=='dft':
            aux = aux0 * qOP_roll
            aux = -np.angle(eigenvalues) / aux        
        else:
            raise Warning("Invalid operator!")    
        if all(np.abs(ax)<=1 for ax in aux):
            flag = True
            break   
    # Ensure argument of arcsin        
    if not flag:        
        ax_list = []
        for ax in aux:
            if np.abs(ax)>1:
                ax = np.sign(ax)
            ax_list.append(ax)
        aux = np.array(ax_list)
    # arcsin    
    angles = np.arcsin(aux)    
    return angles  
    
def music_frft(tt, frft_signals, mo, qOP, d, aOP, L, i, op, align=True, c=3e8, num_points=360):   
    """
    MUSIC algorithm using FrFT for chirp signal DoA estimation
    >>>>> music_frft() <<<<<
    where  tt:    time axis for x(t)      
           Xa:   Received signals in FrFT domain
           mo:    Index of reference sensor, mo=np.where(ula == 0)[0][0] 
           qOP:   Index of peaks at reference sensor, -N/2<=q<N/2
           d:     array distance 
           aOP:   optimal fractional order  
           L:     subarray size
           i:     peak index of i-th target
    """     
    num_sources = len(qOP)
    num_sensors = frft_signals.shape[0]  
        
    if align:    
        frft_signals = peak_alignment(tt, frft_signals, mo, qOP, d, aOP, op, c)   
        
    M = frft_signals.shape[0] # number of sensors
    N = frft_signals.shape[1] # number of snapshots
    P = M-L+1 # Number of subarrays           

    if L==M:
        # Averaged covariance matrix
        R = FB_averaging(frft_signals) # M-by-M matrix
    elif L>1 and L<M:    
        # Smoothed covariance matrix
        R = FB_spatial_smooting(frft_signals, L) # L-by-L matrix 
    else:
        # Basic form
        Xa = frft_signals
        R = Xa @ Xa.conj().T # M-by-M matrix        
           
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    noise_subspace = eigenvectors[:, :-num_sources]  # Smallest eigenvalues correspond to noise      
    
    # MUSIC spectrum
    angles = np.linspace(-90, 90, num_points)
    ula = d*np.arange(L)  
    
    music_spectrum = []
    fs = (len(tt)-1)/(tt[-1]-tt[0])
    for angle in angles:
        steering_vector = steering_vector_FrFT_domain(N, fs,qOP,aOP,np.radians(angle),ula,op,c)
        projection = np.abs(steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector)
        music_spectrum.append(1 / projection[i, i])
    music_spectrum = np.array(music_spectrum)
    
    return angles, music_spectrum                 
    
def esprit_frft(tt, frft_signals, mo, qOP, d, aOP, L, op, align=True, c=3e8):   
    """
    Esprit algorithm using FrFT for chirp signal DoA estimation
    >>>>> esprit_frft() <<<<<
    where  tt:   time axis for x(t)      
           Xa:   Received signals in FrFT domain
           mo:   Index of reference sensor, mo=np.where(ula == 0)[0][0] 
           qOP:  Index of peaks at reference sensor, -N/2<=q<N/2
           d:    array distance 
           aOP:  optimal fractional order  
           L:    subarray size
    """     
    num_sources = len(qOP)
    num_sensors = frft_signals.shape[0]
    
    if align:    
        frft_signals = peak_alignment(tt, frft_signals, mo, qOP, d, aOP, op, c) 

    M = frft_signals.shape[0] # number of sensors
    N = frft_signals.shape[1] # number of snapshots
    P = M-L+1 # Number of subarrays      
        
    if L==M:
        # Averaged covariance matrix
        R = FB_averaging(frft_signals) # M-by-M matrix
    elif L>1 and L<M:    
        # Smoothed covariance matrix
        R = FB_spatial_smooting(frft_signals, L) # L-by-L matrix 
    else:
        # Basic form
        Xa = frft_signals
        R = Xa @ Xa.conj().T # M-by-M matrix 
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    signal_subspace = eigenvectors[:, -num_sources:]  # largest eigenvalues
    
    # Partition the signal subspace into two subarrays
    Phi = signal_subspace[:-1, :]  # First subarray
    Psi = signal_subspace[1:, :]  # Second subarray
    
    # Estimate the rotational matrix
    rotational_matrix = np.linalg.pinv(Phi) @ Psi
    
    # Eigen decomposition of the rotational matrix
    eigenvalues = np.linalg.eigvals(rotational_matrix)
       
    # Estimate DOAs       
    angles = eigenvalue_pairing(np.sort(eigenvalues), tt, qOP, aOP, d, op, c)
    return angles

# Multi-line fitting

def LSM(x,y):    
    """
    Least square method
    >>>>> LSM(x,y) <<<<<
    where  x: Sequence of x-values, ULA.
           y: Sequence of y-values, index of peaks.         
    """     
    # assemble matrix A
    A = np.vstack([np.ones(len(x)), x]).T
    # turn y into a column vector
    y = y[:, np.newaxis]    
    # Direct least square regression
    B = np.dot(A.T,A)        
    # avoid singular matrix, np.linalg.det(B) =0 
    if len(x)<2:        
        return [float("NaN"),float("NaN")]    
    try:
        C = np.dot((np.dot(np.linalg.inv(B),A.T)),y)
        # slope, intercept
        return np.hstack(C)[::-1].tolist()
    except:
        return [float("NaN"),float("NaN")]  
    
def doa_line_fitting(fs,d,slope,aOP,op,c=3e8):
    """
    DoA using LSM and delay formula
    >>>>> doa_line_fitting(fs,d,slope,aOP,op,c) <<<<<
    where  slope:  slope of the fitting curve
           theta:  DoA
    """     
    alpha = aOP*pi/2
    # Constrained LSM method
    if op=='dfrft':        
        aux = slope*c/(fs*d*cos(alpha))
    elif op=='dsmfrft' or op == 'ldsmfrft':   
        aux = slope*c/(fs*d*cos(alpha))
        aux = aux*sin(alpha) # sin() by single DFT constraint
    else:
        raise Warning("Invalid operator!")         
    if abs(aux)>1:
        aux = np.sign(aux)
    theta = np.arcsin(aux)
    return theta    
    
def multi_peaks_ULA(tt, Xa, ula, a, qOP, op, c=3e8):
    """
    Detecting multi peaks in FrFT domain
    >>>>> multi_peaks_ULA(Xa,tt,ula,a,qOP,op,c) <<<<<
    where  tt:   time axis for X(t)      
           Xa:   Received signals in FrFT domain
           ula:  uniform linear array
           a:    optimal fractional order 
           qOP:  index (list) of u at reference sensor
    """     
    [M,N] = Xa.shape    
    Xa = np.abs(Xa)
    mo = np.where(ula == 0)[0][0]
     
    # outlier parameters
    fs = (len(tt)-1)/(tt[-1]-tt[0])
    alpha = a*pi/2
    d = (ula[-1]-ula[0])/(len(ula)-1)
    
    if op == 'dfrft':
        qD = fs*d*cos(alpha)/c
    elif op == 'dsmfrft' or op == 'ldsmfrft':
        qD = fs*d*cos(alpha)/c
        qD = qD/sin(alpha)
    else:
        raise Warning("Invalid method!")            
                
    uidx = [] # peak indices
    ULAi = [] # uniform linear array indices   
    q_list = np.array([int(q) + N//2 for q in qOP]) 
    for q in q_list:       
        uidx.append(q)
        ULAi.append(0)                 
        for i in range(1,M):
            m = np.mod(mo + i,M)           
            # outlier bounds
            qMin = int(np.floor(q-abs((m-mo)*qD)))
            qMax = int(np.ceil(q+abs((m-mo)*qD)))
            if qMin==qMax:
                qMax=qMax+1        
            if qMin < 0:
                qMin = 0
            if qMax > N-1:
                qMax = N-1    
            # outlier detection methods     
            iux_m = qMin + np.argmax(Xa[m,qMin:qMax])                
            uidx.append(iux_m)
            ULAi.append(m-mo)     
            Xa[m,iux_m] = 0           
    return np.array(ULAi),np.array(uidx) 

def fit_k_lines(x, y, K):
    """
    Fits K linear segments to the data (x, y).
    >>>>> fit_k_lines(x, y, K) <<<<<
    where  x: Sequence of x-values.
           y: Sequence of y-values.
           K: Number of lines to fit.  
    Returns:
        models (list): List of (slope, intercept) tuples for each segment.
        segments (list): List of (x_segment, y_segment) for visualization.
    """
    # Ensure x, y are NumPy arrays and sort by x
    x, y = np.array(x), np.array(y)
    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]
    
    # Split data into K segments
    N = len(x)
    segment_size = N // K
    models = []
    segments = []

    for i in range(K):
        start = i * segment_size
        end = (i + 1) * segment_size if i < K - 1 else N  # Ensure last segment includes remaining points
        
        x_segment = x[start:end].reshape(-1, 1)
        y_segment = y[start:end]
        
        # Fit a linear model to the segment
        model = LinearRegression().fit(x_segment, y_segment)
        slope, intercept = model.coef_[0], model.intercept_
        
        models.append((slope, intercept))
        segments.append((x_segment.flatten(), y_segment))

    return models, segments    
    
def plot_results_line_fitting(x, y, models, segments):
    """
    Plot results of line fitting
    >>>>> hough_line_fitting(x, y, models, segments) <<<<<
    where  x: Sequence of x-values.
           y: Sequence of y-values.
    """     
    plt.figure(figsize = (5,4))
    plt.scatter(x, y, label="Peaks", s=10, alpha=0.5)
    for i, (x_seg, y_seg) in enumerate(segments):
        slope, intercept = models[i]
        plt.plot(x_seg, slope * x_seg + intercept, label=f"Line {i+1}")

    plt.legend(loc='upper left', bbox_to_anchor=(0.20, 0.34))
    plt.xlabel("Index of u, $q$")
    plt.ylabel("ULA, $m-m_0$")
    plt.title(f"Piecewise Linear Fit with {i+1} Segments")
    plt.grid(linestyle= "--") 
    plt.show()    
    
def hough_line_fitting(x, y, K):
    """
    Detecting lines in Hough space
    >>>>> hough_line_fitting(x, y, K) <<<<<
    where  x: Sequence of x-values.
           y: Sequence of y-values.
           K: Number of lines to fit.
    """ 
    # Convert data to an image-like space for Hough Transform
    img = np.zeros((np.max(y)-np.min(y)+1, np.max(x)-np.min(x)+1), dtype=np.uint8)
    for i in range(len(x)):
        img[np.max(y)-y[i], x[i]-np.min(x)] = 255
        
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(img, 1, pi / 180, threshold=20, minLineLength=2, maxLineGap=5)
    
    # Line model
    if lines is None:
        print("No lines detected")
        return []    
    else: 
        # Detect K lines by peaks
        if lines.shape[0]>K:
            # Convert to list and sort by line length (descending)
            lines = [line[0] for line in lines]  # Flatten array
            lines.sort(key=lambda line: np.linalg.norm((line[0] - line[2], line[1] - line[3])), reverse=True)            
            lines = np.array([[lines[k]] for k in range(K)])
        #  Compute slope (m) and intercept (b)   
        models = []
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Extract line points            
            if x2 - x1 != 0:  # Avoid division by zero
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
            else: 
                m = np.inf # Vertical line (infinite slope)
                b = x1  # For vertical lines, store x-intercept instead
            models.append((m, b))  
            
    return models, lines   
    
def plot_results_detected_lines(x, y, models, lines):
    """
    Plot results of detected lines 
    >>>>>  plot_results_detected_lines(x, y, models, segments) <<<<<
    where  x: Sequence of x-values.
           y: Sequence of y-values.
    """
    plt.figure(figsize = (5,4))
    plt.scatter(x, y, label="Peaks", s=10, alpha=0.5)
    for i, line in enumerate(lines):
        slope, intercept = models[i]
        x1, y1, x2, y2 = line[0]
        x_seg = np.linspace(x1,x2,100)
        plt.plot(x_seg+np.min(x), np.max(y)-slope * x_seg - intercept, label=f"Line {i+1}")
    #plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(0.20, 0.34))
    plt.xlabel("Index of u, $q$")
    plt.ylabel("ULA, $m-m_0$")
    plt.title(f"Detected {i+1} Lines (Hough Transform)")
    plt.grid(linestyle= "--")            
    plt.show()            

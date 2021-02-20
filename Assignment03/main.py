
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy as sp

def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    a=amplitude
    w=2*np.pi*frequency_Hz
    T = 1 / sampling_rate_Hz
    t=np.linspace(0, length_secs-T,int(sampling_rate_Hz*length_secs))
    # print('t', len(t))
    # print('t', t, len(t))
    x=a*np.sin(w*t+phase_radians)
    # print('x', x, len(x))
    result=[x,t]
    return result

def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    T = 1 / sampling_rate_Hz
    t1 = np.linspace(0, length_secs-T, int(sampling_rate_Hz * length_secs))
    print('t1',len(t1))
    lent=len(t1)
    x = np.zeros(shape=(lent))
    sin= np.zeros(shape=(11,lent))
    e=0
    for j in range(1,11):
        sin[e,:],t=generateSinusoidal(amplitude, sampling_rate_Hz,(2*j-1)*frequency_Hz, length_secs, phase_radians)
        sin[e,:]=sin[e,:]*float((4/((2*j-1)*np.pi)))
        e=e+1
    for i in range(lent):
        x[i]=sum(sin[:,i])
        # x[i]=x[i]+(4/(float((2*j-1))*np.pi))*sin[i]
        # x[i]=sum(v)

    return x,t

def computeSpectrum(x, sample_rate_Hz):

    X = np.fft.fft(x)
    lenx=len(x)
    freqRes=sample_rate_Hz/lenx
    f=np.zeros(shape=lenx)
    f[0]=0
    for i in range(1,lenx):#niq limit apply beshe
        f[i]=f[i-1]+freqRes
    Xmag = np.zeros(shape=lenx)
    for i in range(lenx):
        Xmag[i]=2*np.sqrt(X.real[i]**2+X.imag[i]**2)*2/sample_rate_Hz
    # Xphase = np.zeros(shape=lenx)

    # for i in range(lenx):
    #     Xphase[i]=np.angle(X[i])
        # Xphase[i] = np.arctan2(X.real[i], X.imag[i])
        # Xphase[i] = np.arctan2(X.imag[i],X.real[i])
        # Xphase[i] = np.arctan(X.imag[i]/ X.real[i])
    Xphase= np.angle(X)
    # print("XPhaseeeeeeeeeee: ", Xphase)
    returnVal=int((lenx/2))
    print('fft',max(Xmag))
    return f[0:returnVal],Xmag[0:returnVal],Xphase[0:returnVal],X.real[0:returnVal]/sample_rate_Hz,X.imag[0:returnVal]/sample_rate_Hz

def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    lenx=len(x)
    N=int(lenx/hop_size)+1
    x2=np.zeros(shape=(lenx+block_size))
    for i in range(lenx):
        x2[i]=x[i]
    X= np.zeros(shape=(block_size,N))
    e=0
    for i in range(0,lenx,hop_size):
        for j in range(block_size):
            X[j,e]=x2[i+j]
        e=e+1
    T = 1 / sample_rate_Hz
    t=np.zeros(shape=(N))
    # t=np.linspace(0, length_secs-T,int(sample_rate_Hz*length_secs))
    for i in range(N):
        t[i]=i*hop_size*T
    print('block',np.amax(X))
    return X,t

def mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type):
    winSize=2048
    winRec=np.zeros(shape=(winSize))
    winHan=np.zeros(shape=(winSize))
    lenx = len(x)
    y,t=generateBlocks(x, sampling_rate_Hz, block_size, hop_size)
    N = int(lenx / hop_size) + 1
    for i in range(0,winSize):
        t1=float(i)/float(winSize)
        t1=t1-0.5
        winRec[i]=1
        winHan[i]=float(25/46 + 21/46*np.cos(2*np.pi*t1))
    if window_type=='rect':
        for i in range(N):
            y[:,i]=np.multiply(winRec,y[:,i])

    elif window_type=='hann':
        for i in range(N):
            y[:,i]=np.multiply(winHan,y[:,i])

    print('y',np.amax(y))
    X = np.zeros(shape=(int(block_size/2), N))
    for i in range(N):
        f,Xmag,Xphase,r,im=computeSpectrum(y[:,i], sampling_rate_Hz)
        # f, Xmag, Xphase, r, im = computeSpectrum(y[:, i], 1.5*block_size)
        print('Xmag spec',max(Xmag))
        for j in range(int(block_size/2)):
            X[j,i]=Xmag[j]
    return f,t,X


def plotSpecgram(freq_vector, time_vector, magnitude_spectrogram):
    print('plotspec',len(time_vector))
    if len(freq_vector) < 2 or len(time_vector) < 2:
        return

    Mag = np.flipud(magnitude_spectrogram)
    Mag = 20 * np.log10(Mag)
    pad_xextent = (time_vector[1] - time_vector[0]) / 2
    xmin = np.min(time_vector) - pad_xextent
    xmax = np.max(time_vector) + pad_xextent
    extent = xmin, xmax, freq_vector[0], freq_vector[-1]
    return Mag,extent
    im = plt.imshow(Mag, None, extent=extent,origin='upper')
    plt.axis('auto')
    plt.show()


#Main

if __name__ == '__main__':

    # # #Q1
    x,t=generateSinusoidal(1,44100,400,0.5,np.pi/2)
    print('x',len(x))
    lenx=len(x)
    lenSec=0.5
    lenPltT=0.005
    lenReqTime=int(lenx/(lenSec/lenPltT))
    plt.plot(t[0:lenReqTime],x[0:lenReqTime])
    plt.title("sine wave")
    plt.xlabel("time(s)")
    plt.ylabel("magnitude")
    plt.show()

    # #Q2
    x2,t2=generateSquare(1,44100,400,0.5,0)
    lenx2 = len(x2)
    lenSec=0.5
    lenPltT=0.005
    lenReqTime=int(lenx2/(lenSec/lenPltT))
    print(max(x2))
    plt.plot(t2[0:lenReqTime],x2[0:lenReqTime])
    plt.title("Square wave")
    plt.xlabel("time(s)")
    plt.ylabel("magnitude")
    plt.show()

    #Q3
    f,XAbs,XPhase,XRe,XIm = computeSpectrum(x, 44100)
    print('fft1',max(XAbs))
    fig, axs = plt.subplots(2)
    axs[0].plot(f, XAbs)
    axs[0].set_title('fft of Sine wave')
    axs[0].set(xlabel='Frequency(Hz)', ylabel='magnitude')
    axs[1].plot(f, XPhase)
    axs[1].set(xlabel='Frequency(Hz)', ylabel='Phase')
    plt.show()

    f2,XAbs2,XPhase2,XRe2,XIm2 = computeSpectrum(x2, 44100)
    print('fft1',max(XAbs2))
    fig, axs = plt.subplots(2)
    axs[0].plot(f2, XAbs2)
    axs[0].set_title('fft of square wave')
    axs[0].set(xlabel='Frequency', ylabel='magnitude')
    axs[1].plot(f, XPhase)
    axs[1].set(xlabel='Frequency', ylabel='Phase')
    plt.show()

    #Q4
    x2, t2 = generateSquare(1, 44100, 400, 0.5, 0)
    f,t3,X=mySpecgram(x2, 2048,1024, 44100, 'rect')
    f2, t32, X2 = mySpecgram(x2, 2048, 1024, 44100, 'hann')
    Mag,extent=plotSpecgram(f2, t32, X2)
    im = plt.imshow(Mag, None, extent=extent,origin='upper')
    plt.axis('auto')
    plt.title('rectangular windowing specgram')
    plt.xlabel("time(s)")
    plt.ylabel("mFrequency(Hz)")
    plt.show()
    Mag2,extent2=plotSpecgram(f, t3, X)
    im = plt.imshow(Mag2, None, extent=extent2,origin='upper')
    plt.axis('auto')
    plt.title('Hanning windowing specgram')
    plt.xlabel("time(s)")
    plt.ylabel("mFrequency(Hz)")
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

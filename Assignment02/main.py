
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy as sp
import statistics as st
import time

def loadSoundFile(filename):
    samplerate, data = wavfile.read(filename)
    return data

 # convolved length = 100+200=300 (if len(x)=200 and len(h)=100)
def myTimeConv(x, h):
    lenx = len(x)
    lenh = len(h)
    e=lenx-1
    xconv= np.zeros(shape=((lenx)))
    for i in range(lenx):
        xconv[i]=x[e]
        e=e-1
    print(xconv)
    b = np.zeros(shape=((max(lenx, lenh)) + (2 * (min(lenx, lenh)))))
    j = 0

    for i in range(len(b)):
        if lenx < i < lenx + lenh:
            b[i] = h[j]
            j = j + 1
    r = np.zeros(shape=((lenx + lenh-1)))
    for n in range(lenx + lenh-1):
        r[n]=np.dot(x, b[n:n+lenx])

    return r

def mean(x):
    sumX=np.sum(x)
    meanX=sumX/len(x)
    return meanX

def meanDifference(x,h):
    meanX=mean(x)
    meanH=mean(h)
    return meanX-meanH

def meanAbsDifference(x,h):
    d= np.zeros(shape=((len(x))))
    for i in range(len(x)):
        d[i]=abs(x[i]-h[i])
    mabs=sum(d)/(len(x)**2)

    return mabs

def standardDeviation(x):
    dev=0
    meanX=mean(x)
    for i in range(len(x)):
        dev=dev+(x[i]-meanX)**2
    deviation=np.sqrt(dev/len(x))
    return deviation

def DevDiff(x,h):
    d= np.zeros(shape=((len(x))))
    for i in range(len(x)):
        d[i]=abs(x[i]-h[i])
    return d

def timeCalc(x,h):
    t1=time.time()
    myTimeConv(x,h)
    t2=time.time()
    tMine = t2-t1
    t1 = time.time()
    np.convolve(x,h)
    t2=time.time()
    tPy = t2-t1
    t=[tMine,tPy]
    return t

def CompareConv(x, h):
    m= meanDifference(x,h)
    mabs=meanAbsDifference(x,h)
    stdev=standardDeviation(DevDiff(x,h))
    t=timeCalc(x,h)
    result=[m,mabs,stdev,t]
    return result

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Q1
    x=np.zeros(shape=200)
    lenx=len(x)
    for i in range (lenx):
        x[i]=1
    h=np.zeros(shape=51)
    lenh=len(h)
    j=0.04
    for i in range (lenh):
        if i<=25 :
            h[i]=i*j
        else:
            h[i]=1-(i-25)*j

    # convolved length = 100+200=300 (if len(x)=200 and len(h)=100)

    y_time = myTimeConv(h,x)
    print('leny',len(y_time))
    plt.plot(y_time)
    plt.title("myTimeConv function Q1")
    plt.xlabel("samples")
    plt.ylabel("convolution magnitude")
    plt.show()

    # Q2
    x=loadSoundFile('./impulse-response.wav')
    lenx = len(x)
    print('x',x,lenx)
    x1=x/max(x)
    # plt.plot(x)
    # plt.title("x")
    # plt.show()
    h=loadSoundFile('./piano.wav')
    lenh = len(h)
    h1 = h / max(h)
    print('h',h,lenh)

    y1 = myTimeConv(x1,h1)
    plt.plot(y1)
    plt.title("myTimeConv function")
    plt.xlabel("samples")
    plt.ylabel("normalized convolution magnitude")
    plt.show()
    y=np.convolve(x1,h1)
    # print('lengthy',len(y))
    # print('lengthy1',len(y1))
    plt.plot(y)
    plt.title("Python convolve funcyion")
    plt.xlabel("samples")
    plt.ylabel("normalized convolution magnitude")
    plt.show()
    result=CompareConv(x1, h1)
    print(result)

    f = open("./Results/Q2/CompareConv-report.txt", "w")
    f.write("results of the comparison: ")
    for i in range (len(result)):
        f.write("\n")
        f.write(str(result[i]))
    f.close()

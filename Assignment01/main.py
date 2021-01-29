
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

def loadSoundFile(filename):
    samplerate, data = wavfile.read(filename)
    # return left channel
    return data[:,0]
def crossCorr(x,y):
    lenx = len(x)
    leny = len(y)

    b = np.zeros(shape=((max(lenx, leny)) + 2 * (min(lenx, leny))))
    j = 0

    for i in range(len(b)):
        if lenx < i < lenx + leny:
            b[i] = y[j]
            j = j + 1
    r = np.zeros(shape=((lenx + leny)))
    for n in range(lenx + leny):
        r[n]=np.dot(x, b[n:n+lenx])


    return r/max(r)

def findSnarePosition(snareFilename, drumloopFilename):
    x=loadSoundFile(snareFilename)
    y=loadSoundFile(drumloopFilename)
    lenx=len(x)
    pos1= {}
    pos2=[]
    r=crossCorr(x, y)
    maxr=max(r)
    print(maxr)
    for i in range(len(r)):
       if r[i]>=maxr-0.1:
         pos1.update({i:r[i]})
         pos2.append(i-lenx)
    return pos1,pos2
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
# Q1
    x=loadSoundFile('./download.wav')
    lenx = len(x)
    print('x',x,lenx)
    y=loadSoundFile('./drum_loop.wav')
    leny = len(y)
    print('y',y,leny)
    Z=crossCorr(x,y)
    plt.plot(Z)
    plt.title("Correlation between two wave files")
    plt.xlabel("samples")
    plt.ylabel("normalized correlation magnitude")
    plt.show()
#Q2
    pos,pos1=findSnarePosition('./download.wav', './drum_loop.wav')
    print('samples on the correlation plot: ',pos)
    print('samples on the drumloop file: ',pos1)
    # f = open("./Assignment01/Results/Q2/02-snareLocation.txt", "a")
    # f.write("sample location on the drumloop file: ")
    # f.close()
